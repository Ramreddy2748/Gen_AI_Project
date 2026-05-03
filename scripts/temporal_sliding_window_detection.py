import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_POSE_MANIFEST = Path("data/processed/poses/train_pose_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("results/sliding_window_temporal")


def parse_frame_index(frame_name: str) -> int:
    digits = "".join(ch for ch in Path(frame_name).stem if ch.isdigit())
    return int(digits) if digits else 0


def load_pose_frames(pose_json_path: Path) -> List[Dict[str, object]]:
    with open(pose_json_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    frames: List[Dict[str, object]] = []
    for item in payload.get("frames", []):
        frame_name = str(item.get("frame", ""))
        features = item.get("features") if item.get("pose_detected") else None
        frames.append(
            {
                "frame": frame_name,
                "frame_index": parse_frame_index(frame_name),
                "pose_detected": bool(item.get("pose_detected", False)),
                "features": features if isinstance(features, dict) else {},
            }
        )

    return sorted(frames, key=lambda row: int(row["frame_index"]))


def sliding_windows(
    frames: List[Dict[str, object]],
    window_size: int,
    stride: int,
) -> Iterable[Tuple[int, List[Dict[str, object]]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(frames) < window_size:
        return

    for start in range(0, len(frames) - window_size + 1, stride):
        yield start, frames[start : start + window_size]


def feature_value(frame: Dict[str, object], name: str, default: float = 0.0) -> float:
    features = frame.get("features")
    if not isinstance(features, dict):
        return default
    value = features.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def temporal_motion_score(window: List[Dict[str, object]]) -> float:
    if len(window) < 2:
        return 0.0

    scores: List[float] = []
    for previous, current in zip(window, window[1:]):
        hip_drop = feature_value(current, "hip_y") - feature_value(previous, "hip_y")
        nose_drop = feature_value(current, "nose_y") - feature_value(previous, "nose_y")
        shoulder_drop = feature_value(current, "shoulder_y") - feature_value(previous, "shoulder_y")
        torso_change = abs(
            feature_value(current, "torso_vertical_diff")
            - feature_value(previous, "torso_vertical_diff")
        )
        scores.append(max(0.0, hip_drop) + max(0.0, nose_drop) + max(0.0, shoulder_drop) + torso_change)
    return sum(scores) / len(scores)


def classify_window(
    window: List[Dict[str, object]],
    motion_threshold: float,
    ground_torso_threshold: float,
    bend_torso_threshold: float,
) -> Dict[str, object]:
    detected = [frame for frame in window if frame.get("pose_detected")]
    if not detected:
        return {
            "prediction": "no_fall",
            "confidence": "low",
            "motion_score": 0.0,
            "reason": "No pose detected in the window.",
            "hallucination_pattern": "missing_pose",
        }

    torso_values = [feature_value(frame, "torso_vertical_diff", 1.0) for frame in detected]
    hip_values = [feature_value(frame, "hip_y") for frame in detected]
    nose_values = [feature_value(frame, "nose_y") for frame in detected]
    motion_score = temporal_motion_score(detected)

    min_torso = min(torso_values)
    max_torso = max(torso_values)
    hip_drop = hip_values[-1] - hip_values[0]
    nose_drop = nose_values[-1] - nose_values[0]
    ends_low = min_torso <= ground_torso_threshold
    rapid_descent = motion_score >= motion_threshold or hip_drop >= 0.18 or nose_drop >= 0.18
    bending_only = min_torso <= bend_torso_threshold and not rapid_descent

    if ends_low and rapid_descent:
        prediction = "fall"
        confidence = "high" if motion_score >= motion_threshold * 1.5 else "medium"
        reason = "Temporal pose shows descent plus a low or horizontal body posture."
        hallucination_pattern = "fall_sequence"
    else:
        prediction = "no_fall"
        confidence = "medium" if bending_only else "high"
        reason = "Window lacks sustained descent into a low or horizontal posture."
        hallucination_pattern = "bending_only" if bending_only else "stable_or_ambiguous"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "motion_score": round(motion_score, 4),
        "min_torso_vertical_diff": round(min_torso, 4),
        "max_torso_vertical_diff": round(max_torso, 4),
        "hip_drop": round(hip_drop, 4),
        "nose_drop": round(nose_drop, 4),
        "reason": reason,
        "hallucination_pattern": hallucination_pattern,
    }


def aggregate_video(
    window_predictions: List[Dict[str, object]],
    min_fall_windows: int,
) -> Dict[str, object]:
    fall_windows = [
        row for row in window_predictions
        if row.get("prediction") == "fall"
    ]
    no_fall_windows = [
        row for row in window_predictions
        if row.get("prediction") == "no_fall"
    ]
    bending_only_windows = [
        row for row in window_predictions
        if row.get("hallucination_pattern") == "bending_only"
    ]

    majority_prediction = "fall" if len(fall_windows) > len(no_fall_windows) else "no_fall"
    evidence_prediction = "fall" if len(fall_windows) >= min_fall_windows else "no_fall"
    hallucination_filter_applied = (
        len(fall_windows) < min_fall_windows
        and len(bending_only_windows) >= max(1, len(window_predictions) // 2)
    )
    final_prediction = "no_fall" if hallucination_filter_applied else evidence_prediction

    return {
        "window_count": len(window_predictions),
        "fall_window_count": len(fall_windows),
        "no_fall_window_count": len(no_fall_windows),
        "bending_only_window_count": len(bending_only_windows),
        "majority_prediction": majority_prediction,
        "evidence_prediction": evidence_prediction,
        "hallucination_filter_applied": hallucination_filter_applied,
        "prediction": final_prediction,
    }


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(video_rows: List[Dict[str, object]]) -> Dict[str, float]:
    tp = sum(1 for row in video_rows if row["target"] == 1 and row["prediction"] == "fall")
    tn = sum(1 for row in video_rows if row["target"] == 0 and row["prediction"] == "no_fall")
    fp = sum(1 for row in video_rows if row["target"] == 0 and row["prediction"] == "fall")
    fn = sum(1 for row in video_rows if row["target"] == 1 and row["prediction"] == "no_fall")

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)

    return {
        "videos": total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def process_manifest(args: argparse.Namespace) -> Dict[str, object]:
    manifest_path = Path(args.pose_manifest)
    output_root = Path(args.output_root)
    rows = load_manifest_rows(manifest_path)

    window_rows: List[Dict[str, object]] = []
    video_rows: List[Dict[str, object]] = []

    for row in rows:
        pose_json_path = Path(row["pose_json"])
        frames = load_pose_frames(pose_json_path)
        video_windows: List[Dict[str, object]] = []

        for window_index, (start, window) in enumerate(
            sliding_windows(frames, args.window_size, args.stride)
        ):
            verdict = classify_window(
                window=window,
                motion_threshold=args.motion_threshold,
                ground_torso_threshold=args.ground_torso_threshold,
                bend_torso_threshold=args.bend_torso_threshold,
            )
            frame_names = [str(item["frame"]) for item in window]
            frame_indices = [str(item["frame_index"]) for item in window]
            frame_paths = [str(Path(row["frame_dir"]) / name) for name in frame_names]
            window_row = {
                "video_id": row["video_id"],
                "split": row["split"],
                "label": row["label"],
                "target": 1 if row["label"] == "fall" else 0,
                "window_index": window_index,
                "window_start_position": start,
                "frame_dir": row["frame_dir"],
                "frame_names": "|".join(frame_names),
                "frame_indices": "|".join(frame_indices),
                "frame_paths": "|".join(frame_paths),
                **verdict,
            }
            video_windows.append(window_row)
            window_rows.append(window_row)

        if video_windows:
            aggregation = aggregate_video(video_windows, args.min_fall_windows)
        else:
            aggregation = {
                "window_count": 0,
                "fall_window_count": 0,
                "no_fall_window_count": 0,
                "bending_only_window_count": 0,
                "majority_prediction": "no_fall",
                "evidence_prediction": "no_fall",
                "hallucination_filter_applied": False,
                "prediction": "no_fall",
            }

        video_rows.append(
            {
                "video_id": row["video_id"],
                "split": row["split"],
                "label": row["label"],
                "target": 1 if row["label"] == "fall" else 0,
                "frame_count": int(row.get("frame_count", 0)),
                "pose_frame_count": int(row.get("pose_frame_count", 0)),
                **aggregation,
            }
        )

    window_fieldnames = [
        "video_id",
        "split",
        "label",
        "target",
        "window_index",
        "window_start_position",
        "frame_dir",
        "frame_names",
        "frame_indices",
        "frame_paths",
        "prediction",
        "confidence",
        "motion_score",
        "min_torso_vertical_diff",
        "max_torso_vertical_diff",
        "hip_drop",
        "nose_drop",
        "hallucination_pattern",
        "reason",
    ]
    video_fieldnames = [
        "video_id",
        "split",
        "label",
        "target",
        "frame_count",
        "pose_frame_count",
        "window_count",
        "fall_window_count",
        "no_fall_window_count",
        "bending_only_window_count",
        "majority_prediction",
        "evidence_prediction",
        "hallucination_filter_applied",
        "prediction",
    ]

    write_csv(output_root / "window_predictions.csv", window_rows, window_fieldnames)
    write_csv(output_root / "video_predictions.csv", video_rows, video_fieldnames)

    metrics = compute_metrics(video_rows)
    summary = {
        "pose_manifest": str(manifest_path),
        "output_root": str(output_root),
        "window_size": args.window_size,
        "stride": args.stride,
        "min_fall_windows": args.min_fall_windows,
        "motion_threshold": args.motion_threshold,
        "ground_torso_threshold": args.ground_torso_threshold,
        "bend_torso_threshold": args.bend_torso_threshold,
        "metrics": metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding-window temporal fall detection from pose-frame manifests."
    )
    parser.add_argument("--pose-manifest", default=str(DEFAULT_POSE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--min-fall-windows",
        type=int,
        default=2,
        help="Minimum fall windows needed before the video can remain a fall after filtering.",
    )
    parser.add_argument("--motion-threshold", type=float, default=0.04)
    parser.add_argument("--ground-torso-threshold", type=float, default=0.25)
    parser.add_argument("--bend-torso-threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    summary = process_manifest(parse_args())
    print("Sliding-window temporal fall detection complete.")
    print(f"Window size: {summary['window_size']} | stride: {summary['stride']}")
    print(f"Metrics: {summary['metrics']}")
    print(f"Outputs saved to: {summary['output_root']}")


if __name__ == "__main__":
    main()
