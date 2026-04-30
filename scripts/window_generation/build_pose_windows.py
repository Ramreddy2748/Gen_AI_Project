import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_POSE_MANIFEST = Path("data/pose_splits/train_pose_manifest.csv")
DEFAULT_FRAME_MANIFEST = Path("data/frame_splits/train_frame_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("data/window_splits")
SPLIT_DIR_NAME = {"train": "train", "val": "valid", "test": "test"}


def parse_frame_index(frame_name: str) -> int:
    digits = "".join(ch for ch in Path(frame_name).stem if ch.isdigit())
    return int(digits) if digits else 0


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def load_frame_metadata(frame_manifest_path: Path) -> Dict[str, Dict[str, str]]:
    if not frame_manifest_path.exists():
        return {}
    with open(frame_manifest_path, "r", newline="", encoding="utf-8") as file_obj:
        return {
            row["video_id"]: row
            for row in csv.DictReader(file_obj)
        }


def float_value(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def time_from_frame_index(frame_index: int, fps: float) -> float:
    return round(frame_index / fps, 4) if fps > 0 else 0.0


def load_pose_frames(pose_json_path: Path) -> List[Dict[str, object]]:
    with open(pose_json_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    frames: List[Dict[str, object]] = []
    for item in payload.get("frames", []):
        frame_name = str(item.get("frame", ""))
        features = item.get("features") if item.get("pose_detected") else {}
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
        raise ValueError("--window-size must be positive")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if len(frames) < window_size:
        return

    for start in range(0, len(frames) - window_size + 1, stride):
        yield start, frames[start : start + window_size]


def build_window_record(
    manifest_row: Dict[str, str],
    frame_metadata: Dict[str, str],
    window_index: int,
    start_position: int,
    window_frames: List[Dict[str, object]],
) -> Dict[str, object]:
    video_id = manifest_row["video_id"]
    label = manifest_row["label"]
    split = manifest_row["split"]
    frame_dir = Path(manifest_row["frame_dir"])
    window_id = f"{video_id}_w{window_index:04d}"
    fps = float_value(frame_metadata, "fps")
    duration_seconds = float_value(frame_metadata, "duration_seconds")
    source_video_path = frame_metadata.get("path", "")

    frames: List[Dict[str, object]] = []
    for position, frame in enumerate(window_frames):
        frame_name = str(frame["frame"])
        frame_index = int(frame["frame_index"])
        frames.append(
            {
                "position": position,
                "frame": frame_name,
                "frame_index": frame_index,
                "timestamp_seconds": time_from_frame_index(frame_index, fps),
                "frame_path": str(frame_dir / frame_name),
                "pose_detected": bool(frame["pose_detected"]),
                "features": frame["features"],
            }
        )

    window_start_time = frames[0]["timestamp_seconds"] if frames else 0.0
    window_end_time = frames[-1]["timestamp_seconds"] if frames else 0.0

    return {
        "window_id": window_id,
        "video_id": video_id,
        "split": split,
        "label": label,
        "target": 1 if label == "fall" else 0,
        "source_video_path": source_video_path,
        "video_start_time_seconds": 0.0,
        "video_end_time_seconds": round(duration_seconds, 4),
        "window_start_time_seconds": window_start_time,
        "window_end_time_seconds": window_end_time,
        "window_index": window_index,
        "window_start_position": start_position,
        "window_size": len(window_frames),
        "frame_dir": str(frame_dir),
        "pose_json": manifest_row["pose_json"],
        "frames": frames,
    }


def csv_row_from_record(record: Dict[str, object]) -> Dict[str, object]:
    frames = record["frames"]
    if not isinstance(frames, list):
        frames = []

    return {
        "window_id": record["window_id"],
        "video_id": record["video_id"],
        "split": record["split"],
        "label": record["label"],
        "target": record["target"],
        "source_video_path": record["source_video_path"],
        "video_start_time_seconds": record["video_start_time_seconds"],
        "video_end_time_seconds": record["video_end_time_seconds"],
        "window_start_time_seconds": record["window_start_time_seconds"],
        "window_end_time_seconds": record["window_end_time_seconds"],
        "window_index": record["window_index"],
        "window_start_position": record["window_start_position"],
        "window_size": record["window_size"],
        "frame_dir": record["frame_dir"],
        "pose_json": record["pose_json"],
        "frame_names": "|".join(str(frame["frame"]) for frame in frames),
        "frame_indices": "|".join(str(frame["frame_index"]) for frame in frames),
        "frame_timestamps_seconds": "|".join(str(frame["timestamp_seconds"]) for frame in frames),
        "frame_paths": "|".join(str(frame["frame_path"]) for frame in frames),
        "pose_detected_count": sum(1 for frame in frames if frame.get("pose_detected")),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_id",
        "video_id",
        "split",
        "label",
        "target",
        "source_video_path",
        "video_start_time_seconds",
        "video_end_time_seconds",
        "window_start_time_seconds",
        "window_end_time_seconds",
        "window_index",
        "window_start_position",
        "window_size",
        "frame_dir",
        "pose_json",
        "frame_names",
        "frame_indices",
        "frame_timestamps_seconds",
        "frame_paths",
        "pose_detected_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")


def build_windows(args: argparse.Namespace) -> Dict[str, object]:
    pose_manifest = Path(args.pose_manifest)
    output_root = Path(args.output_root)
    split_dir_name = SPLIT_DIR_NAME[args.split]
    output_dir = output_root / split_dir_name
    frame_metadata_by_video = load_frame_metadata(Path(args.frame_manifest))

    records: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []
    video_count = 0

    for row in load_manifest_rows(pose_manifest):
        if row["split"] != args.split:
            continue

        video_count += 1
        pose_frames = load_pose_frames(Path(row["pose_json"]))
        for window_index, (start, window_frames) in enumerate(
            sliding_windows(pose_frames, args.window_size, args.stride)
        ):
            record = build_window_record(
                manifest_row=row,
                frame_metadata=frame_metadata_by_video.get(row["video_id"], {}),
                window_index=window_index,
                start_position=start,
                window_frames=window_frames,
            )
            records.append(record)
            csv_rows.append(csv_row_from_record(record))

    csv_path = output_dir / f"{split_dir_name}_window_manifest.csv"
    jsonl_path = output_dir / f"{split_dir_name}_llm_windows.jsonl"
    summary_path = output_dir / f"{split_dir_name}_window_summary.json"

    write_csv(csv_path, csv_rows)
    write_jsonl(jsonl_path, records)

    summary = {
        "pose_manifest": str(pose_manifest),
        "frame_manifest": str(args.frame_manifest),
        "output_root": str(output_root),
        "split": args.split,
        "window_size": args.window_size,
        "stride": args.stride,
        "video_count": video_count,
        "window_count": len(records),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 3-frame pose windows for the pre-LLM fall-detection stage."
    )
    parser.add_argument("--pose-manifest", default=str(DEFAULT_POSE_MANIFEST))
    parser.add_argument("--frame-manifest", default=str(DEFAULT_FRAME_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = build_windows(parse_args())
    print("Built pre-LLM pose windows.")
    print(f"Videos: {summary['video_count']}")
    print(f"Windows: {summary['window_count']}")
    print(f"CSV: {summary['csv_path']}")
    print(f"JSONL: {summary['jsonl_path']}")


if __name__ == "__main__":
    main()
