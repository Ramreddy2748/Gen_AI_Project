import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_WINDOW_JSONL = Path("data/processed/windows/train/train_llm_windows.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/processed/window_poses")


def safe_name(value: object) -> str:
    text = str(value).strip()
    keep = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        elif char in {" ", ".", ":", "/"}:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "unknown"


def time_token(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 0.0
    return f"{seconds:.3f}s".replace(".", "p")


def frame_index_range(frames: List[Dict[str, object]]) -> str:
    indices = []
    for frame in frames:
        try:
            indices.append(int(frame.get("frame_index", 0)))
        except (TypeError, ValueError):
            indices.append(0)
    if not indices:
        return "frames_unknown"
    return f"frames_{indices[0]:06d}_to_{indices[-1]:06d}"


def descriptive_output_path(
    output_root: Path,
    payload: Dict[str, object],
) -> Path:
    frames = payload["frames"]
    if not isinstance(frames, list):
        frames = []

    split = safe_name(payload["split"])
    label = safe_name(payload["label"])
    video_id = safe_name(payload["video_id"])
    window_index = int(payload.get("window_index", 0))
    video_start = time_token(payload["video_start_time_seconds"])
    video_end = time_token(payload["video_end_time_seconds"])
    window_start = time_token(payload["window_start_time_seconds"])
    window_end = time_token(payload["window_end_time_seconds"])

    label_dir = f"label_{label}"
    video_dir = f"{video_id}__video_{video_start}_to_{video_end}"
    file_name = (
        f"window_{window_index:04d}"
        f"__time_{window_start}_to_{window_end}"
        f"__{frame_index_range(frames)}.json"
    )
    return output_root / split / label_dir / video_dir / file_name


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def window_pose_payload(window: Dict[str, object]) -> Dict[str, object]:
    frames = window.get("frames")
    if not isinstance(frames, list):
        frames = []

    pose_frames: List[Dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        pose_frames.append(
            {
                "position": frame.get("position"),
                "frame": frame.get("frame"),
                "frame_index": frame.get("frame_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "frame_path": frame.get("frame_path"),
                "pose_detected": bool(frame.get("pose_detected", False)),
                "features": frame.get("features") if isinstance(frame.get("features"), dict) else {},
            }
        )

    return {
        "window_id": window["window_id"],
        "video_id": window["video_id"],
        "split": window.get("split", ""),
        "label": window.get("label", ""),
        "target": window.get("target", ""),
        "source_video_path": window.get("source_video_path", ""),
        "video_start_time_seconds": window.get("video_start_time_seconds", 0.0),
        "video_end_time_seconds": window.get("video_end_time_seconds", 0.0),
        "window_start_time_seconds": window.get("window_start_time_seconds", 0.0),
        "window_end_time_seconds": window.get("window_end_time_seconds", 0.0),
        "window_index": window.get("window_index", 0),
        "window_start_position": window.get("window_start_position", 0),
        "window_size": window.get("window_size", len(pose_frames)),
        "pose_frame_count": sum(1 for frame in pose_frames if frame["pose_detected"]),
        "frames": pose_frames,
    }


def manifest_row(payload: Dict[str, object], pose_json_path: Path) -> Dict[str, object]:
    frames = payload["frames"]
    if not isinstance(frames, list):
        frames = []

    return {
        "window_id": payload["window_id"],
        "video_id": payload["video_id"],
        "split": payload["split"],
        "label": payload["label"],
        "target": payload["target"],
        "source_video_path": payload["source_video_path"],
        "video_start_time_seconds": payload["video_start_time_seconds"],
        "video_end_time_seconds": payload["video_end_time_seconds"],
        "window_start_time_seconds": payload["window_start_time_seconds"],
        "window_end_time_seconds": payload["window_end_time_seconds"],
        "window_index": payload["window_index"],
        "window_size": payload["window_size"],
        "pose_json": str(pose_json_path),
        "pose_frame_count": payload["pose_frame_count"],
        "frame_paths": "|".join(str(frame["frame_path"]) for frame in frames),
        "frame_timestamps_seconds": "|".join(str(frame["timestamp_seconds"]) for frame in frames),
    }


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
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
        "window_size",
        "pose_json",
        "pose_frame_count",
        "frame_paths",
        "frame_timestamps_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_split(window: Dict[str, object], fallback: str) -> str:
    split = str(window.get("split", "")).strip()
    return split or fallback


def extract_window_pose_splits(args: argparse.Namespace) -> Dict[str, object]:
    window_jsonl = Path(args.window_jsonl)
    output_root = Path(args.output_root)
    rows: List[Dict[str, object]] = []
    count = 0

    for window in iter_jsonl(window_jsonl):
        payload = window_pose_payload(window)
        if args.naming == "descriptive":
            pose_json_path = descriptive_output_path(output_root, payload)
        else:
            split = infer_split(window, args.split)
            label = str(window.get("label", "unknown")).strip() or "unknown"
            video_id = str(window["video_id"])
            window_id = str(window["window_id"])
            pose_json_path = output_root / split / label / video_id / f"{window_id}.json"

        pose_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_json_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

        rows.append(manifest_row(payload, pose_json_path))
        count += 1

    manifest_path = output_root / args.split / f"{args.split}_window_pose_manifest.csv"
    write_manifest(manifest_path, rows)

    summary = {
        "window_jsonl": str(window_jsonl),
        "output_root": str(output_root),
        "split": args.split,
        "window_count": count,
        "manifest_path": str(manifest_path),
    }
    summary_path = output_root / args.split / f"{args.split}_window_pose_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one pose JSON per pre-LLM window split."
    )
    parser.add_argument("--window-jsonl", default=str(DEFAULT_WINDOW_JSONL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--naming",
        choices=["compact", "descriptive"],
        default="descriptive",
        help="Use descriptive folder/file names or the original compact layout.",
    )
    return parser.parse_args()


def main() -> None:
    summary = extract_window_pose_splits(parse_args())
    print("Extracted window pose splits.")
    print(f"Windows: {summary['window_count']}")
    print(f"Manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
