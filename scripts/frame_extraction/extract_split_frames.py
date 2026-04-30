import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2


DEFAULT_MANIFEST_CANDIDATES = [
    Path("data/meta_data/video_manifest.csv"),
    Path("data/manifests/video_manifest.csv"),
]
DEFAULT_SPLIT_ROOT = Path("data/splits/video_data")
DEFAULT_OUTPUT_ROOT = Path("data/frame_splits")
SPLIT_DIR_NAME = {"train": "train", "val": "valid", "test": "test"}


def resolve_manifest_path(explicit_path: str) -> Path:
    if explicit_path:
        manifest_path = Path(explicit_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return manifest_path

    for candidate in DEFAULT_MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No manifest found in the default manifest locations.")


def load_manifest_rows(manifest_path: Path, split_name: str) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
    return [row for row in rows if row["split"] == split_name]


def frame_output_dir(output_root: Path, row: Dict[str, str]) -> Path:
    return output_root / SPLIT_DIR_NAME[row["split"]] / row["label"] / row["video_id"]


def sampling_interval_seconds(sample_fps: float) -> float:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    return 1.0 / sample_fps


def clear_existing_frames(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for frame_path in output_dir.glob("*.jpg"):
        frame_path.unlink()


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    sample_fps: float,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_frames(output_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 else 0.0
    interval_seconds = sampling_interval_seconds(sample_fps)

    frame_index = 0
    saved_count = 0
    next_sample_at_seconds = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_time_seconds = (frame_index / fps) if fps > 0 else 0.0
            if current_time_seconds + 1e-9 >= next_sample_at_seconds:
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
                next_sample_at_seconds += interval_seconds

                if max_frames > 0 and saved_count >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "sample_fps": sample_fps,
        "sampling_interval_seconds": interval_seconds,
        "sampled_frame_count": saved_count,
        "total_frames_read": frame_index,
    }


def write_summary(summary_path: Path, rows: List[Dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "dataset",
        "label",
        "target",
        "group_id",
        "camera",
        "path",
        "split",
        "split_strategy",
        "frame_dir",
        "fps",
        "frame_count",
        "duration_seconds",
        "sample_fps",
        "sampling_interval_seconds",
        "sampled_frame_count",
        "total_frames_read",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sampled frames for one split from the video manifest."
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=3.0,
        help="Number of frames to sample per second from each video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.manifest)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)

    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")

    rows = load_manifest_rows(manifest_path, args.split)
    summary_rows: List[Dict[str, str]] = []

    for row in rows:
        video_path = Path(row["path"])
        output_dir = frame_output_dir(output_root, row)
        stats = extract_video_frames(
            video_path=video_path,
            output_dir=output_dir,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )

        summary_row = dict(row)
        summary_row["frame_dir"] = str(output_dir)
        summary_row["fps"] = f"{stats['fps']:.4f}"
        summary_row["frame_count"] = str(int(stats["frame_count"]))
        summary_row["duration_seconds"] = f"{stats['duration_seconds']:.4f}"
        summary_row["sample_fps"] = f"{stats['sample_fps']:.2f}"
        summary_row["sampling_interval_seconds"] = f"{stats['sampling_interval_seconds']:.4f}"
        summary_row["sampled_frame_count"] = str(stats["sampled_frame_count"])
        summary_row["total_frames_read"] = str(stats["total_frames_read"])
        summary_rows.append(summary_row)
        print(f"Extracted frames for {row['video_id']} -> {output_dir}")

    summary_path = output_root / f"{SPLIT_DIR_NAME[args.split]}_frame_manifest.csv"
    write_summary(summary_path, summary_rows)
    print(f"Saved frame manifest to: {summary_path}")


if __name__ == "__main__":
    main()
