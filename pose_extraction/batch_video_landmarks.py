import argparse
import json
from pathlib import Path
from typing import Dict, List

from pose_extraction.sample_video_frames_and_landmarks import ensure_model, process_video


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def collect_videos(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def infer_label(video_path: Path, input_dir: Path) -> str:
    relative_parts = [part.lower() for part in video_path.relative_to(input_dir).parts]
    if any(part == "fall" for part in relative_parts):
        return "fall"
    if any(part == "no fall" for part in relative_parts):
        return "no_fall"
    return "unknown"


def safe_video_id(video_path: Path, input_dir: Path) -> str:
    relative = video_path.relative_to(input_dir).with_suffix("")
    return "__".join(relative.parts)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    sample_every_n: int,
    max_frames: int,
    save_annotated: bool,
) -> Path:
    videos = collect_videos(input_dir)
    summary: Dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "video_count": len(videos),
        "videos": [],
    }

    for video_path in videos:
        label = infer_label(video_path, input_dir)
        video_id = safe_video_id(video_path, input_dir)
        video_output_dir = output_dir / label / video_id
        json_path = process_video(
            video_path=video_path,
            output_dir=video_output_dir,
            sample_every_n=sample_every_n,
            max_frames=max_frames,
            save_annotated=save_annotated,
        )

        summary["videos"].append(
            {
                "video_id": video_id,
                "label": label,
                "video_path": str(video_path),
                "landmarks_json": str(json_path),
            }
        )
        print(f"Processed {video_path.name} -> {json_path}")

    summary_path = output_dir / "dataset_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all videos in a dataset folder and extract sampled-frame pose landmarks."
    )
    parser.add_argument("--input-dir", required=True, help="Root folder containing videos.")
    parser.add_argument(
        "--output-dir",
        default="data/video_dataset_pose_output",
        help="Directory where per-video outputs will be saved.",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        help="Keep one frame every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on sampled frames per video. Use 0 for no cap.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save sampled frames with pose landmarks drawn on them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.sample_every_n <= 0:
        raise ValueError("--sample-every-n must be greater than 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")

    ensure_model()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_every_n=args.sample_every_n,
        max_frames=args.max_frames,
        save_annotated=args.save_annotated,
    )
    print(f"Dataset summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
