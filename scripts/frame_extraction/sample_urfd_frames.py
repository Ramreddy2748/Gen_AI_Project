import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


MANIFEST_PATH = Path("data/pose_pipeline/split_manifest.json")
OUTPUT_ROOT = Path("data/pose_pipeline/sampled_frames")


def load_manifest() -> Dict[str, object]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_frame_paths(sequence_dir: Path, every_n: int, max_frames: int) -> List[Path]:
    all_frames = sorted(p for p in sequence_dir.iterdir() if p.suffix.lower() == ".png")
    sampled = all_frames[::every_n]
    if max_frames > 0:
        sampled = sampled[:max_frames]
    return sampled


def process_split(split_name: str, items: List[Dict[str, str]], every_n: int, max_frames: int) -> None:
    split_output_dir = OUTPUT_ROOT / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split_name,
        "every_n": every_n,
        "max_frames": max_frames,
        "videos": [],
    }

    for item in items:
        sequence_dir = Path(item["sequence_dir"])
        video_output_dir = split_output_dir / item["video_id"]
        video_output_dir.mkdir(parents=True, exist_ok=True)

        sampled_frames = sample_frame_paths(sequence_dir, every_n=every_n, max_frames=max_frames)
        for frame_path in sampled_frames:
            shutil.copy2(frame_path, video_output_dir / frame_path.name)

        video_summary = {
            "video_id": item["video_id"],
            "label": item["label"],
            "source_dir": str(sequence_dir),
            "sampled_frame_count": len(sampled_frames),
            "sampled_frames": [frame.name for frame in sampled_frames],
        }
        summary["videos"].append(video_summary)
        print(f"Sampled {len(sampled_frames)} frames from {item['video_id']}")

    summary_path = OUTPUT_ROOT / f"{split_name}_sampling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved sampling summary to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample frames from URFD sequences.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--every-n", type=int, default=10, help="Keep one frame every N frames.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap per video. Use 0 for no cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    splits = manifest["splits"]

    selected_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split_name in selected_splits:
        process_split(
            split_name=split_name,
            items=splits[split_name],
            every_n=args.every_n,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
