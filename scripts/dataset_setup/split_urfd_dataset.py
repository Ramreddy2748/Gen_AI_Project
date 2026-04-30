import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "pose_pipeline"


def collect_videos() -> List[Dict[str, str]]:
    videos: List[Dict[str, str]] = []

    adl_dir = DATA_DIR / "adl"
    for seq_dir in sorted(adl_dir.glob("adl-*-cam0-rgb")):
        if seq_dir.is_dir():
            videos.append(
                {
                    "video_id": seq_dir.name,
                    "label": "no_fall",
                    "category": "adl",
                    "sequence_dir": str(seq_dir),
                }
            )

    falls_dir = DATA_DIR / "falls"
    for fall_dir in sorted(falls_dir.glob("fall-*")):
        if not fall_dir.is_dir():
            continue
        for seq_dir in sorted(fall_dir.glob("*cam0-rgb*")):
            if seq_dir.is_dir():
                videos.append(
                    {
                        "video_id": fall_dir.name,
                        "label": "fall",
                        "category": "falls",
                        "sequence_dir": str(seq_dir),
                    }
                )

    return videos


def split_items(
    items: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, str]]] = {"fall": [], "no_fall": []}

    for item in items:
        grouped[item["label"]].append(item)

    splits = {"train": [], "val": [], "test": []}
    for label_items in grouped.values():
        rng.shuffle(label_items)
        total = len(label_items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits["train"].extend(label_items[:train_end])
        splits["val"].extend(label_items[train_end:val_end])
        splits["test"].extend(label_items[val_end:])

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda x: x["video_id"])

    return splits


def build_manifest(
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, object]:
    videos = collect_videos()
    splits = split_items(videos, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    return {
        "dataset": "URFD",
        "note": "Video-level split for the fall detection pipeline up to pose extraction.",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
        "summary": {
            split_name: {
                "total_videos": len(split_items_list),
                "fall_videos": sum(1 for item in split_items_list if item["label"] == "fall"),
                "no_fall_videos": sum(
                    1 for item in split_items_list if item["label"] == "no_fall"
                ),
            }
            for split_name, split_items_list in splits.items()
        },
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for URFD at the video level."
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_path = OUTPUT_DIR / "split_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest to: {out_path}")
    for split_name, summary in manifest["summary"].items():
        print(
            f"{split_name}: total={summary['total_videos']}, "
            f"fall={summary['fall_videos']}, no_fall={summary['no_fall_videos']}"
        )


if __name__ == "__main__":
    main()
