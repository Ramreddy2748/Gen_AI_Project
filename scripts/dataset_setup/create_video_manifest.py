import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


VOXEL_ROOT = Path("data/raw/voxel_huggingface_data")
URFD_ROOT = Path("data/raw/urfd_videos")
OUTPUT_CSV = Path("data/manifests/video_manifest.csv")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
DEFAULT_SPLIT_STRATEGY = "grouped-stratified-by-label"


def normalize_label(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"fall", "falls"}:
        return "fall"
    if lowered in {"no fall", "nofall", "no_fall", "adl"}:
        return "no_fall"
    raise ValueError(f"Unsupported label source: {value}")


def target_from_label(label: str) -> int:
    return 1 if label == "fall" else 0


def valid_ratio(train_ratio: float, val_ratio: float) -> bool:
    test_ratio = 1.0 - train_ratio - val_ratio
    return 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and test_ratio > 0.0


def collect_voxel_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not root.exists():
        return rows

    for video_path in sorted(root.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        label = normalize_label(video_path.parent.name)
        video_id = video_path.stem.replace(" ", "_")
        rows.append(
            {
                "video_id": f"voxel_{video_id}",
                "dataset": "voxel_huggingface_data",
                "label": label,
                "target": str(target_from_label(label)),
                "group_id": f"voxel::{video_path.stem}",
                "camera": "unknown",
                "path": str(video_path),
                "split_strategy": DEFAULT_SPLIT_STRATEGY,
            }
        )

    return rows


def collect_urfd_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not root.exists():
        return rows

    for video_path in sorted(root.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative_parts = video_path.relative_to(root).parts
        label = normalize_label(relative_parts[0])
        sequence_id = relative_parts[1]
        camera = relative_parts[2]
        rows.append(
            {
                "video_id": video_path.stem.replace("-", "_"),
                "dataset": "urfd_videos",
                "label": label,
                "target": str(target_from_label(label)),
                "group_id": f"urfd::{sequence_id}",
                "camera": camera,
                "path": str(video_path),
                "split_strategy": DEFAULT_SPLIT_STRATEGY,
            }
        )

    return rows


def assign_group_splits(
    rows: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    label_to_groups: Dict[str, List[str]] = defaultdict(list)

    for row in rows:
        group_id = row["group_id"]
        if group_id not in label_to_groups[row["label"]]:
            label_to_groups[row["label"]].append(group_id)

    split_by_group: Dict[str, str] = {}
    rng = random.Random(seed)

    for label, group_ids in label_to_groups.items():
        group_ids = sorted(group_ids)
        rng.shuffle(group_ids)

        total = len(group_ids)
        train_cutoff = int(total * train_ratio)
        val_cutoff = int(total * (train_ratio + val_ratio))

        for index, group_id in enumerate(group_ids):
            if index < train_cutoff:
                split_by_group[group_id] = "train"
            elif index < val_cutoff:
                split_by_group[group_id] = "val"
            else:
                split_by_group[group_id] = "test"

    return split_by_group


def write_manifest(rows: List[Dict[str, str]], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a training manifest CSV for local video datasets."
    )
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not valid_ratio(args.train_ratio, args.val_ratio):
        raise ValueError("Invalid split ratios. train + val must be less than 1.0.")

    rows = collect_voxel_rows(VOXEL_ROOT) + collect_urfd_rows(URFD_ROOT)
    split_by_group = assign_group_splits(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    finalized_rows: List[Dict[str, str]] = []
    for row in rows:
        row = dict(row)
        row["split"] = split_by_group[row["group_id"]]
        finalized_rows.append(row)

    finalized_rows.sort(key=lambda item: (item["dataset"], item["split"], item["label"], item["video_id"]))
    output_path = write_manifest(finalized_rows, Path(args.output))
    print(f"Saved manifest to: {output_path}")
    print(f"Total rows: {len(finalized_rows)}")


if __name__ == "__main__":
    main()
