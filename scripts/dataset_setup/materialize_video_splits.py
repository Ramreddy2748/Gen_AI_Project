import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List


DEFAULT_MANIFEST = Path("data/manifests/video_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("data/splits/videos")
SPLIT_NAME_MAP = {"train": "train", "val": "valid", "test": "test"}


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as file_obj:
        return [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(file_obj)
        ]


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_output_path(output_root: Path, row: Dict[str, str]) -> Path:
    split_dir = SPLIT_NAME_MAP[row["split"]]
    label_dir = row["label"]
    filename = Path(row["path"]).name
    return output_root / split_dir / label_dir / filename


def remove_existing_target(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def create_link(source: Path, target: Path) -> None:
    remove_existing_target(target)
    os.symlink(source.resolve(), target)


def create_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    shutil.copy2(source, target)


def materialize_rows(rows: List[Dict[str, str]], output_root: Path, mode: str) -> Dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}

    for row in rows:
        source = Path(row["path"])
        if not source.exists():
            print(f"Skipping missing source: {source}")
            continue

        target = split_output_path(output_root, row)
        ensure_clean_dir(target.parent)

        if mode == "symlink":
            create_link(source, target)
        else:
            create_copy(source, target)

        counts[SPLIT_NAME_MAP[row["split"]]] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/valid/test dataset folders from the video manifest."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Input manifest CSV.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder where split directories will be created.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks to avoid duplication, or copy files physically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    counts = materialize_rows(rows=rows, output_root=output_root, mode=args.mode)

    print(f"Created split folders at: {output_root}")
    print(
        f"Counts -> train: {counts['train']}, "
        f"valid: {counts['valid']}, test: {counts['test']}"
    )


if __name__ == "__main__":
    main()
