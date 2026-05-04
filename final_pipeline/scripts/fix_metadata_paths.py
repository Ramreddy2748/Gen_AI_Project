"""
Fix Metadata Paths
==================

The repo's `ground_truth.csv` has absolute paths baked in from a teammate's
machine (Abhishek's MacBook). On any other machine, the row-level
`window_path` column points to a directory that doesn't exist, which is why
the prompting scripts fail with FileNotFoundError.

This helper rewrites the prefix of every window_path in ground_truth.csv to
match the current machine's project location. It also creates a backup
(`ground_truth.csv.bak`) before saving, so the change is reversible.

Usage:
    python final_pipeline/scripts/fix_metadata_paths.py

You can also override the prefixes from the command line:
    python fix_metadata_paths.py \\
        --old-prefix "/Users/abhishekkumar/Downloads/Gen AI assignments/Final_Project/Gen_AI_Project/" \\
        --new-prefix "/Users/harshitha/Desktop/Gen_AI_Project/"
"""

import argparse
import csv
import shutil
from pathlib import Path

# Default prefixes — change here if your project is at a different path.
DEFAULT_OLD_PREFIX = (
    "/Users/abhishekkumar/Downloads/Gen AI assignments/"
    "Final_Project/Gen_AI_Project/"
)
DEFAULT_NEW_PREFIX = "/Users/harshitha/Desktop/Gen_AI_Project/"

# Path to the metadata file relative to this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent             # .../Gen_AI_Project/
CSV_PATH = PROJECT_DIR / "final_pipeline" / "data" / "metadata" / "ground_truth.csv"
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak")


def fix_paths(old_prefix: str, new_prefix: str, csv_path: Path = CSV_PATH) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"ground_truth.csv not found at: {csv_path}")

    # 1. Backup the original (only if a backup doesn't already exist)
    if not BACKUP_PATH.exists():
        shutil.copy2(csv_path, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Backup already exists, leaving it alone: {BACKUP_PATH}")

    # 2. Read all rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # 3. Rewrite window_path values
    fixed = 0
    untouched = 0
    for row in rows:
        path = row.get("window_path", "")
        if path.startswith(old_prefix):
            row["window_path"] = new_prefix + path[len(old_prefix):]
            fixed += 1
        else:
            untouched += 1

    # 4. Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRewrote {fixed} window_path entries.")
    if untouched:
        print(f"{untouched} entries did not start with the old prefix and were left as-is.")
    print(f"Updated CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix absolute paths in ground_truth.csv")
    parser.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX,
                        help="Path prefix to replace (default: Abhishek's Mac path).")
    parser.add_argument("--new-prefix", default=DEFAULT_NEW_PREFIX,
                        help="Path prefix to replace it with (default: your Mac path).")
    args = parser.parse_args()

    print(f"Old prefix: {args.old_prefix}")
    print(f"New prefix: {args.new_prefix}")
    print()

    fix_paths(args.old_prefix, args.new_prefix)


if __name__ == "__main__":
    main()
