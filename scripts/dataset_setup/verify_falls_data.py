from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_CSV_SUFFIXES = ["-acc.csv", "-data.csv"]
REQUIRED_ZIP_SUFFIXES = ["-cam0-d.zip", "-cam0-rgb.zip", "-cam1-d.zip", "-cam1-rgb.zip"]


def check_csv(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader)
        return True, f"ok ({rows} rows)"
    except Exception as exc:
        return False, f"unreadable ({exc})"


def check_zip(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"

    # Quick signature check before opening as ZIP.
    try:
        with path.open("rb") as f:
            sig = f.read(4)
    except Exception as exc:
        return False, f"unreadable ({exc})"

    if sig != b"PK\x03\x04":
        return False, "invalid signature (likely HTML or partial download)"

    try:
        with zipfile.ZipFile(path) as zf:
            count = len(zf.infolist())
            bad_member = zf.testzip()
        if bad_member:
            return False, f"corrupt member ({bad_member})"
        return True, f"ok ({count} entries)"
    except Exception as exc:
        return False, f"invalid zip ({exc})"


def verify_fall_dir(fall_dir: Path) -> Dict[str, Tuple[bool, str]]:
    base = fall_dir.name
    checks: Dict[str, Tuple[bool, str]] = {}

    for suffix in REQUIRED_CSV_SUFFIXES:
        file_name = f"{base}{suffix}"
        checks[file_name] = check_csv(fall_dir / file_name)

    for suffix in REQUIRED_ZIP_SUFFIXES:
        file_name = f"{base}{suffix}"
        checks[file_name] = check_zip(fall_dir / file_name)

    return checks


def find_fall_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("fall-*") if p.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify fall dataset integrity.")
    parser.add_argument(
        "--root",
        default="data/falls",
        help="Path to falls root directory (default: data/falls)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}")
        return 2

    fall_dirs = find_fall_dirs(root)
    if not fall_dirs:
        print(f"ERROR: no fall-* directories found under: {root}")
        return 2

    total_failures = 0
    print(f"Verifying fall folders under: {root}\n")
    for fall_dir in fall_dirs:
        print(f"[{fall_dir.name}]")
        results = verify_fall_dir(fall_dir)
        for file_name, (ok, message) in results.items():
            status = "OK  " if ok else "FAIL"
            if not ok:
                total_failures += 1
            print(f"  {status} {file_name}: {message}")
        print()

    if total_failures:
        print(f"Verification finished with {total_failures} failure(s).")
        return 1

    print("Verification passed with no failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
