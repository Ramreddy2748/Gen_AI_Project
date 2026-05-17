"""
Build sliding windows of multiple sizes from the existing pose JSONs.

No MediaPipe re-run is needed because `data/poses/{split}/{label}/*.json`
already contains the per-frame landmarks. We re-derive enhanced features
(velocity, acceleration, real jerk, angular velocity) per window size.

Output:
    experiments/data/windows_n{N}/{split}/{label}/{video_id}/{wid}.json
    experiments/data/windows_n{N}/manifest_{split}.json

Usage:
    python experiments/scripts/02_build_window_sizes.py \\
        --sizes 5 7 9 12 15 --splits train val test --stride 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core.window_builder import build_window_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 7, 9, 12, 15],
                    help="Window sizes to build (default: 5 7 9 12 15)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=1,
                    help="Window stride (default 1; matches fall-class stride from balanced_pipeline)")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    print(f"Building windows for sizes={args.sizes} splits={args.splits} stride={args.stride}")
    for size in args.sizes:
        for split in args.splits:
            build_window_set(
                split=split,
                window_size=size,
                stride=args.stride,
                fps=args.fps,
            )


if __name__ == "__main__":
    main()
