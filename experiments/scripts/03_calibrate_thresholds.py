"""
Recompute per-frame thresholds using **train-split statistics only**.

This addresses the small but legitimate leakage in
`final_pipeline/scripts/balanced_pipeline.py` where THRESHOLDS were
hand-picked (e.g. `rapid_velocity_min = 0.15`) and likely tuned against
the full dataset. The new thresholds come from percentiles of the train
pose distributions.

The output JSON can be passed to `core.window_builder.build_window_set` via
the `thresholds=` argument when constructing windows for downstream
experiments.

Usage:
    python experiments/scripts/03_calibrate_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core.config import EXP_DATA, DEFAULT_THRESHOLDS
from experiments.core.data_loader import iter_pose_videos
from experiments.core.features import calibrate_thresholds_on_train


def main():
    print("Loading TRAIN pose videos...")
    train_videos = list(iter_pose_videos("train"))
    print(f"Loaded {len(train_videos)} train videos")

    thresholds = calibrate_thresholds_on_train(train_videos)
    EXP_DATA.mkdir(parents=True, exist_ok=True)
    out = EXP_DATA / "thresholds_train_calibrated.json"
    with open(out, "w") as f:
        json.dump({"defaults": DEFAULT_THRESHOLDS, "calibrated": thresholds}, f, indent=2)

    print("\nThresholds (from balanced_pipeline.py defaults vs train-calibrated):")
    width = max(len(k) for k in thresholds)
    for k in thresholds:
        d = DEFAULT_THRESHOLDS.get(k, "n/a")
        v = thresholds[k]
        print(f"  {k:<{width}}  default={d}   calibrated={v:.4f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
