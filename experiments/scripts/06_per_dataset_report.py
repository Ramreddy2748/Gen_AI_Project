"""
Break video-level metrics down by dataset (URFD / Le2i / GMNCSA24).

This lets us check whether one dataset is dragging the aggregate up or down,
and is the start of the cross-dataset generalization study.

Usage:
    python experiments/scripts/06_per_dataset_report.py \\
        --strategy best_prompt --split test --rule ratio_0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core import aggregation as agg
from experiments.core.config import EXP_RESULTS
from experiments.core.data_loader import video_id_to_dataset, video_window_index
from experiments.core.metrics import format_metrics, video_metrics


RULES = {
    "ratio_0.25": agg.make_ratio(0.25),
    "ratio_0.30": agg.make_ratio(0.30),
    "ratio_0.40": agg.make_ratio(0.40),
    "ratio_0.50": agg.make_ratio(0.50),
    "ratio_0.60": agg.make_ratio(0.60),
    "consecutive_k2": agg.make_consecutive(2),
    "consecutive_k3": agg.make_consecutive(3),
    "consecutive_k4": agg.make_consecutive(4),
    "end_state_min1": agg.end_state,
    "consecutive_k2_and_end": agg.make_consecutive_end(2),
    "any_fall_window": agg.any_fall,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="best_prompt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rule", default="ratio_0.50", choices=list(RULES.keys()))
    args = ap.parse_args()

    out_dir = EXP_RESULTS / "06_per_dataset" / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    by_video = video_window_index(args.strategy, args.split)
    fn = RULES[args.rule]
    preds = agg.predict_all(by_video, fn)

    per_ds: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for vid, recs in by_video.items():
        ds = video_id_to_dataset(vid)
        per_ds[ds]["y_true"].append(recs[0]["true_label"])
        per_ds[ds]["y_pred"].append(preds[vid])

    print(f"Per-dataset breakdown for strategy={args.strategy} split={args.split} rule={args.rule}\n")
    summary: Dict[str, Dict] = {}
    for ds, d in sorted(per_ds.items()):
        m = video_metrics(d["y_true"], d["y_pred"])
        summary[ds] = m
        print(f"=== {ds} ===")
        print(format_metrics(m))
        print()

    # Overall
    y_true = [r["true_label"] for recs in by_video.values() for r in recs[:1]]
    y_pred = [preds[v] for v in by_video]
    m_all = video_metrics(y_true, y_pred)
    summary["overall"] = m_all
    print("=== overall ===")
    print(format_metrics(m_all))

    with open(out_dir / f"per_dataset_{args.split}_{args.rule}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
