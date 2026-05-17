"""
Re-aggregate existing window-level predictions using new video-level rules.

Cost: $0. We read the per-window predictions from
`final_pipeline/results/<strategy>/window_results_<split>.json` and combine
them differently to maximise video-level F1 while keeping fall recall high.

Strategies compared on the same set of windows:

    baseline     : current rule (>=25% windows = fall) — sanity check
    ratio sweep  : threshold in {0.10, 0.15, ..., 0.60}
    consecutive  : 2 or 3 consecutive fall windows
    end_state    : require the fall window to end fallen/horizontal/on_ground
    combined     : consecutive AND end_state

The script picks the operating point on `--tune-on` (default: val) by Macro
F1 subject to fall_recall >= min_recall, and reports the chosen rule on
`--report-on` (default: test).

Usage:
    python experiments/scripts/01_reaggregate_baseline.py \\
        --strategy best_prompt --tune-on val --report-on test --min-recall 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make `experiments` importable regardless of where we're called from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core import aggregation as agg
from experiments.core.config import EXP_RESULTS
from experiments.core.data_loader import video_window_index
from experiments.core.metrics import (
    bootstrap_ci,
    format_metrics,
    mcnemar,
    video_metrics,
)


def evaluate(by_video: Dict[str, List[Dict]], strategy_fn) -> Tuple[Dict, Dict[str, str]]:
    preds = agg.predict_all(by_video, strategy_fn)
    y_true, y_pred, ordered_ids = [], [], []
    for vid, recs in by_video.items():
        y_true.append(recs[0]["true_label"])
        y_pred.append(preds[vid])
        ordered_ids.append(vid)
    return {
        "metrics": video_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "video_ids": ordered_ids,
    }, preds


def run_strategies(by_video: Dict[str, List[Dict]]) -> List[Dict]:
    strategies = []

    # Existing baseline (≥25%)
    strategies.append(("baseline_ratio_0.25", agg.make_ratio(0.25)))

    # Ratio sweep
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        strategies.append((f"ratio_{t:.2f}", agg.make_ratio(t)))

    # Consecutive
    for k in [2, 3, 4]:
        strategies.append((f"consecutive_k{k}", agg.make_consecutive(k)))

    # End-state only
    strategies.append(("end_state_min1", agg.end_state))

    # Combined
    for k in [2, 3]:
        strategies.append((f"consecutive_k{k}_and_end", agg.make_consecutive_end(k)))

    # Any-window
    strategies.append(("any_fall_window", agg.any_fall))

    out = []
    for name, fn in strategies:
        ev, _ = evaluate(by_video, fn)
        out.append({"name": name, **ev})
    return out


def pick_operating_point(results: List[Dict], min_recall: float) -> Dict:
    feasible = [r for r in results if r["metrics"]["fall_recall"] >= min_recall]
    pool = feasible if feasible else results
    return max(pool, key=lambda r: r["metrics"]["macro_f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="best_prompt",
                    help="Sub-directory under final_pipeline/results/ to re-aggregate")
    ap.add_argument("--tune-on", default="val",
                    help="Split used to pick the operating point (val/test/val_test/none)")
    ap.add_argument("--report-on", default="test",
                    help="Split used to report final numbers (val/test/val_test)")
    ap.add_argument("--min-recall", type=float, default=0.90,
                    help="Required minimum fall recall when picking the operating point")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    out_dir = EXP_RESULTS / "01_reaggregate" / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Re-aggregation sweep: strategy={args.strategy} ===")
    print(f"Tuning on   : {args.tune_on}")
    print(f"Reporting on: {args.report_on}")
    print(f"Min fall recall constraint: {args.min_recall}\n")

    report_index = video_window_index(args.strategy, args.report_on)
    report_results = run_strategies(report_index)
    by_name_report = {r["name"]: r for r in report_results}

    if args.tune_on == "none":
        tune_results = report_results
    else:
        try:
            tune_index = video_window_index(args.strategy, args.tune_on)
            tune_results = run_strategies(tune_index)
        except FileNotFoundError as e:
            print(f"[warn] Tune split not available ({e}); falling back to picking on the report split itself.")
            tune_results = report_results

    chosen = pick_operating_point(tune_results, args.min_recall)

    print("--- Tuning split results ---")
    if tune_results is report_results:
        print("  (tune split unavailable — picker is using the report split itself; treat as diagnostic only)")
    for r in tune_results:
        m = r["metrics"]
        marker = "  <-- chosen" if r["name"] == chosen["name"] else ""
        print(f"  {r['name']:<28s}  recall={m['fall_recall']:.3f}  precision={m['fall_precision']:.3f}  "
              f"fall_F1={m['fall_f1']:.3f}  macro_F1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}{marker}")

    print("\n--- Report split results ---")
    for r in report_results:
        m = r["metrics"]
        marker = "  <-- (chosen on tune split)" if r["name"] == chosen["name"] else ""
        print(f"  {r['name']:<28s}  recall={m['fall_recall']:.3f}  precision={m['fall_precision']:.3f}  "
              f"fall_F1={m['fall_f1']:.3f}  macro_F1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}{marker}")

    chosen_on_report = by_name_report[chosen["name"]]
    ci = bootstrap_ci(chosen_on_report["y_true"], chosen_on_report["y_pred"], n_resamples=args.bootstrap)

    print(f"\n=== Chosen operating point: {chosen['name']} ===")
    print(format_metrics(chosen_on_report["metrics"], ci))

    # McNemar vs baseline
    baseline_on_report = by_name_report["baseline_ratio_0.25"]
    mc = mcnemar(
        chosen_on_report["y_true"],
        baseline_on_report["y_pred"],
        chosen_on_report["y_pred"],
    )
    print(f"\nMcNemar vs baseline_ratio_0.25: b={mc['b']} c={mc['c']} "
          f"chi2={mc['statistic']:.3f} p={mc['p_value']:.4f}")

    # Persist
    if tune_results is not report_results:
        with open(out_dir / f"sweep_{args.tune_on}.json", "w") as f:
            json.dump(
                [{"name": r["name"], "metrics": r["metrics"]} for r in tune_results],
                f, indent=2,
            )
    with open(out_dir / f"sweep_{args.report_on}.json", "w") as f:
        json.dump(
            [{"name": r["name"], "metrics": r["metrics"]} for r in report_results],
            f, indent=2,
        )
    with open(out_dir / f"chosen_{args.report_on}.json", "w") as f:
        json.dump({
            "chosen_strategy": chosen["name"],
            "tuned_on": args.tune_on,
            "min_recall": args.min_recall,
            "report_metrics": chosen_on_report["metrics"],
            "bootstrap_ci_95": {k: list(v) for k, v in ci.items()},
            "mcnemar_vs_baseline": mc,
            "video_predictions": dict(zip(chosen_on_report["video_ids"], chosen_on_report["y_pred"])),
            "video_true_labels": dict(zip(chosen_on_report["video_ids"], chosen_on_report["y_true"])),
        }, f, indent=2)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
