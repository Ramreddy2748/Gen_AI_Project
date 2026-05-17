"""
Confidence-weighted classification + video aggregation.

For each window in `--split`:
    - call Stage-1 safety-first prompt with logprobs=True
    - record (label, fall_confidence)

For each video:
    - sum fall_confidence over windows
    - tune a threshold on `--tune-on` to maximise macro F1
      with fall_recall >= --min-recall, then evaluate on --report-on.

NOTE: requires OPENAI_API_KEY. Cost: 1 chat completion per window in
the evaluated splits. For the existing balanced test set (272 windows)
this is on the order of $0.30 with gpt-4o.

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/scripts/04_confidence_weighted.py \\
        --tune-on val --report-on test --model gpt-4o --min-recall 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.core import aggregation as agg
from experiments.core.config import EXP_RESULTS, WINDOWS_BALANCED_DIR
from experiments.core.data_loader import load_window
from experiments.core.llm_client import call_with_confidence, make_client
from experiments.core.metrics import bootstrap_ci, format_metrics, video_metrics
from experiments.core.prompts import STAGE1_SAFETY_FIRST, format_window_for_prompt


def collect_window_ids(split: str) -> List[Dict]:
    out: List[Dict] = []
    base = WINDOWS_BALANCED_DIR / split
    for label_dir in base.iterdir() if base.exists() else []:
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for window_file in sorted(video_dir.glob("*.json")):
                out.append({
                    "window_id": window_file.stem,
                    "video_id": video_dir.name,
                    "label": label,
                    "path": window_file,
                })
    return out


def run_stage1(client, model: str, windows: List[Dict], cache_path: Path) -> List[Dict]:
    """Call Stage-1 on each window, caching to disk so we can resume."""
    cached: Dict[str, Dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = {r["window_id"]: r for r in json.load(f)}

    results: List[Dict] = []
    for i, w in enumerate(windows):
        wid = w["window_id"]
        if wid in cached:
            results.append(cached[wid])
            continue
        with open(w["path"]) as f:
            data = json.load(f)
        user = format_window_for_prompt(data)
        label, conf, raw = call_with_confidence(client, model, STAGE1_SAFETY_FIRST, user)
        rec = {
            "window_id": wid,
            "video_id": w["video_id"],
            "true_label": w["label"],
            "predicted": label,
            "confidence": conf,
            "raw_response": raw,
        }
        results.append(rec)
        cached[wid] = rec
        if (i + 1) % 25 == 0:
            print(f"  [stage1] {i + 1}/{len(windows)} done; caching")
            with open(cache_path, "w") as f:
                json.dump(list(cached.values()), f, indent=2)
        time.sleep(0.08)
    with open(cache_path, "w") as f:
        json.dump(list(cached.values()), f, indent=2)
    return results


def group_by_video(records: List[Dict]) -> Dict[str, List[Dict]]:
    by_video: Dict[str, List[Dict]] = {}
    for r in records:
        by_video.setdefault(r["video_id"], []).append(r)
    return by_video


def sweep_and_pick(by_video: Dict[str, List[Dict]], min_recall: float):
    best = None
    for t in [round(0.5 * x, 2) for x in range(1, 21)]:  # 0.5, 1.0, ..., 10.0
        preds = agg.predict_all(by_video, agg.make_confidence_weighted(t))
        y_true = [recs[0]["true_label"] for recs in by_video.values()]
        y_pred = [preds[v] for v in by_video]
        m = video_metrics(y_true, y_pred)
        if m["fall_recall"] >= min_recall and (best is None or m["macro_f1"] > best["metrics"]["macro_f1"]):
            best = {"threshold": t, "metrics": m, "y_true": y_true, "y_pred": y_pred,
                    "video_ids": list(by_video.keys())}
    if best is None:
        # fall back to highest macro f1 overall
        for t in [round(0.5 * x, 2) for x in range(1, 21)]:
            preds = agg.predict_all(by_video, agg.make_confidence_weighted(t))
            y_true = [recs[0]["true_label"] for recs in by_video.values()]
            y_pred = [preds[v] for v in by_video]
            m = video_metrics(y_true, y_pred)
            if best is None or m["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"threshold": t, "metrics": m, "y_true": y_true, "y_pred": y_pred,
                        "video_ids": list(by_video.keys())}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--tune-on", default="val", choices=["train", "val", "test"])
    ap.add_argument("--report-on", default="test", choices=["train", "val", "test"])
    ap.add_argument("--min-recall", type=float, default=0.90)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    out_dir = EXP_RESULTS / "04_confidence_weighted"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = make_client()

    print(f"Collecting windows for tune={args.tune_on} report={args.report_on}")
    tune_windows = collect_window_ids(args.tune_on)
    report_windows = collect_window_ids(args.report_on)
    print(f"  tune  windows: {len(tune_windows)}")
    print(f"  report windows: {len(report_windows)}")

    print("Running Stage-1 with logprobs (this is the expensive step)...")
    tune_records = run_stage1(client, args.model, tune_windows, out_dir / f"stage1_{args.tune_on}.json")
    report_records = run_stage1(client, args.model, report_windows, out_dir / f"stage1_{args.report_on}.json")

    tune_by_video = group_by_video(tune_records)
    report_by_video = group_by_video(report_records)

    chosen = sweep_and_pick(tune_by_video, args.min_recall)
    t = chosen["threshold"]
    print(f"\nChosen threshold (tune={args.tune_on}): {t}")
    print(format_metrics(chosen["metrics"]))

    # Evaluate on report
    preds = agg.predict_all(report_by_video, agg.make_confidence_weighted(t))
    y_true = [recs[0]["true_label"] for recs in report_by_video.values()]
    y_pred = [preds[v] for v in report_by_video]
    final = video_metrics(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred, n_resamples=args.bootstrap)
    print(f"\nReport split ({args.report_on}) using threshold={t}:")
    print(format_metrics(final, ci))

    with open(out_dir / f"final_{args.report_on}.json", "w") as f:
        json.dump({
            "chosen_threshold": t,
            "tune_metrics": chosen["metrics"],
            "report_metrics": final,
            "bootstrap_ci_95": {k: list(v) for k, v in ci.items()},
            "video_predictions": dict(zip(report_by_video.keys(), y_pred)),
            "video_true_labels": dict(zip(report_by_video.keys(), y_true)),
        }, f, indent=2)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
