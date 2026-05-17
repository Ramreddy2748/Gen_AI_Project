"""
Two-stage cascade for higher video-level F1 without sacrificing recall.

Stage 1: Best-Prompt safety-first classifier (current high-recall champion).
         Flags candidate videos (any video aggregated as "fall" by --stage1-rule).
Stage 2: Strict FP-filter prompt that only sees the candidate videos. For
         each candidate, we send the *entire timeline* of its windows in a
         single prompt and ask the model to decide whether it's a true fall
         or a sit/bend/lie/kneel false positive.

This is the highest-expected-impact change for video-F1 because it surgically
attacks the FP videos that Best Prompt already misclassifies.

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/scripts/05_two_stage_cascade.py \\
        --stage1-strategy best_prompt --stage1-rule ratio_0.50 \\
        --report-on test --model gpt-4o
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
from experiments.core.config import EXP_RESULTS
from experiments.core.data_loader import load_window, video_window_index
from experiments.core.llm_client import call_with_confidence, make_client
from experiments.core.metrics import bootstrap_ci, format_metrics, mcnemar, video_metrics
from experiments.core.prompts import STAGE2_FP_FILTER, format_window_for_prompt


def stage1_predict(by_video: Dict[str, List[Dict]], rule: str) -> Dict[str, str]:
    """Apply a stage-1 aggregation rule (parsed from a name like 'ratio_0.50')."""
    if rule.startswith("ratio_"):
        t = float(rule.split("_", 1)[1])
        fn = agg.make_ratio(t)
    elif rule.startswith("consecutive_k") and rule.endswith("_and_end"):
        k = int(rule[len("consecutive_k"):rule.find("_and_end")])
        fn = agg.make_consecutive_end(k)
    elif rule.startswith("consecutive_k"):
        k = int(rule[len("consecutive_k"):])
        fn = agg.make_consecutive(k)
    elif rule == "end_state":
        fn = agg.end_state
    elif rule == "any_fall":
        fn = agg.any_fall
    else:
        raise ValueError(f"Unknown stage1 rule: {rule}")
    return agg.predict_all(by_video, fn)


def build_video_timeline_prompt(records: List[Dict]) -> str:
    """Concatenate every window of a video into a single prompt body."""
    parts = ["This video has the following pose-window timeline:"]
    for rec in records:
        wid = rec["window_id"]
        true_label = rec["true_label"]
        win = load_window(wid, "test", true_label) or load_window(wid, "val", true_label)
        if win is None:
            parts.append(f"\n[window {rec['window_index']}] (data missing)")
            continue
        parts.append(f"\n--- window {rec['window_index']} ---")
        parts.append(format_window_for_prompt(win))
    parts.append("\nIs this a real fall or a benign activity? Answer FALL or NO_FALL.")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-strategy", default="best_prompt",
                    help="Sub-dir under final_pipeline/results/")
    ap.add_argument("--stage1-rule", default="ratio_0.50",
                    help="Aggregation rule used for stage 1 (e.g. ratio_0.50, consecutive_k2)")
    ap.add_argument("--report-on", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    out_dir = EXP_RESULTS / "05_two_stage_cascade"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 1 = {args.stage1_strategy} aggregated by {args.stage1_rule}")
    by_video = video_window_index(args.stage1_strategy, args.report_on)
    stage1 = stage1_predict(by_video, args.stage1_rule)

    candidates = [vid for vid, pred in stage1.items() if pred == "fall"]
    print(f"Stage 1 flagged {len(candidates)}/{len(stage1)} candidate videos")

    # Stage 2: only run on candidates.
    client = make_client()
    stage2: Dict[str, Dict] = {}
    cache_path = out_dir / f"stage2_cache_{args.report_on}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            stage2 = {r["video_id"]: r for r in json.load(f)}

    for i, vid in enumerate(candidates):
        if vid in stage2:
            continue
        prompt = build_video_timeline_prompt(by_video[vid])
        label, conf, raw = call_with_confidence(
            client, args.model, STAGE2_FP_FILTER, prompt,
            temperature=0.0, max_tokens=12,
        )
        stage2[vid] = {
            "video_id": vid,
            "stage1_pred": "fall",
            "stage2_pred": label,
            "stage2_confidence_fall": conf,
            "raw": raw,
            "true_label": by_video[vid][0]["true_label"],
        }
        if (i + 1) % 5 == 0:
            with open(cache_path, "w") as f:
                json.dump(list(stage2.values()), f, indent=2)
        time.sleep(0.15)
    with open(cache_path, "w") as f:
        json.dump(list(stage2.values()), f, indent=2)

    # Final cascade decision: video is fall iff stage1==fall AND stage2!=no_fall
    final_preds: Dict[str, str] = {}
    for vid, s1 in stage1.items():
        if s1 != "fall":
            final_preds[vid] = "no_fall"
        else:
            s2 = stage2.get(vid, {}).get("stage2_pred", "fall")
            final_preds[vid] = "fall" if s2 == "fall" else "no_fall"

    y_true = [by_video[v][0]["true_label"] for v in stage1]
    y_pred = [final_preds[v] for v in stage1]
    y_pred_stage1_only = [stage1[v] for v in stage1]

    m_stage1 = video_metrics(y_true, y_pred_stage1_only)
    m_cascade = video_metrics(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred, n_resamples=args.bootstrap)
    mc = mcnemar(y_true, y_pred_stage1_only, y_pred)

    print("\n--- Stage-1 only ---")
    print(format_metrics(m_stage1))
    print("\n--- Cascade (Stage-1 -> Stage-2) ---")
    print(format_metrics(m_cascade, ci))
    print(f"\nMcNemar (Stage-1 vs Cascade): b={mc['b']} c={mc['c']} "
          f"chi2={mc['statistic']:.3f} p={mc['p_value']:.4f}")

    with open(out_dir / f"final_{args.report_on}.json", "w") as f:
        json.dump({
            "stage1_strategy": args.stage1_strategy,
            "stage1_rule": args.stage1_rule,
            "model": args.model,
            "stage1_metrics": m_stage1,
            "cascade_metrics": m_cascade,
            "bootstrap_ci_95": {k: list(v) for k, v in ci.items()},
            "mcnemar_stage1_vs_cascade": mc,
            "predictions": final_preds,
            "true_labels": dict(zip(stage1.keys(), y_true)),
        }, f, indent=2)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
