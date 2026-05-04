"""
Chain-of-Thought (CoT) Prompting for Fall Detection using DeepSeek API
=======================================================================

Mirrors the prompting logic of the existing chain_of_thought.py pipeline, but
routes all calls to DeepSeek instead of OpenAI. Results are saved to a separate
folder so the original GPT-4o-mini results stay intact for comparison.

Usage:
    export DEEPSEEK_API_KEY="api-key"
    python chain_of_thought_deepseek.py --split test

NOTE on the model identifier:
    DEEPSEEK_MODEL below is the API model string. "deepseek-chat" routes to
    DeepSeek's current general-purpose model. If your key supports a newer
    version (e.g. a V4 release), replace the value of DEEPSEEK_MODEL with
    the exact string from DeepSeek's API reference. Only this one line
    needs to change.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

# Reuse shared utilities and prompt logic from existing modules.
# We are NOT modifying zero_shot.py or chain_of_thought.py — only importing.
from zero_shot import (
    load_metadata,
    sanitize_window_for_llm,
    aggregate_video_predictions,
    compute_metrics,
    print_metrics,
)
from chain_of_thought import (
    COT_SYSTEM_PROMPT,
    COT_TEMPERATURE,
    COT_MAX_TOKENS,
    create_cot_prompt,
    parse_cot_response,
)

# =============================================================================
# DEEPSEEK CONFIG  ←  edit DEEPSEEK_MODEL here if you want a different version
# =============================================================================

DEEPSEEK_MODEL = "deepseek-chat"          # change to a V4 string if available
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Separate results folder so the original chain_of_thought/ results are untouched.
RESULTS_DIR = Path(__file__).parent.parent / "results" / "chain_of_thought_deepseek"


# =============================================================================
# DEEPSEEK API CALL (with CoT settings — higher max_tokens for reasoning)
# =============================================================================

def call_deepseek(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call DeepSeek's OpenAI-compatible chat completions endpoint."""
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=COT_TEMPERATURE,
            max_tokens=COT_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_cot_evaluation(split: str = "test", sample_size: int = None,
                       aggregation: str = "majority"):
    """Run Chain-of-Thought evaluation against DeepSeek."""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    print(f"Loading {split} split...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")

    window_ids = list(metadata.keys())
    if sample_size and sample_size < len(window_ids):
        random.seed(42)
        window_ids = random.sample(window_ids, sample_size)
        print(f"Sampled {sample_size} windows")

    video_windows = defaultdict(list)
    for wid in window_ids:
        video_id = metadata[wid]["video_id"]
        video_windows[video_id].append(wid)

    print(f"Videos: {len(video_windows)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    window_results = []
    video_results = []

    total_windows = len(window_ids)
    processed = 0

    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}

        for wid in sorted(wids):
            window_path = Path(metadata[wid]["window_path"])

            with open(window_path) as f:
                window_data = json.load(f)

            sanitized = sanitize_window_for_llm(window_data)
            user_prompt = create_cot_prompt(sanitized)

            response = call_deepseek(client, COT_SYSTEM_PROMPT, user_prompt)
            prediction, reasoning = parse_cot_response(response)

            window_preds[wid] = prediction

            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": prediction,
                "reasoning": reasoning,
            })

            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_windows}...")

            time.sleep(0.1)

        video_pred = aggregate_video_predictions(window_preds, aggregation)

        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
            "window_count": len(wids),
            "fall_windows": sum(1 for p in window_preds.values() if p == "fall"),
        })

    # Save per-window and per-video results
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)

    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)

    # Compute metrics
    print("\n" + "=" * 60)
    print(f"CHAIN-OF-THOUGHT (DEEPSEEK / {DEEPSEEK_MODEL}) RESULTS")
    print("=" * 60)

    window_true = [r["true_label"] for r in window_results]
    window_pred = [r["predicted"] for r in window_results]
    window_metrics = compute_metrics(window_true, window_pred)
    print_metrics(window_metrics, "Window")

    video_true = [r["true_label"] for r in video_results]
    video_pred = [r["predicted"] for r in video_results]
    video_metrics = compute_metrics(video_true, video_pred)
    print_metrics(video_metrics, "Video")

    # Save metrics
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({
            "window_level": window_metrics,
            "video_level": video_metrics,
            "config": {
                "provider": "deepseek",
                "model": DEEPSEEK_MODEL,
                "base_url": DEEPSEEK_BASE_URL,
                "split": split,
                "temperature": COT_TEMPERATURE,
                "aggregation": aggregation,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")

    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought Fall Detection (DeepSeek)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--aggregation", default="majority")
    args = parser.parse_args()

    run_cot_evaluation(
        split=args.split,
        sample_size=args.sample,
        aggregation=args.aggregation,
    )


if __name__ == "__main__":
    main()
