"""
Chain-of-Thought (CoT) Prompting for Fall Detection
====================================================

Encourages step-by-step reasoning before making a decision.
This should improve accuracy by forcing explicit analysis.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python chain_of_thought.py --split test
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from zero_shot import (
    load_metadata, sanitize_window_for_llm,
    aggregate_video_predictions, compute_metrics, print_metrics,
    DATA_DIR, WINDOWS_DIR, MODEL, MAX_TOKENS
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "chain_of_thought"

# Higher temperature for more diverse reasoning
COT_TEMPERATURE = 0.3
COT_MAX_TOKENS = 500  # More tokens for reasoning

# =============================================================================
# CHAIN-OF-THOUGHT PROMPTING
# =============================================================================

COT_SYSTEM_PROMPT = """You are an expert fall detection system analyzing human pose data.

Your task is to determine if a person is FALLING or NOT FALLING based on 3 consecutive frames of pose data.

You MUST think step-by-step before giving your answer:

STEP 1 - Analyze Position Changes:
- Is hip_y increasing (moving down in frame)?
- Is body_angle decreasing (becoming more horizontal)?

STEP 2 - Analyze Motion:
- Is velocity high (>0.1 indicates rapid movement)?
- Is acceleration significant?
- Is there a velocity spike?

STEP 3 - Analyze Posture Progression:
- Does posture change from "upright" → "transitioning" → "fallen"?
- Are fall flags (rapid_descent, on_ground, horizontal) becoming true?

STEP 4 - Distinguish Fall from Controlled Movement:
- Controlled sitting/bending: gradual changes, velocity stays low
- Fall: SUDDEN velocity spike, uncontrolled descent, ending on ground

STEP 5 - Final Decision:
Based on your analysis, conclude with exactly: "FINAL: FALL" or "FINAL: NO_FALL"
"""


def create_cot_prompt(sanitized_data: Dict) -> str:
    """Create CoT prompt asking for step-by-step reasoning."""
    prompt = "Analyze this 3-frame pose sequence step-by-step:\n\n"
    
    for frame in sanitized_data["frames"]:
        prompt += f"Frame {frame['frame_number']}:\n"
        
        if frame.get("pose_detected", True) and "position" in frame:
            hip_y = frame['position'].get('hip_y') or 0
            body_angle = frame['position'].get('body_angle') or 0
            vel_hip = frame['motion'].get('velocity_hip') or 0
            accel = frame['motion'].get('acceleration') or 0
            
            prompt += f"  hip_y={hip_y:.3f}, body_angle={body_angle:.1f}°\n"
            prompt += f"  velocity_hip={vel_hip:.3f}, acceleration={accel:.3f}\n"
            prompt += f"  posture={frame.get('posture', 'unknown')}\n"
            prompt += f"  rapid_descent={frame['flags'].get('rapid_descent', False)}, "
            prompt += f"on_ground={frame['flags'].get('on_ground', False)}, "
            prompt += f"horizontal={frame['flags'].get('horizontal', False)}\n"
        else:
            prompt += "  Pose not detected\n"
        prompt += "\n"
    
    summary = sanitized_data["window_summary"]
    max_vel = summary.get('max_velocity') or 0
    prompt += f"Window Summary:\n"
    prompt += f"  trajectory={summary.get('trajectory', 'unknown')}\n"
    prompt += f"  max_velocity={max_vel:.3f}\n"
    prompt += f"  velocity_spike={summary.get('has_velocity_spike', False)}\n"
    
    prompt += "\nNow analyze step-by-step and give your FINAL decision:"
    
    return prompt


def parse_cot_response(response: str) -> tuple:
    """Parse CoT response to extract reasoning and prediction."""
    response_upper = response.upper()
    
    # Look for FINAL: pattern
    if "FINAL:" in response_upper:
        if "FINAL: NO_FALL" in response_upper or "FINAL:NO_FALL" in response_upper:
            prediction = "no_fall"
        elif "FINAL: FALL" in response_upper or "FINAL:FALL" in response_upper:
            prediction = "fall"
        else:
            # Check what comes after FINAL:
            match = re.search(r'FINAL:\s*(FALL|NO_FALL|NO FALL)', response_upper)
            if match:
                result = match.group(1)
                prediction = "no_fall" if "NO" in result else "fall"
            else:
                prediction = "unknown"
    else:
        # Fallback: look for last occurrence
        if "NO_FALL" in response_upper or "NO FALL" in response_upper:
            prediction = "no_fall"
        elif "FALL" in response_upper:
            prediction = "fall"
        else:
            prediction = "unknown"
    
    return prediction, response


def call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API with CoT settings."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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
    """Run Chain-of-Thought evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
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
            
            response = call_openai(client, COT_SYSTEM_PROMPT, user_prompt)
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
    
    # Save results
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    
    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("CHAIN-OF-THOUGHT PROMPTING RESULTS")
    print("="*60)
    
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
                "model": MODEL,
                "split": split,
                "temperature": COT_TEMPERATURE,
                "aggregation": aggregation,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought Fall Detection")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--aggregation", default="majority")
    args = parser.parse_args()
    
    run_cot_evaluation(
        split=args.split,
        sample_size=args.sample,
        aggregation=args.aggregation
    )


if __name__ == "__main__":
    main()
