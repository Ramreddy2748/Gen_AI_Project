"""
Self-Consistency Prompting for Fall Detection
==============================================

Generates multiple reasoning paths with temperature sampling,
then uses majority voting to select the final answer.

This reduces variance and improves reliability.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python self_consistency.py --split test --num-samples 5
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from zero_shot import (
    load_metadata, sanitize_window_for_llm,
    aggregate_video_predictions, compute_metrics, print_metrics,
    DATA_DIR, WINDOWS_DIR, MODEL
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "self_consistency"

# Higher temperature for diverse samples
SC_TEMPERATURE = 0.7
SC_MAX_TOKENS = 400
DEFAULT_NUM_SAMPLES = 5

# =============================================================================
# SELF-CONSISTENCY PROMPTING
# =============================================================================

SC_SYSTEM_PROMPT = """You are an expert fall detection system analyzing human pose data.

Analyze the 3-frame pose sequence and determine if the person is FALLING or NOT FALLING.

Think through your reasoning, considering:
1. Position changes (hip_y increasing = moving down, body_angle decreasing = more horizontal)
2. Motion patterns (high velocity = rapid movement, acceleration spikes)
3. Posture changes (upright → transitioning → fallen)
4. Fall indicators (rapid_descent, on_ground, horizontal flags)

Key distinction:
- Controlled sitting/bending: gradual, low velocity, intentional
- Fall: sudden, high velocity, uncontrolled, ending on ground

After your analysis, conclude with exactly: "ANSWER: FALL" or "ANSWER: NO_FALL"
"""


def create_sc_prompt(sanitized_data: Dict) -> str:
    """Create prompt for self-consistency sampling."""
    prompt = "Analyze this pose sequence:\n\n"
    
    for frame in sanitized_data["frames"]:
        prompt += f"Frame {frame['frame_number']}:\n"
        
        if frame.get("pose_detected", True) and "position" in frame:
            hip_y = frame['position'].get('hip_y') or 0
            body_angle = frame['position'].get('body_angle') or 0
            vel_hip = frame['motion'].get('velocity_hip') or 0
            accel = frame['motion'].get('acceleration') or 0
            
            prompt += f"  hip_y={hip_y:.3f}, body_angle={body_angle:.1f}°\n"
            prompt += f"  velocity={vel_hip:.3f}, acceleration={accel:.3f}\n"
            prompt += f"  posture={frame.get('posture', 'unknown')}\n"
            prompt += f"  flags: rapid_descent={frame['flags'].get('rapid_descent', False)}, "
            prompt += f"on_ground={frame['flags'].get('on_ground', False)}, "
            prompt += f"horizontal={frame['flags'].get('horizontal', False)}\n"
        else:
            prompt += "  Pose not detected\n"
    
    summary = sanitized_data["window_summary"]
    max_vel = summary.get('max_velocity') or 0
    prompt += f"\nSummary: trajectory={summary.get('trajectory', 'unknown')}, "
    prompt += f"max_velocity={max_vel:.3f}, velocity_spike={summary.get('has_velocity_spike', False)}\n"
    
    prompt += "\nProvide your reasoning and ANSWER:"
    
    return prompt


def parse_sc_response(response: str) -> str:
    """Parse response to extract prediction."""
    response_upper = response.upper()
    
    # Look for ANSWER: pattern
    if "ANSWER:" in response_upper:
        match = re.search(r'ANSWER:\s*(FALL|NO_FALL|NO FALL)', response_upper)
        if match:
            result = match.group(1)
            return "no_fall" if "NO" in result else "fall"
    
    # Fallback
    if "NO_FALL" in response_upper or "NO FALL" in response_upper:
        return "no_fall"
    elif "FALL" in response_upper:
        return "fall"
    
    return "unknown"


def call_openai_multi(client: OpenAI, system_prompt: str, user_prompt: str, 
                      num_samples: int) -> List[str]:
    """Call OpenAI API multiple times with temperature sampling."""
    responses = []
    
    for _ in range(num_samples):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=SC_TEMPERATURE,
                max_tokens=SC_MAX_TOKENS,
            )
            responses.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"API Error: {e}")
            responses.append("ERROR")
        
        time.sleep(0.05)  # Small delay between samples
    
    return responses


def majority_vote(predictions: List[str]) -> str:
    """Get majority vote from predictions."""
    valid = [p for p in predictions if p in ["fall", "no_fall"]]
    
    if not valid:
        return "unknown"
    
    counter = Counter(valid)
    most_common = counter.most_common(1)[0]
    
    return most_common[0]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_sc_evaluation(split: str = "test", sample_size: int = None,
                      num_samples: int = DEFAULT_NUM_SAMPLES,
                      aggregation: str = "majority"):
    """Run Self-Consistency evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Loading {split} split...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    print(f"Using {num_samples} samples per window for self-consistency")
    
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
            user_prompt = create_sc_prompt(sanitized)
            
            # Get multiple samples
            responses = call_openai_multi(client, SC_SYSTEM_PROMPT, user_prompt, num_samples)
            
            # Parse each response
            sample_predictions = [parse_sc_response(r) for r in responses]
            
            # Majority vote
            final_prediction = majority_vote(sample_predictions)
            
            window_preds[wid] = final_prediction
            
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": final_prediction,
                "sample_predictions": sample_predictions,
                "vote_distribution": dict(Counter(sample_predictions)),
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
    print("SELF-CONSISTENCY PROMPTING RESULTS")
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
                "num_samples": num_samples,
                "temperature": SC_TEMPERATURE,
                "aggregation": aggregation,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Self-Consistency Fall Detection")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of samples per window")
    parser.add_argument("--aggregation", default="majority")
    args = parser.parse_args()
    
    run_sc_evaluation(
        split=args.split,
        sample_size=args.sample,
        num_samples=args.num_samples,
        aggregation=args.aggregation
    )


if __name__ == "__main__":
    main()
