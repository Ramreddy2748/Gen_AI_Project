"""
Few-Shot Prompting for Fall Detection
=====================================

Provides examples of fall and no-fall patterns to guide the LLM.
This should improve recall compared to zero-shot.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python few_shot.py --split test --num-examples 3
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

# Import shared utilities from zero_shot
from zero_shot import (
    load_metadata, sanitize_window_for_llm, parse_llm_response,
    aggregate_video_predictions, compute_metrics, print_metrics,
    DATA_DIR, WINDOWS_DIR, MODEL, TEMPERATURE, MAX_TOKENS
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "few_shot"

# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

def load_training_examples(num_fall: int = 3, num_nofall: int = 3) -> List[Dict]:
    """Load diverse examples from training set."""
    examples = []
    
    # Load fall examples
    fall_dir = WINDOWS_DIR / "train" / "fall"
    if fall_dir.exists():
        fall_videos = list(fall_dir.iterdir())
        random.seed(42)
        selected_videos = random.sample(fall_videos, min(num_fall, len(fall_videos)))
        
        for video_dir in selected_videos:
            windows = list(video_dir.glob("*.json"))
            if windows:
                # Pick a window from middle of video (more likely to show actual fall)
                mid_idx = len(windows) // 2
                window_file = sorted(windows)[mid_idx]
                
                with open(window_file) as f:
                    data = json.load(f)
                
                examples.append({
                    "data": sanitize_window_for_llm(data),
                    "label": "FALL",
                    "reasoning": "Rapid downward hip movement, body angle decreasing, transitioning to fallen posture"
                })
    
    # Load no-fall examples
    nofall_dir = WINDOWS_DIR / "train" / "no_fall"
    if nofall_dir.exists():
        nofall_videos = list(nofall_dir.iterdir())
        random.seed(43)
        selected_videos = random.sample(nofall_videos, min(num_nofall, len(nofall_videos)))
        
        for video_dir in selected_videos:
            windows = list(video_dir.glob("*.json"))
            if windows:
                window_file = random.choice(windows)
                
                with open(window_file) as f:
                    data = json.load(f)
                
                examples.append({
                    "data": sanitize_window_for_llm(data),
                    "label": "NO_FALL",
                    "reasoning": "Stable body position, controlled movement, maintaining upright posture"
                })
    
    # Shuffle to mix fall and no-fall
    random.seed(44)
    random.shuffle(examples)
    
    return examples


def format_example(example: Dict, idx: int) -> str:
    """Format a single example for the prompt."""
    data = example["data"]
    
    text = f"\n--- Example {idx} ---\n"
    
    for frame in data["frames"]:
        text += f"Frame {frame['frame_number']}:\n"
        
        if frame.get("pose_detected", True) and "position" in frame:
            hip_y = frame['position'].get('hip_y') or 0
            body_angle = frame['position'].get('body_angle') or 0
            vel_hip = frame['motion'].get('velocity_hip') or 0
            accel = frame['motion'].get('acceleration') or 0
            
            text += f"  Position: hip_y={hip_y:.3f}, body_angle={body_angle:.1f}°\n"
            text += f"  Motion: velocity={vel_hip:.3f}, acceleration={accel:.3f}\n"
            text += f"  Posture: {frame.get('posture', 'unknown')}\n"
            text += f"  Flags: rapid_descent={frame['flags'].get('rapid_descent', False)}, "
            text += f"on_ground={frame['flags'].get('on_ground', False)}\n"
        else:
            text += "  Pose not detected\n"
    
    summary = data["window_summary"]
    max_vel = summary.get('max_velocity') or 0
    text += f"Window: trajectory={summary.get('trajectory', 'unknown')}, max_velocity={max_vel:.3f}\n"
    text += f"Classification: {example['label']}\n"
    text += f"Reasoning: {example['reasoning']}\n"
    
    return text


# =============================================================================
# FEW-SHOT PROMPTING
# =============================================================================

FEW_SHOT_SYSTEM_PROMPT = """You are an expert fall detection system analyzing human pose data from video frames.

Your task is to determine if a person is FALLING or NOT FALLING based on pose keypoint data from 3 consecutive frames.

I will show you several labeled examples first, then ask you to classify a new sequence.

Key patterns to look for:
- FALL: Rapid downward movement (high velocity_hip), body angle decreasing toward 0°, posture becoming "fallen", on_ground=True
- NO_FALL: Stable or controlled movements, body angle staying near 90°, posture remaining "upright", normal walking/sitting patterns

IMPORTANT: Controlled sitting or bending has gradual velocity changes. Falls have SUDDEN, UNCONTROLLED descent.

After analyzing the examples, respond to the new case with ONLY: "FALL" or "NO_FALL"
"""


def create_few_shot_prompt(examples: List[Dict], query_data: Dict) -> str:
    """Create few-shot prompt with examples and query."""
    prompt = "Here are labeled examples:\n"
    
    for idx, example in enumerate(examples, 1):
        prompt += format_example(example, idx)
    
    prompt += "\n" + "="*50 + "\n"
    prompt += "Now classify this NEW sequence:\n\n"
    
    # Format query
    for frame in query_data["frames"]:
        prompt += f"Frame {frame['frame_number']}:\n"
        
        if frame.get("pose_detected", True) and "position" in frame:
            hip_y = frame['position'].get('hip_y') or 0
            body_angle = frame['position'].get('body_angle') or 0
            vel_hip = frame['motion'].get('velocity_hip') or 0
            accel = frame['motion'].get('acceleration') or 0
            
            prompt += f"  Position: hip_y={hip_y:.3f}, body_angle={body_angle:.1f}°\n"
            prompt += f"  Motion: velocity={vel_hip:.3f}, acceleration={accel:.3f}\n"
            prompt += f"  Posture: {frame.get('posture', 'unknown')}\n"
            prompt += f"  Flags: rapid_descent={frame['flags'].get('rapid_descent', False)}, "
            prompt += f"on_ground={frame['flags'].get('on_ground', False)}\n"
        else:
            prompt += "  Pose not detected\n"
    
    summary = query_data["window_summary"]
    max_vel = summary.get('max_velocity') or 0
    prompt += f"Window: trajectory={summary.get('trajectory', 'unknown')}, max_velocity={max_vel:.3f}\n"
    
    prompt += "\nClassification (respond with ONLY 'FALL' or 'NO_FALL'):"
    
    return prompt


def call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_few_shot_evaluation(split: str = "test", sample_size: int = None,
                            num_examples: int = 3, aggregation: str = "majority"):
    """Run few-shot evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load examples from training set
    print(f"Loading {num_examples} fall + {num_examples} no-fall examples...")
    examples = load_training_examples(num_examples, num_examples)
    print(f"Loaded {len(examples)} examples")
    
    # Load test metadata
    print(f"Loading {split} split...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    
    window_ids = list(metadata.keys())
    if sample_size and sample_size < len(window_ids):
        random.seed(42)
        window_ids = random.sample(window_ids, sample_size)
        print(f"Sampled {sample_size} windows")
    
    # Group by video
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
            user_prompt = create_few_shot_prompt(examples, sanitized)
            
            response = call_openai(client, FEW_SHOT_SYSTEM_PROMPT, user_prompt)
            prediction = parse_llm_response(response)
            
            window_preds[wid] = prediction
            
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": prediction,
                "raw_response": response,
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
    print("FEW-SHOT PROMPTING RESULTS")
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
                "num_examples": num_examples * 2,
                "aggregation": aggregation,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Fall Detection")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=3, help="Examples per class")
    parser.add_argument("--aggregation", default="majority")
    args = parser.parse_args()
    
    run_few_shot_evaluation(
        split=args.split,
        sample_size=args.sample,
        num_examples=args.num_examples,
        aggregation=args.aggregation
    )


if __name__ == "__main__":
    main()
