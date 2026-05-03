"""
Enhanced Few-Shot v2 with Retry Logic and Optimizations
========================================================

Improvements:
- Retry logic for rate limits
- Condensed examples to reduce token usage
- Better error handling

Target: 80%+ accuracy
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

from zero_shot import (
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "enhanced_few_shot_v2"

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 50
MAX_RETRIES = 3
RETRY_DELAY = 3.0


def interpret_features(frame: Dict) -> str:
    """Create concise feature interpretation."""
    features = frame.get("features", {})
    enhanced = frame.get("enhanced", {})
    
    if not frame.get("pose_detected", False):
        return "not visible"
    
    hip_y = features.get("hip_y", 0)
    body_angle = features.get("body_angle_degrees", 90)
    velocity = enhanced.get("velocity_hip_y", 0)
    posture = enhanced.get("posture_state", "unknown")
    on_ground = enhanced.get("is_on_ground", False)
    rapid = enhanced.get("is_rapid_descent", False)
    horizontal = enhanced.get("is_horizontal", False)
    
    parts = []
    
    # Position
    if hip_y > 0.7:
        parts.append("LOW position")
    elif hip_y > 0.55:
        parts.append("mid position")
    else:
        parts.append("HIGH position")
    
    # Angle
    if body_angle < 30:
        parts.append("HORIZONTAL body")
    elif body_angle < 50:
        parts.append("tilted body")
    else:
        parts.append("upright body")
    
    # Velocity
    if velocity and abs(velocity) > 0.12:
        parts.append("RAPID movement")
    elif velocity and abs(velocity) > 0.05:
        parts.append("moderate movement")
    else:
        parts.append("stable")
    
    # Alerts
    if rapid:
        parts.append("⚠️FALLING")
    if on_ground:
        parts.append("⚠️ON_GROUND")
    if horizontal:
        parts.append("⚠️LYING")
    
    return ", ".join(parts)


def create_compact_description(window_data: Dict) -> str:
    """Create compact description for lower token usage."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    lines = []
    for i, key in enumerate(["frame_1", "frame_2", "frame_3"], 1):
        frame = sequence.get(key, {})
        lines.append(f"F{i}: {interpret_features(frame)}")
    
    trajectory = analysis.get("trajectory_direction", "unknown")
    max_vel = analysis.get("max_velocity", 0) or 0
    spike = analysis.get("has_velocity_spike", False)
    
    lines.append(f"Trajectory: {trajectory}, MaxVel: {max_vel:.2f}, Spike: {spike}")
    
    return "\n".join(lines)


def load_compact_examples(num_per_class: int = 3) -> List[Dict]:
    """Load compact examples."""
    examples = []
    
    for label, label_name in [("fall", "FALL"), ("no_fall", "NO_FALL")]:
        label_dir = WINDOWS_DIR / "train" / label
        if not label_dir.exists():
            continue
        
        all_windows = []
        for video_dir in label_dir.iterdir():
            if video_dir.is_dir():
                windows = sorted(video_dir.glob("*.json"))
                if windows:
                    mid_idx = len(windows) // 2
                    all_windows.append(windows[mid_idx])
        
        random.seed(42 if label == "fall" else 43)
        selected = random.sample(all_windows, min(num_per_class, len(all_windows)))
        
        for wf in selected:
            with open(wf) as f:
                data = json.load(f)
            examples.append({
                "description": create_compact_description(data),
                "label": label_name
            })
    
    random.seed(44)
    random.shuffle(examples)
    return examples


SYSTEM_PROMPT = """You are a fall detection AI. Analyze 3-frame sequences.

FALL indicators: ⚠️FALLING, ⚠️ON_GROUND, ⚠️LYING, RAPID movement, HORIZONTAL body, LOW position
NO_FALL indicators: stable, upright body, HIGH/mid position, no alerts

Respond ONLY with: FALL or NO_FALL"""


def create_prompt(examples: List[Dict], query: str) -> str:
    """Create compact prompt."""
    prompt = "Examples:\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"{i}. {ex['description']}\n→ {ex['label']}\n\n"
    
    prompt += f"---\nClassify:\n{query}\n→"
    return prompt


def call_with_retry(client: OpenAI, system: str, user: str) -> str:
    """Call API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"API Error: {e}")
                return "ERROR"
    return "ERROR"


def parse_response(response: str) -> str:
    response = response.upper().strip()
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


def run_evaluation(split: str = "test", sample_size: int = None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    print("Loading examples...")
    examples = load_compact_examples(3)
    print(f"Loaded {len(examples)} examples")
    
    print(f"Loading {split} split...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    
    window_ids = list(metadata.keys())
    if sample_size:
        random.seed(42)
        window_ids = random.sample(window_ids, min(sample_size, len(window_ids)))
        print(f"Sampled {sample_size}")
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[metadata[wid]["video_id"]].append(wid)
    
    print(f"Videos: {len(video_windows)}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    processed = 0
    total = len(window_ids)
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            desc = create_compact_description(data)
            prompt = create_prompt(examples, desc)
            
            response = call_with_retry(client, SYSTEM_PROMPT, prompt)
            pred = parse_response(response)
            
            window_preds[wid] = pred
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{total}...")
            
            time.sleep(0.2)  # Slower to avoid rate limits
        
        video_pred = aggregate_video_predictions(window_preds, "majority")
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
        })
    
    # Save
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Metrics
    print("\n" + "="*60)
    print("ENHANCED FEW-SHOT v2 RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({"window_level": w_metrics, "video_level": v_metrics}, f, indent=2)
    
    print(f"\nSaved to: {RESULTS_DIR}")
    return w_metrics, v_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    run_evaluation(parser.parse_args().split, parser.parse_args().sample)
