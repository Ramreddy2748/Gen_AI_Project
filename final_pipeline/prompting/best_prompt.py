"""
Best Performing Prompt Strategy
================================

Combines safety-first prompting with optimal aggregation.
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
    load_metadata, compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "best_prompt"
MODEL = "gpt-4o"


BEST_PROMPT = """You are a SAFETY-CRITICAL fall detection system for elderly monitoring.

MISSION: Protect elderly people by detecting falls. Missing a real fall could be life-threatening.

Analyze the 3-frame pose sequence for FALL INDICATORS:

STRONG FALL SIGNS (any 1 = likely fall):
- rapid_descent flag = True
- on_ground flag = True  
- horizontal flag = True
- posture = "fallen"
- velocity > 0.1 (rapid movement)
- body angle < 30° (horizontal)

MODERATE FALL SIGNS (2+ together = likely fall):
- hip_y > 0.6 (person low)
- body angle < 50° (tilted)
- descending trajectory
- velocity spike

CLEAR NO-FALL SIGNS (all needed for NO_FALL):
- posture = "upright" throughout
- body angle > 60° throughout
- velocity < 0.05 (stable)
- no fall flags triggered

When in doubt, choose FALL (safety first).

Respond with ONLY: FALL or NO_FALL"""


def extract_features(window_data: Dict) -> Dict:
    """Extract features."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    frames = []
    for key in ["frame_1", "frame_2", "frame_3"]:
        frame = sequence.get(key, {})
        if frame.get("pose_detected", False):
            f = frame.get("features", {})
            e = frame.get("enhanced", {})
            frames.append({
                "hip_y": f.get("hip_y", 0),
                "angle": f.get("body_angle_degrees", 90),
                "vel": e.get("velocity_hip_y", 0),
                "posture": e.get("posture_state", "unknown"),
                "ground": e.get("is_on_ground", False),
                "rapid": e.get("is_rapid_descent", False),
                "horiz": e.get("is_horizontal", False),
            })
        else:
            frames.append(None)
    
    return {
        "frames": frames,
        "trajectory": analysis.get("trajectory_direction", "unknown"),
        "max_vel": analysis.get("max_velocity", 0) or 0,
        "spike": analysis.get("has_velocity_spike", False),
    }


def format_data(features: Dict) -> str:
    """Format features."""
    text = "Pose Data:\n"
    for i, f in enumerate(features["frames"], 1):
        if f:
            text += f"Frame {i}: hip_y={f['hip_y']:.2f}, angle={f['angle']:.0f}°, "
            text += f"velocity={f['vel']:.3f}, posture={f['posture']}\n"
            text += f"  Flags: rapid_descent={f['rapid']}, on_ground={f['ground']}, horizontal={f['horiz']}\n"
        else:
            text += f"Frame {i}: pose not detected\n"
    
    text += f"\nSequence: trajectory={features['trajectory']}, "
    text += f"max_velocity={features['max_vel']:.3f}, velocity_spike={features['spike']}\n"
    text += "\nClassification:"
    return text


def call_api(client: OpenAI, prompt: str, data: str) -> str:
    """Call API with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": data}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


def parse_response(response: str) -> str:
    """Parse response."""
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


def aggregate_video(preds: List[str], threshold: float = 0.2) -> str:
    """Aggregate with low threshold for safety."""
    fall_count = sum(1 for p in preds if p == "fall")
    return "fall" if fall_count / max(len(preds), 1) >= threshold else "no_fall"


def run_evaluation(split: str = "test", sample_size: int = None):
    """Run evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    metadata = load_metadata(split)
    window_ids = list(metadata.keys())
    
    if sample_size:
        random.seed(42)
        window_ids = random.sample(window_ids, min(sample_size, len(window_ids)))
    
    print(f"Windows: {len(window_ids)}")
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[metadata[wid]["video_id"]].append(wid)
    
    print(f"Videos: {len(video_windows)}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        preds = []
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            features = extract_features(data)
            user_text = format_data(features)
            
            response = call_api(client, BEST_PROMPT, user_text)
            pred = parse_response(response)
            preds.append(pred)
            
            window_results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{len(window_ids)}...")
            
            time.sleep(0.12)
        
        # Aggregate with 25% threshold (slightly higher for better accuracy)
        video_pred = aggregate_video(preds, 0.25)
        
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
            "fall_windows": sum(1 for p in preds if p == "fall"),
            "total_windows": len(preds),
        })
    
    # Save
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Metrics
    print("\n" + "="*60)
    print("BEST PROMPT STRATEGY RESULTS")
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
        json.dump({"window": w_metrics, "video": v_metrics}, f, indent=2)
    
    return w_metrics, v_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.split, args.sample)
