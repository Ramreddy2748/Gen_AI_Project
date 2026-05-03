"""
Systematic Prompt Experimentation
==================================

Test multiple variations to find optimal configuration:
1. Different prompt framings
2. Different example counts
3. Different aggregation strategies
4. Different decision thresholds
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from zero_shot import (
    load_metadata, compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"
MODEL = "gpt-4o"


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


# =============================================================================
# PROMPT VARIATIONS
# =============================================================================

PROMPT_V1_SAFETY = """You are a SAFETY-CRITICAL fall detection system for elderly care.

PRIORITY: Never miss a real fall. False alarms are acceptable, missed falls are NOT.

Analyze the 3-frame pose sequence. Look for ANY of these fall indicators:
- Rapid downward movement (velocity > 0.08)
- Body becoming horizontal (angle < 45°)
- Person ending up low/on ground (hip_y > 0.6)
- Posture transitioning to "fallen"
- Any fall flags (rapid_descent, on_ground, horizontal)

If you see ANY strong indicator, classify as FALL.
Only classify as NO_FALL if the person is clearly stable and upright throughout.

Respond: FALL or NO_FALL"""

PROMPT_V2_BALANCED = """You are analyzing pose data for fall detection.

A FALL shows: rapid uncontrolled descent, body going horizontal, person ending on ground.
NOT A FALL: controlled sitting, bending, walking, standing - all have stable controlled movement.

Key difference: Falls are SUDDEN and UNCONTROLLED. Normal activities are GRADUAL and CONTROLLED.

Examine the 3 frames and classify: FALL or NO_FALL"""

PROMPT_V3_CHECKLIST = """Fall Detection Checklist Analysis.

Check each indicator in the 3-frame sequence:
□ Velocity spike (sudden movement change)
□ Rapid descent (velocity > 0.1)
□ Body horizontal (angle < 30°)
□ Person on ground (hip_y > 0.7)
□ Posture = "fallen"
□ rapid_descent flag = True
□ on_ground flag = True
□ horizontal flag = True

RULE: If 3+ indicators are checked, classify as FALL.
Otherwise, classify as NO_FALL.

After checking, respond: FALL or NO_FALL"""

PROMPT_V4_EXPERT = """You are a biomechanics expert analyzing human movement.

In a fall:
1. Center of mass drops rapidly (hip_y increases quickly)
2. Body orientation changes from vertical to horizontal (angle decreases)
3. Movement is uncontrolled (high velocity, acceleration)
4. Final position: person horizontal or on ground

In normal activity (sitting, bending):
1. Movement is slow and controlled
2. Person maintains balance throughout
3. No sudden velocity changes

Analyze the pose sequence with your expert knowledge.
Respond: FALL or NO_FALL"""


def format_data(features: Dict) -> str:
    """Format features for prompt."""
    text = ""
    for i, f in enumerate(features["frames"], 1):
        if f:
            text += f"F{i}: hip={f['hip_y']:.2f}, angle={f['angle']:.0f}°, "
            text += f"vel={f['vel']:.3f}, posture={f['posture']}"
            flags = []
            if f['ground']: flags.append("GROUND")
            if f['rapid']: flags.append("RAPID")
            if f['horiz']: flags.append("HORIZ")
            if flags:
                text += f" [{','.join(flags)}]"
            text += "\n"
        else:
            text += f"F{i}: not detected\n"
    
    text += f"Overall: trajectory={features['trajectory']}, "
    text += f"max_vel={features['max_vel']:.3f}, spike={features['spike']}"
    return text


def call_api(client: OpenAI, system: str, user: str) -> str:
    """Call API."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        time.sleep(2)
        return "ERROR"


def parse_response(response: str) -> str:
    """Parse response."""
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


# =============================================================================
# AGGREGATION STRATEGIES
# =============================================================================

def aggregate_majority(preds: List[str]) -> str:
    """Standard majority voting."""
    fall_count = sum(1 for p in preds if p == "fall")
    return "fall" if fall_count > len(preds) / 2 else "no_fall"


def aggregate_any(preds: List[str]) -> str:
    """Any fall window = video is fall."""
    return "fall" if any(p == "fall" for p in preds) else "no_fall"


def aggregate_threshold(preds: List[str], threshold: float) -> str:
    """Fall if >= threshold fraction are fall."""
    fall_count = sum(1 for p in preds if p == "fall")
    return "fall" if fall_count / len(preds) >= threshold else "no_fall"


def aggregate_consecutive(preds: List[str], n: int = 2) -> str:
    """Fall if N consecutive fall predictions."""
    consecutive = 0
    for p in preds:
        if p == "fall":
            consecutive += 1
            if consecutive >= n:
                return "fall"
        else:
            consecutive = 0
    return "no_fall"


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(client: OpenAI, prompt: str, prompt_name: str,
                   metadata: Dict, window_ids: List[str],
                   aggregation: str = "majority") -> Dict:
    """Run single experiment configuration."""
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[metadata[wid]["video_id"]].append(wid)
    
    window_results = []
    video_results = []
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        preds = []
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            features = extract_features(data)
            user_text = format_data(features)
            
            response = call_api(client, prompt, user_text)
            pred = parse_response(response)
            preds.append(pred)
            
            window_results.append({
                "true": metadata[wid]["label"],
                "pred": pred,
            })
            
            time.sleep(0.1)
        
        # Aggregate
        if aggregation == "majority":
            video_pred = aggregate_majority(preds)
        elif aggregation == "any":
            video_pred = aggregate_any(preds)
        elif aggregation == "consecutive2":
            video_pred = aggregate_consecutive(preds, 2)
        elif aggregation == "consecutive3":
            video_pred = aggregate_consecutive(preds, 3)
        elif aggregation == "threshold30":
            video_pred = aggregate_threshold(preds, 0.30)
        elif aggregation == "threshold20":
            video_pred = aggregate_threshold(preds, 0.20)
        else:
            video_pred = aggregate_majority(preds)
        
        video_results.append({
            "true": video_label,
            "pred": video_pred,
        })
    
    # Compute metrics
    w_true = [r["true"] for r in window_results]
    w_pred = [r["pred"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    
    v_true = [r["true"] for r in video_results]
    v_pred = [r["pred"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    
    return {
        "prompt": prompt_name,
        "aggregation": aggregation,
        "window": w_metrics,
        "video": v_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--prompt", default="all", 
                        choices=["all", "safety", "balanced", "checklist", "expert"])
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    metadata = load_metadata("test")
    window_ids = list(metadata.keys())
    
    random.seed(42)
    window_ids = random.sample(window_ids, min(args.sample, len(window_ids)))
    print(f"Testing on {len(window_ids)} windows")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    prompts = {
        "safety": PROMPT_V1_SAFETY,
        "balanced": PROMPT_V2_BALANCED,
        "checklist": PROMPT_V3_CHECKLIST,
        "expert": PROMPT_V4_EXPERT,
    }
    
    aggregations = ["majority", "any", "consecutive2", "threshold30", "threshold20"]
    
    results = []
    
    if args.prompt == "all":
        test_prompts = prompts.items()
    else:
        test_prompts = [(args.prompt, prompts[args.prompt])]
    
    for prompt_name, prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Testing prompt: {prompt_name}")
        print('='*50)
        
        for agg in aggregations:
            print(f"  Aggregation: {agg}...", end=" ", flush=True)
            
            result = run_experiment(client, prompt, prompt_name, 
                                   metadata, window_ids, agg)
            results.append(result)
            
            w_acc = result["window"]["overall"]["accuracy"]
            w_recall = result["window"]["fall"]["recall"]
            v_acc = result["video"]["overall"]["accuracy"]
            v_recall = result["video"]["fall"]["recall"]
            
            print(f"W:{w_acc:.1%}/{w_recall:.1%} V:{v_acc:.1%}/{v_recall:.1%}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Prompt':<12} {'Aggregation':<15} {'W.Acc':>8} {'W.Recall':>10} {'V.Acc':>8} {'V.Recall':>10}")
    print("-"*80)
    
    # Sort by video accuracy + recall combined
    results.sort(key=lambda x: x["video"]["overall"]["accuracy"] + x["video"]["fall"]["recall"], reverse=True)
    
    for r in results:
        print(f"{r['prompt']:<12} {r['aggregation']:<15} "
              f"{r['window']['overall']['accuracy']:>7.1%} "
              f"{r['window']['fall']['recall']:>9.1%} "
              f"{r['video']['overall']['accuracy']:>7.1%} "
              f"{r['video']['fall']['recall']:>9.1%}")
    
    # Best result
    best = results[0]
    print("\n" + "="*80)
    print(f"BEST CONFIG: {best['prompt']} + {best['aggregation']}")
    print(f"  Video Accuracy: {best['video']['overall']['accuracy']:.1%}")
    print(f"  Video Fall Recall: {best['video']['fall']['recall']:.1%}")
    print(f"  Video Macro F1: {best['video']['overall']['macro_f1']:.3f}")
    
    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
