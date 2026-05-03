"""
Enhanced Few-Shot Prompting for Fall Detection
===============================================

Key Improvements:
1. Uses GPT-4o (more capable) instead of GPT-4o-mini
2. Converts numerical features to rich textual descriptions
3. Better prompt structure with clearer decision criteria
4. More examples with diverse fall patterns

Target: 80-90% accuracy
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "enhanced_few_shot"

# Use GPT-4o for better performance
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 150

# =============================================================================
# CONVERT NUMERICAL DATA TO RICH TEXT DESCRIPTIONS
# =============================================================================

def interpret_hip_position(hip_y: float) -> str:
    """Convert hip_y to human-readable position."""
    if hip_y is None:
        return "unknown position"
    if hip_y > 0.75:
        return "very low (near ground level)"
    elif hip_y > 0.6:
        return "low (crouching/sitting height)"
    elif hip_y > 0.45:
        return "medium (bent over)"
    else:
        return "high (standing upright)"


def interpret_body_angle(angle: float) -> str:
    """Convert body angle to orientation description."""
    if angle is None:
        return "unknown orientation"
    if angle > 70:
        return "upright/vertical"
    elif angle > 45:
        return "tilted/leaning"
    elif angle > 20:
        return "significantly tilted"
    else:
        return "nearly horizontal/lying down"


def interpret_velocity(velocity: float) -> str:
    """Convert velocity to movement description."""
    if velocity is None:
        return "unknown movement"
    abs_vel = abs(velocity)
    direction = "downward" if velocity > 0 else "upward"
    
    if abs_vel > 0.15:
        return f"rapid {direction} movement (FALL-LIKE)"
    elif abs_vel > 0.08:
        return f"moderate {direction} movement"
    elif abs_vel > 0.03:
        return f"slow {direction} movement"
    else:
        return "minimal/no movement"


def interpret_acceleration(accel: float) -> str:
    """Convert acceleration to force description."""
    if accel is None:
        return "unknown"
    abs_acc = abs(accel)
    
    if abs_acc > 0.1:
        return "HIGH acceleration (impact or rapid change)"
    elif abs_acc > 0.05:
        return "moderate acceleration"
    else:
        return "low/steady acceleration"


def interpret_posture(posture: str, on_ground: bool, horizontal: bool) -> str:
    """Create comprehensive posture description."""
    if posture == "fallen":
        return "FALLEN - person is down"
    elif posture == "transitioning":
        if on_ground:
            return "transitioning while near ground"
        return "transitioning between positions"
    else:
        return "upright/standing"


def create_rich_description(window_data: Dict) -> str:
    """Convert window data to rich textual description."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    description = ""
    
    for i, frame_key in enumerate(["frame_1", "frame_2", "frame_3"], 1):
        frame = sequence.get(frame_key, {})
        
        if not frame.get("pose_detected", False):
            description += f"Frame {i}: Person not clearly visible\n"
            continue
        
        features = frame.get("features", {})
        enhanced = frame.get("enhanced", {})
        
        hip_y = features.get("hip_y")
        body_angle = features.get("body_angle_degrees")
        velocity = enhanced.get("velocity_hip_y")
        accel = enhanced.get("acceleration_hip_y")
        posture = enhanced.get("posture_state", "unknown")
        on_ground = enhanced.get("is_on_ground", False)
        horizontal = enhanced.get("is_horizontal", False)
        rapid = enhanced.get("is_rapid_descent", False)
        
        description += f"Frame {i}:\n"
        description += f"  Body position: {interpret_hip_position(hip_y)}\n"
        description += f"  Body orientation: {interpret_body_angle(body_angle)}\n"
        description += f"  Movement: {interpret_velocity(velocity)}\n"
        description += f"  Acceleration: {interpret_acceleration(accel)}\n"
        description += f"  Posture: {interpret_posture(posture, on_ground, horizontal)}\n"
        
        if rapid:
            description += f"  ⚠️ RAPID DESCENT DETECTED\n"
        if on_ground:
            description += f"  ⚠️ PERSON IS ON/NEAR GROUND\n"
        if horizontal:
            description += f"  ⚠️ BODY IS HORIZONTAL\n"
        
        description += "\n"
    
    # Overall trajectory
    trajectory = analysis.get("trajectory_direction", "unknown")
    max_vel = analysis.get("max_velocity", 0) or 0
    spike = analysis.get("has_velocity_spike", False)
    
    description += "OVERALL SEQUENCE ANALYSIS:\n"
    description += f"  Trajectory: {trajectory}\n"
    
    if max_vel > 0.15:
        description += f"  Maximum velocity: HIGH ({max_vel:.3f}) - FALL INDICATOR\n"
    elif max_vel > 0.08:
        description += f"  Maximum velocity: MODERATE ({max_vel:.3f})\n"
    else:
        description += f"  Maximum velocity: LOW ({max_vel:.3f}) - controlled movement\n"
    
    if spike:
        description += "  ⚠️ VELOCITY SPIKE DETECTED - sudden movement change\n"
    
    return description


# =============================================================================
# LOAD DIVERSE EXAMPLES
# =============================================================================

def load_diverse_examples(num_per_class: int = 4) -> List[Dict]:
    """Load diverse examples covering different fall/no-fall patterns."""
    examples = []
    
    # Get fall examples from different sources
    fall_dir = WINDOWS_DIR / "train" / "fall"
    if fall_dir.exists():
        all_fall_windows = []
        for video_dir in fall_dir.iterdir():
            if video_dir.is_dir():
                windows = list(video_dir.glob("*.json"))
                if windows:
                    # Pick window from middle (likely actual fall moment)
                    sorted_windows = sorted(windows)
                    mid_idx = len(sorted_windows) // 2
                    all_fall_windows.append(sorted_windows[mid_idx])
        
        random.seed(42)
        selected = random.sample(all_fall_windows, min(num_per_class, len(all_fall_windows)))
        
        for window_file in selected:
            with open(window_file) as f:
                data = json.load(f)
            examples.append({
                "description": create_rich_description(data),
                "label": "FALL",
                "explanation": "Rapid downward movement with body transitioning to horizontal/ground position"
            })
    
    # Get no-fall examples
    nofall_dir = WINDOWS_DIR / "train" / "no_fall"
    if nofall_dir.exists():
        all_nofall_windows = []
        for video_dir in nofall_dir.iterdir():
            if video_dir.is_dir():
                windows = list(video_dir.glob("*.json"))
                if windows:
                    all_nofall_windows.append(random.choice(windows))
        
        random.seed(43)
        selected = random.sample(all_nofall_windows, min(num_per_class, len(all_nofall_windows)))
        
        for window_file in selected:
            with open(window_file) as f:
                data = json.load(f)
            examples.append({
                "description": create_rich_description(data),
                "label": "NO_FALL",
                "explanation": "Controlled movement, body remains stable or moves slowly"
            })
    
    random.seed(44)
    random.shuffle(examples)
    return examples


# =============================================================================
# ENHANCED PROMPTING
# =============================================================================

ENHANCED_SYSTEM_PROMPT = """You are an expert fall detection AI analyzing human movement patterns.

Your task: Determine if the person in the sequence is FALLING or performing normal activity.

## CRITICAL FALL INDICATORS (if 2+ present = likely FALL):
1. ⚠️ RAPID DESCENT - sudden fast downward movement
2. ⚠️ BODY HORIZONTAL - person's body becomes horizontal
3. ⚠️ ON GROUND - person ends up at ground level
4. ⚠️ VELOCITY SPIKE - sudden change in movement speed
5. Posture changes from "upright" → "transitioning" → "FALLEN"

## NORMAL ACTIVITY INDICATORS:
1. Slow, controlled movements
2. Body stays upright or tilted (not horizontal)
3. Smooth velocity (no sudden spikes)
4. Person remains at standing/sitting height

## KEY DISTINCTION:
- FALL: UNCONTROLLED, rapid, ends with person on ground/horizontal
- Sitting/bending: CONTROLLED, gradual, person maintains stability

You will see labeled examples first, then classify a new sequence.
Respond with ONLY: "FALL" or "NO_FALL"
"""


def create_enhanced_prompt(examples: List[Dict], query_description: str) -> str:
    """Create enhanced few-shot prompt."""
    prompt = "## LABELED EXAMPLES:\n\n"
    
    for i, ex in enumerate(examples, 1):
        prompt += f"### Example {i}:\n"
        prompt += ex["description"]
        prompt += f"\n**Classification: {ex['label']}**\n"
        prompt += f"Reason: {ex['explanation']}\n"
        prompt += "-" * 40 + "\n\n"
    
    prompt += "=" * 50 + "\n"
    prompt += "## NEW SEQUENCE TO CLASSIFY:\n\n"
    prompt += query_description
    prompt += "\n\n**Your classification (FALL or NO_FALL):**"
    
    return prompt


def call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API with GPT-4o."""
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


def parse_response(response: str) -> str:
    """Parse response."""
    response = response.upper().strip()
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_enhanced_evaluation(split: str = "test", sample_size: int = None,
                            num_examples: int = 4):
    """Run enhanced few-shot evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Using model: {MODEL}")
    print(f"Loading {num_examples} examples per class...")
    examples = load_diverse_examples(num_examples)
    print(f"Loaded {len(examples)} total examples")
    
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
            
            # Create rich description
            description = create_rich_description(window_data)
            user_prompt = create_enhanced_prompt(examples, description)
            
            response = call_openai(client, ENHANCED_SYSTEM_PROMPT, user_prompt)
            prediction = parse_response(response)
            
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
        
        video_pred = aggregate_video_predictions(window_preds, "majority")
        
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
    print("ENHANCED FEW-SHOT RESULTS (GPT-4o + Rich Descriptions)")
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
                "technique": "enhanced_few_shot_with_rich_descriptions"
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=4)
    args = parser.parse_args()
    
    run_enhanced_evaluation(args.split, args.sample, args.num_examples)


if __name__ == "__main__":
    main()
