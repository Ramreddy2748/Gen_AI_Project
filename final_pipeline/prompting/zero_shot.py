"""
Zero-Shot Prompting for Fall Detection
======================================

This script implements zero-shot prompting using OpenAI GPT-4o-mini
for fall detection from sliding window pose features.

Key Features:
1. NO data leakage - labels are kept separate from LLM input
2. Window-level prediction with video-level aggregation
3. Comprehensive evaluation metrics

Usage:
    export OPENAI_API_KEY="your-api-key"
    python zero_shot.py --split test --sample 50
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random

from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WINDOWS_DIR = DATA_DIR / "windows_balanced"
RESULTS_DIR = PROJECT_ROOT / "results" / "zero_shot"

# Model configuration
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 100

# =============================================================================
# METADATA MANAGEMENT (Prevents Data Leakage)
# =============================================================================

def create_metadata_file() -> Path:
    """
    Create metadata file mapping window_id to ground truth label.
    This file is NEVER sent to the LLM - used only for evaluation.
    """
    metadata_path = DATA_DIR / "metadata" / "ground_truth.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for split in ["train", "val", "test"]:
        for label in ["fall", "no_fall"]:
            label_dir = WINDOWS_DIR / split / label
            if not label_dir.exists():
                continue
            
            for video_dir in label_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                for window_file in video_dir.glob("*.json"):
                    window_id = window_file.stem
                    metadata.append({
                        "window_id": window_id,
                        "video_id": video_id,
                        "label": label,
                        "split": split,
                        "window_path": str(window_file),
                    })
    
    # Save metadata
    with open(metadata_path, "w", newline="") as f:
        fieldnames = ["window_id", "video_id", "label", "split", "window_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"Created metadata file: {metadata_path}")
    print(f"Total windows: {len(metadata)}")
    
    return metadata_path


def load_metadata(split: str = None) -> Dict[str, Dict]:
    """Load metadata, optionally filtered by split."""
    metadata_path = DATA_DIR / "metadata" / "ground_truth.csv"
    
    if not metadata_path.exists():
        create_metadata_file()
    
    metadata = {}
    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is None or row["split"] == split:
                metadata[row["window_id"]] = row
    
    return metadata


# =============================================================================
# DATA SANITIZATION (Removes Labels Before LLM)
# =============================================================================

def sanitize_window_for_llm(window_data: Dict) -> Dict:
    """
    Remove all label-related information from window data.
    This prevents data leakage to the LLM.
    """
    # Create a clean copy with only pose features
    sanitized = {
        "frames": []
    }
    
    sequence = window_data.get("sequence", {})
    
    for frame_key in ["frame_1", "frame_2", "frame_3"]:
        frame = sequence.get(frame_key, {})
        
        if frame.get("pose_detected", False):
            clean_frame = {
                "frame_number": int(frame_key.split("_")[1]),
                "timestamp": frame.get("timestamp_seconds", 0),
                "position": {
                    "hip_y": frame.get("features", {}).get("hip_y"),
                    "shoulder_y": frame.get("features", {}).get("shoulder_y"),
                    "body_center_y": frame.get("features", {}).get("body_center_y"),
                    "body_angle": frame.get("features", {}).get("body_angle_degrees"),
                },
                "motion": {
                    "velocity_hip": frame.get("enhanced", {}).get("velocity_hip_y"),
                    "velocity_magnitude": frame.get("enhanced", {}).get("velocity_magnitude"),
                    "acceleration": frame.get("enhanced", {}).get("acceleration_hip_y"),
                },
                "posture": frame.get("enhanced", {}).get("posture_state"),
                "flags": {
                    "rapid_descent": frame.get("enhanced", {}).get("is_rapid_descent"),
                    "on_ground": frame.get("enhanced", {}).get("is_on_ground"),
                    "horizontal": frame.get("enhanced", {}).get("is_horizontal"),
                }
            }
        else:
            clean_frame = {
                "frame_number": int(frame_key.split("_")[1]),
                "pose_detected": False
            }
        
        sanitized["frames"].append(clean_frame)
    
    # Add window-level features (but NOT fall_likelihood_score as it's our computed label)
    analysis = window_data.get("window_analysis", {})
    sanitized["window_summary"] = {
        "pose_detection_rate": analysis.get("pose_detection_rate"),
        "trajectory": analysis.get("trajectory_direction"),
        "max_velocity": analysis.get("max_velocity"),
        "has_velocity_spike": analysis.get("has_velocity_spike"),
    }
    
    return sanitized


# =============================================================================
# ZERO-SHOT PROMPTING
# =============================================================================

ZERO_SHOT_SYSTEM_PROMPT = """You are an expert fall detection system analyzing human pose data from video frames.

Your task is to determine if a person is FALLING or NOT FALLING based on pose keypoint data from 3 consecutive frames.

Key indicators of a FALL:
- Rapid downward movement (high positive velocity_hip)
- Body angle decreasing toward horizontal (body_angle approaching 0)
- Posture transitioning from "upright" to "fallen"
- Person ending up "on_ground" with "horizontal" body position
- Sudden acceleration changes

Key indicators of NO FALL (normal activity):
- Stable or slow movements
- Body remains upright (body_angle near 90 degrees)
- Posture stays "upright" or "transitioning" without reaching "fallen"
- Controlled movements (sitting, bending, walking)

IMPORTANT: Some activities like sitting down or bending over may look similar to falls but are controlled movements. Falls are characterized by UNCONTROLLED rapid descent.

Respond with ONLY one word: "FALL" or "NO_FALL"
"""

def create_zero_shot_prompt(sanitized_data: Dict) -> str:
    """Create the user prompt with sanitized pose data."""
    prompt = "Analyze this 3-frame pose sequence and determine if the person is falling:\n\n"
    
    for frame in sanitized_data["frames"]:
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
            prompt += f"on_ground={frame['flags'].get('on_ground', False)}, "
            prompt += f"horizontal={frame['flags'].get('horizontal', False)}\n"
        else:
            prompt += "  Pose not detected\n"
        prompt += "\n"
    
    summary = sanitized_data["window_summary"]
    max_vel = summary.get('max_velocity') or 0
    prompt += f"Window Summary:\n"
    prompt += f"  Trajectory: {summary.get('trajectory', 'unknown')}\n"
    prompt += f"  Max velocity: {max_vel:.3f}\n"
    prompt += f"  Velocity spike: {summary.get('has_velocity_spike', False)}\n"
    
    prompt += "\nIs this person FALLING or NOT FALLING? Respond with only: FALL or NO_FALL"
    
    return prompt


def call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API and return response."""
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


def parse_llm_response(response: str) -> str:
    """Parse LLM response to get prediction."""
    response = response.upper().strip()
    
    if "NO_FALL" in response or "NO FALL" in response or "NOT FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    else:
        return "unknown"


# =============================================================================
# VIDEO-LEVEL AGGREGATION
# =============================================================================

def aggregate_video_predictions(window_predictions: Dict[str, str], 
                                 strategy: str = "majority") -> str:
    """
    Aggregate window-level predictions to video-level.
    
    Strategies:
    - "majority": >50% fall windows = video is fall
    - "any": any fall window = video is fall
    - "consecutive": >=2 consecutive fall windows = fall
    """
    predictions = list(window_predictions.values())
    fall_count = sum(1 for p in predictions if p == "fall")
    total = len(predictions)
    
    if strategy == "majority":
        return "fall" if fall_count > total / 2 else "no_fall"
    elif strategy == "any":
        return "fall" if fall_count > 0 else "no_fall"
    elif strategy == "consecutive":
        # Check for 2+ consecutive falls
        consecutive = 0
        max_consecutive = 0
        for p in predictions:
            if p == "fall":
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        return "fall" if max_consecutive >= 2 else "no_fall"
    else:
        return "fall" if fall_count > total / 2 else "no_fall"


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute comprehensive evaluation metrics."""
    # Confusion matrix components
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "fall" and p == "fall")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "no_fall" and p == "no_fall")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "no_fall" and p == "fall")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "fall" and p == "no_fall")
    
    # Per-class metrics
    fall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fall_f1 = 2 * fall_precision * fall_recall / (fall_precision + fall_recall) if (fall_precision + fall_recall) > 0 else 0
    
    nofall_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    nofall_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    nofall_f1 = 2 * nofall_precision * nofall_recall / (nofall_precision + nofall_recall) if (nofall_precision + nofall_recall) > 0 else 0
    
    # Overall metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    macro_f1 = (fall_f1 + nofall_f1) / 2
    
    return {
        "confusion_matrix": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        },
        "fall": {
            "precision": round(fall_precision, 4),
            "recall": round(fall_recall, 4),
            "f1": round(fall_f1, 4),
            "support": tp + fn
        },
        "no_fall": {
            "precision": round(nofall_precision, 4),
            "recall": round(nofall_recall, 4),
            "f1": round(nofall_f1, 4),
            "support": tn + fp
        },
        "overall": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "total_samples": tp + tn + fp + fn
        }
    }


def print_metrics(metrics: Dict, level: str = "Window"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{level}-Level Evaluation Results")
    print('='*60)
    
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"                 FALL    NO_FALL")
    print(f"  Actual FALL    {cm['TP']:4d}    {cm['FN']:4d}")
    print(f"  Actual NO_FALL {cm['FP']:4d}    {cm['TN']:4d}")
    
    print("\nPer-Class Metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*50}")
    for cls in ["fall", "no_fall"]:
        m = metrics[cls]
        print(f"  {cls:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['overall']['macro_f1']:.4f}")
    print(f"  Total Samples: {metrics['overall']['total_samples']}")
    print('='*60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_zero_shot_evaluation(split: str = "test", sample_size: int = None,
                              aggregation: str = "majority"):
    """Run zero-shot evaluation on specified split."""
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load metadata
    print(f"Loading {split} split metadata...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    
    # Sample if requested
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
    
    print(f"Videos to process: {len(video_windows)}")
    
    # Results storage
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    
    # Process each video
    total_windows = len(window_ids)
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]  # Ground truth
        window_preds = {}
        
        for wid in sorted(wids):
            window_path = Path(metadata[wid]["window_path"])
            
            # Load window data
            with open(window_path, "r") as f:
                window_data = json.load(f)
            
            # Sanitize (remove labels)
            sanitized = sanitize_window_for_llm(window_data)
            
            # Create prompt
            user_prompt = create_zero_shot_prompt(sanitized)
            
            # Call LLM
            response = call_openai(client, ZERO_SHOT_SYSTEM_PROMPT, user_prompt)
            prediction = parse_llm_response(response)
            
            window_preds[wid] = prediction
            
            # Store window result
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": prediction,
                "raw_response": response,
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_windows} windows...")
            
            # Rate limiting
            time.sleep(0.1)
        
        # Aggregate to video level
        video_pred = aggregate_video_predictions(window_preds, aggregation)
        
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
            "window_count": len(wids),
            "fall_windows": sum(1 for p in window_preds.values() if p == "fall"),
        })
    
    # Save results
    window_results_path = RESULTS_DIR / f"window_results_{split}.json"
    with open(window_results_path, "w") as f:
        json.dump(window_results, f, indent=2)
    
    video_results_path = RESULTS_DIR / f"video_results_{split}.json"
    with open(video_results_path, "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("ZERO-SHOT PROMPTING RESULTS")
    print("="*60)
    
    # Window-level metrics
    window_true = [r["true_label"] for r in window_results]
    window_pred = [r["predicted"] for r in window_results]
    window_metrics = compute_metrics(window_true, window_pred)
    print_metrics(window_metrics, "Window")
    
    # Video-level metrics
    video_true = [r["true_label"] for r in video_results]
    video_pred = [r["predicted"] for r in video_results]
    video_metrics = compute_metrics(video_true, video_pred)
    print_metrics(video_metrics, "Video")
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "window_level": window_metrics,
            "video_level": video_metrics,
            "config": {
                "model": MODEL,
                "split": split,
                "sample_size": sample_size,
                "aggregation": aggregation,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Fall Detection")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None = all)")
    parser.add_argument("--aggregation", default="majority", 
                        choices=["majority", "any", "consecutive"])
    parser.add_argument("--create-metadata", action="store_true", 
                        help="Create metadata file only")
    args = parser.parse_args()
    
    if args.create_metadata:
        create_metadata_file()
        return
    
    run_zero_shot_evaluation(
        split=args.split,
        sample_size=args.sample,
        aggregation=args.aggregation
    )


if __name__ == "__main__":
    main()
