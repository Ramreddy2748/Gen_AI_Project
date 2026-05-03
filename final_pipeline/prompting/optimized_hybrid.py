"""
Optimized Hybrid Fall Detection
================================

Combines rule-based scoring with targeted LLM verification.
Goal: 85%+ accuracy and 85%+ fall recall

Strategy:
1. Strong rule-based detection catches clear cases
2. LLM only reviews borderline cases
3. Safety bias: when in doubt, predict fall
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
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "optimized_hybrid"

MODEL = "gpt-4o"


def extract_features(window_data: Dict) -> Dict:
    """Extract key features from window."""
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
                "body_angle": f.get("body_angle_degrees", 90),
                "velocity": e.get("velocity_hip_y", 0),
                "accel": e.get("acceleration_hip_y", 0),
                "posture": e.get("posture_state", "unknown"),
                "on_ground": e.get("is_on_ground", False),
                "rapid": e.get("is_rapid_descent", False),
                "horizontal": e.get("is_horizontal", False),
            })
        else:
            frames.append(None)
    
    return {
        "frames": frames,
        "trajectory": analysis.get("trajectory_direction", "unknown"),
        "max_vel": analysis.get("max_velocity", 0) or 0,
        "spike": analysis.get("has_velocity_spike", False),
    }


def rule_based_classify(features: Dict) -> Tuple[str, float, List[str]]:
    """
    Rule-based classification with confidence score.
    Returns: (prediction, confidence, reasons)
    
    Confidence:
    - HIGH (>= 0.8): Clear fall or clear no-fall
    - MEDIUM (0.5-0.8): Likely but needs verification  
    - LOW (< 0.5): Uncertain
    """
    frames = [f for f in features["frames"] if f is not None]
    
    if len(frames) < 2:
        return "no_fall", 0.3, ["Insufficient pose data"]
    
    reasons = []
    fall_score = 0
    no_fall_score = 0
    
    last = frames[-1]
    first = frames[0]
    
    # ===== STRONG FALL INDICATORS =====
    
    # 1. Final posture is fallen
    if last["posture"] == "fallen":
        fall_score += 30
        reasons.append("Posture: FALLEN")
    
    # 2. Person on ground at end
    if last["on_ground"]:
        fall_score += 25
        reasons.append("On ground")
    
    # 3. Body horizontal at end
    if last["horizontal"] or last["body_angle"] < 25:
        fall_score += 25
        reasons.append(f"Horizontal body ({last['body_angle']:.0f}°)")
    
    # 4. Rapid descent detected
    if any(f["rapid"] for f in frames):
        fall_score += 20
        reasons.append("Rapid descent")
    
    # 5. High velocity
    if features["max_vel"] > 0.12:
        fall_score += 20
        reasons.append(f"High velocity ({features['max_vel']:.2f})")
    
    # 6. Velocity spike
    if features["spike"]:
        fall_score += 15
        reasons.append("Velocity spike")
    
    # 7. Significant hip drop
    hip_change = last["hip_y"] - first["hip_y"]
    if hip_change > 0.12:
        fall_score += 15
        reasons.append(f"Hip dropped ({hip_change:.2f})")
    
    # 8. Descending trajectory
    if features["trajectory"] == "descending":
        fall_score += 10
        reasons.append("Descending trajectory")
    
    # ===== STRONG NO-FALL INDICATORS =====
    
    # 1. Upright posture maintained
    if all(f["posture"] == "upright" for f in frames):
        no_fall_score += 30
        reasons.append("Upright throughout")
    
    # 2. Body stays vertical
    if all(f["body_angle"] > 60 for f in frames):
        no_fall_score += 25
        reasons.append("Vertical body maintained")
    
    # 3. Low velocity (stable)
    if features["max_vel"] < 0.04:
        no_fall_score += 20
        reasons.append("Stable (low velocity)")
    
    # 4. No flags triggered
    if not any(f["on_ground"] or f["rapid"] or f["horizontal"] for f in frames):
        no_fall_score += 15
        reasons.append("No fall flags")
    
    # 5. Ascending or stable trajectory
    if features["trajectory"] in ["ascending", "stable"]:
        no_fall_score += 10
        reasons.append(f"Trajectory: {features['trajectory']}")
    
    # ===== DECISION =====
    
    # Calculate confidence
    total = fall_score + no_fall_score
    if total == 0:
        return "no_fall", 0.5, ["No clear indicators"]
    
    if fall_score > no_fall_score:
        confidence = fall_score / (fall_score + no_fall_score)
        # Boost confidence for multiple strong indicators
        if fall_score >= 60:
            confidence = min(confidence + 0.15, 0.95)
        return "fall", confidence, reasons
    else:
        confidence = no_fall_score / (fall_score + no_fall_score)
        if no_fall_score >= 50:
            confidence = min(confidence + 0.1, 0.95)
        return "no_fall", confidence, reasons


def llm_verify(client: OpenAI, features: Dict, rule_pred: str, 
               confidence: float, reasons: List[str]) -> str:
    """Use LLM to verify borderline cases."""
    
    system = """You are verifying a fall detection decision.

The rule-based system made a preliminary classification. Your job is to CONFIRM or CORRECT it.

FALL indicators: rapid descent, horizontal body, on ground, velocity spike, fallen posture
NO_FALL indicators: upright posture, vertical body, stable/low velocity, no fall flags

IMPORTANT: For safety, lean toward FALL when uncertain.

Respond with ONLY: "FALL" or "NO_FALL"
"""
    
    frames = features["frames"]
    user = f"Rule-based: {rule_pred.upper()} (confidence: {confidence:.0%})\n"
    user += f"Reasons: {', '.join(reasons)}\n\n"
    user += "Frame data:\n"
    
    for i, f in enumerate(frames, 1):
        if f:
            user += f"F{i}: hip={f['hip_y']:.2f}, angle={f['body_angle']:.0f}°, "
            user += f"vel={f['velocity']:.2f}, posture={f['posture']}"
            flags = []
            if f['on_ground']: flags.append("GROUND")
            if f['rapid']: flags.append("RAPID")
            if f['horizontal']: flags.append("HORIZONTAL")
            if flags:
                user += f" [{','.join(flags)}]"
            user += "\n"
    
    user += f"\nOverall: trajectory={features['trajectory']}, max_vel={features['max_vel']:.2f}\n"
    user += f"\nConfirm or correct: FALL or NO_FALL?"
    
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
        result = response.choices[0].message.content.strip().upper()
        
        if "NO_FALL" in result or "NO FALL" in result:
            return "no_fall"
        elif "FALL" in result:
            return "fall"
        return rule_pred  # Default to rule-based
        
    except Exception as e:
        print(f"API error: {e}")
        return rule_pred


def hybrid_classify(client: OpenAI, window_data: Dict) -> str:
    """Hybrid rule-based + LLM classification."""
    features = extract_features(window_data)
    
    # Rule-based classification
    rule_pred, confidence, reasons = rule_based_classify(features)
    
    # High confidence cases: trust rule-based
    if confidence >= 0.75:
        return rule_pred
    
    # Low-medium confidence: verify with LLM
    return llm_verify(client, features, rule_pred, confidence, reasons)


def run_evaluation(split: str = "test", sample_size: int = None):
    """Run hybrid evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    print("Loading metadata...")
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
    
    # Stats
    rule_only = 0
    llm_verified = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            features = extract_features(data)
            rule_pred, confidence, reasons = rule_based_classify(features)
            
            if confidence >= 0.75:
                pred = rule_pred
                rule_only += 1
            else:
                pred = llm_verify(client, features, rule_pred, confidence, reasons)
                llm_verified += 1
                time.sleep(0.15)
            
            window_preds[wid] = pred
            window_results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
                "rule_pred": rule_pred,
                "confidence": confidence,
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{len(window_ids)}...")
        
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
    print("OPTIMIZED HYBRID RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    print(f"\nProcessing Stats:")
    print(f"  Rule-based only: {rule_only} ({rule_only*100/len(window_ids):.1f}%)")
    print(f"  LLM verified: {llm_verified} ({llm_verified*100/len(window_ids):.1f}%)")
    
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({
            "window_level": w_metrics,
            "video_level": v_metrics,
            "stats": {"rule_only": rule_only, "llm_verified": llm_verified}
        }, f, indent=2)
    
    return w_metrics, v_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    run_evaluation(args.split, args.sample)
