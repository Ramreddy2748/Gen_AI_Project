"""
Aggressive Fall Detection with High Recall
============================================

Priority: Catch ALL falls (high recall), then use LLM to filter false positives

Strategy:
1. Aggressive rule-based: predict FALL if ANY fall indicator present
2. LLM filters the predicted falls to remove false positives
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "aggressive"
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


def has_fall_indicators(features: Dict) -> Tuple[bool, List[str], int]:
    """
    Check if ANY fall indicators are present.
    Returns: (has_indicators, reasons, severity 0-100)
    """
    frames = [f for f in features["frames"] if f]
    
    if not frames:
        return False, ["No pose data"], 0
    
    indicators = []
    severity = 0
    
    last = frames[-1]
    first = frames[0]
    
    # Critical indicators (high severity)
    if last["posture"] == "fallen":
        indicators.append("FALLEN posture")
        severity += 40
    
    if last["ground"]:
        indicators.append("On GROUND")
        severity += 35
    
    if last["horiz"]:
        indicators.append("HORIZONTAL body")
        severity += 30
    
    # Movement indicators
    if any(f["rapid"] for f in frames):
        indicators.append("Rapid descent")
        severity += 25
    
    if features["spike"]:
        indicators.append("Velocity spike")
        severity += 20
    
    if features["max_vel"] > 0.08:
        indicators.append(f"High velocity ({features['max_vel']:.2f})")
        severity += 15
    
    # Position indicators
    hip_drop = last["hip_y"] - first["hip_y"]
    if hip_drop > 0.08:
        indicators.append(f"Hip dropped ({hip_drop:.2f})")
        severity += 12
    
    if last["angle"] < 40:
        indicators.append(f"Low angle ({last['angle']:.0f}°)")
        severity += 15
    
    if features["trajectory"] == "descending":
        indicators.append("Descending")
        severity += 10
    
    return len(indicators) > 0, indicators, min(severity, 100)


def is_clearly_not_fall(features: Dict) -> Tuple[bool, List[str]]:
    """Check if this is clearly NOT a fall."""
    frames = [f for f in features["frames"] if f]
    
    if not frames:
        return False, []
    
    reasons = []
    
    # All upright throughout
    if all(f["posture"] == "upright" for f in frames):
        reasons.append("Upright throughout")
    
    # All angles high (vertical body)
    if all(f["angle"] > 65 for f in frames):
        reasons.append("Body vertical")
    
    # Very low velocity
    if features["max_vel"] < 0.03:
        reasons.append("Very stable")
    
    # No fall flags
    if not any(f["ground"] or f["rapid"] or f["horiz"] for f in frames):
        reasons.append("No fall flags")
    
    # Need at least 3 of these for confident no-fall
    return len(reasons) >= 3, reasons


def llm_filter_fall(client: OpenAI, features: Dict, 
                    indicators: List[str], severity: int) -> str:
    """LLM decides if the detected fall indicators are a real fall or false positive."""
    
    system = """You are filtering fall detection predictions.

The rule system flagged this as a POTENTIAL FALL. Your job: confirm if it's a REAL FALL or a FALSE POSITIVE.

REAL FALL characteristics:
- UNCONTROLLED rapid descent (not gradual sitting)
- Person ends up on ground or horizontal
- Multiple strong indicators together

FALSE POSITIVE (not a real fall):
- Controlled sitting down or lying down
- Slow, gradual movement
- Person maintains control throughout
- Just bending over momentarily

Examine the evidence and respond: "FALL" or "NO_FALL"
"""

    frames = features["frames"]
    user = f"Fall indicators detected (severity {severity}/100):\n"
    user += f"- {chr(10).join(indicators)}\n\n"
    user += "Frame data:\n"
    
    for i, f in enumerate(frames, 1):
        if f:
            user += f"F{i}: hip={f['hip_y']:.2f}, angle={f['angle']:.0f}°, "
            user += f"vel={f['vel']:.3f}, posture={f['posture']}"
            flags = []
            if f['ground']: flags.append("GROUND")
            if f['rapid']: flags.append("RAPID")  
            if f['horiz']: flags.append("HORIZ")
            if flags:
                user += f" [{','.join(flags)}]"
            user += "\n"
    
    user += f"\nSequence: trajectory={features['trajectory']}, max_vel={features['max_vel']:.3f}\n"
    user += "\nIs this a REAL FALL or FALSE POSITIVE? Respond: FALL or NO_FALL"
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            max_tokens=30,
        )
        result = response.choices[0].message.content.strip().upper()
        
        if "NO_FALL" in result or "NO FALL" in result or "FALSE" in result:
            return "no_fall"
        return "fall"
        
    except Exception as e:
        # On error, trust the rule-based detection
        return "fall" if severity >= 30 else "no_fall"


def classify(client: OpenAI, window_data: Dict) -> Tuple[str, dict]:
    """Classify with aggressive fall detection + LLM filtering."""
    features = extract_features(window_data)
    
    stats = {"method": "rule"}
    
    # Check if clearly not a fall
    is_clear_nofall, nofall_reasons = is_clearly_not_fall(features)
    if is_clear_nofall:
        stats["reasons"] = nofall_reasons
        return "no_fall", stats
    
    # Check for fall indicators
    has_fall, indicators, severity = has_fall_indicators(features)
    
    if not has_fall:
        return "no_fall", stats
    
    # High severity: definitely a fall
    if severity >= 65:
        stats["method"] = "rule_high_severity"
        stats["indicators"] = indicators
        return "fall", stats
    
    # Medium severity: let LLM decide
    stats["method"] = "llm_filter"
    stats["indicators"] = indicators
    stats["severity"] = severity
    
    pred = llm_filter_fall(client, features, indicators, severity)
    return pred, stats


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
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    
    method_counts = {"rule": 0, "rule_high_severity": 0, "llm_filter": 0}
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            pred, stats = classify(client, data)
            method_counts[stats["method"]] = method_counts.get(stats["method"], 0) + 1
            
            if stats["method"] == "llm_filter":
                time.sleep(0.12)
            
            window_preds[wid] = pred
            window_results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
                "method": stats["method"],
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
    
    # Metrics
    print("\n" + "="*60)
    print("AGGRESSIVE FALL DETECTION RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    print(f"\nMethod distribution:")
    for m, c in method_counts.items():
        print(f"  {m}: {c} ({c*100/len(window_ids):.1f}%)")
    
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({"window": w_metrics, "video": v_metrics, "methods": method_counts}, f, indent=2)
    
    return w_metrics, v_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.split, args.sample)
