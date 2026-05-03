"""
Optimized v2 - Balanced High Performance
=========================================

Target: 85%+ accuracy AND 85%+ fall recall

Improvements:
1. Better threshold calibration
2. Multi-feature scoring
3. Smarter LLM prompting for edge cases
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "optimized_v2"
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


def compute_scores(features: Dict) -> Tuple[float, float, List[str]]:
    """
    Compute fall and no-fall scores separately.
    Returns: (fall_score, nofall_score, reasons)
    """
    frames = [f for f in features["frames"] if f]
    reasons = []
    
    if len(frames) < 2:
        return 0, 30, ["No pose data"]
    
    fall_score = 0
    nofall_score = 0
    
    last = frames[-1]
    first = frames[0]
    
    # ===== FALL SCORING =====
    
    # Critical indicators (high weight)
    if last["posture"] == "fallen":
        fall_score += 35
        reasons.append("FALLEN posture")
    
    if last["ground"]:
        fall_score += 30
        reasons.append("On GROUND")
    
    if last["horiz"]:
        fall_score += 25
        reasons.append("HORIZONTAL")
    
    # Movement indicators
    if any(f["rapid"] for f in frames):
        fall_score += 20
        reasons.append("Rapid descent")
    
    if features["spike"]:
        fall_score += 15
        reasons.append("Velocity spike")
    
    if features["max_vel"] > 0.1:
        fall_score += 15
        reasons.append(f"High velocity ({features['max_vel']:.2f})")
    
    # Position changes
    hip_drop = last["hip_y"] - first["hip_y"]
    if hip_drop > 0.1:
        fall_score += 12
        reasons.append(f"Hip drop ({hip_drop:.2f})")
    
    angle_drop = first["angle"] - last["angle"]
    if angle_drop > 30:
        fall_score += 12
        reasons.append(f"Angle decreased ({angle_drop:.0f}°)")
    
    if features["trajectory"] == "descending":
        fall_score += 8
        reasons.append("Descending")
    
    # ===== NO-FALL SCORING =====
    
    # Stability indicators
    if all(f["posture"] == "upright" for f in frames):
        nofall_score += 35
        reasons.append("Upright maintained")
    
    if all(f["angle"] > 55 for f in frames):
        nofall_score += 25
        reasons.append("Vertical body")
    
    if features["max_vel"] < 0.03:
        nofall_score += 20
        reasons.append("Very stable")
    elif features["max_vel"] < 0.06:
        nofall_score += 10
        reasons.append("Mostly stable")
    
    if not any(f["ground"] or f["rapid"] or f["horiz"] for f in frames):
        nofall_score += 20
        reasons.append("No fall flags")
    
    if features["trajectory"] in ["ascending", "stable"]:
        nofall_score += 10
        reasons.append(f"Trajectory: {features['trajectory']}")
    
    return fall_score, nofall_score, reasons


def rule_classify(features: Dict) -> Tuple[str, float, List[str]]:
    """Rule-based classification."""
    fall_score, nofall_score, reasons = compute_scores(features)
    
    total = fall_score + nofall_score
    if total == 0:
        return "no_fall", 0.5, reasons
    
    # Decision logic - Lower thresholds to catch more falls
    if fall_score >= 40:  # Moderate fall signal
        conf = min(0.95, 0.6 + fall_score / 150)
        return "fall", conf, reasons
    
    if nofall_score >= 60 and fall_score < 25:  # Strong no-fall
        conf = min(0.95, 0.7 + (nofall_score - 60) / 100)
        return "no_fall", conf, reasons
    
    # Moderate cases
    if fall_score > nofall_score:
        conf = fall_score / total
        return "fall", conf, reasons
    else:
        conf = nofall_score / total
        return "no_fall", conf, reasons


def llm_classify(client: OpenAI, features: Dict, rule_pred: str, 
                 fall_score: float, nofall_score: float, reasons: List[str]) -> str:
    """LLM for borderline cases with balanced prompting."""
    
    system = """You are analyzing a fall detection case.

Given frame-by-frame pose data, determine: FALL or NO_FALL

FALL = rapid uncontrolled descent, person ends horizontal/on ground
NO_FALL = controlled movement, person stays upright/vertical

Key signals:
- Posture "fallen" + on_ground + horizontal = almost certainly FALL
- Rapid descent + velocity spike = likely FALL
- All upright + stable velocity = likely NO_FALL
- Sitting down slowly = NO_FALL (controlled)

Be balanced. Respond: FALL or NO_FALL"""

    frames = features["frames"]
    user = f"Scores: FALL={fall_score:.0f}, NO_FALL={nofall_score:.0f}\n"
    user += f"Rule prediction: {rule_pred.upper()}\n\n"
    
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
    
    user += f"\nmax_vel={features['max_vel']:.3f}, spike={features['spike']}\n"
    user += "Classification:"
    
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
        return rule_pred
        
    except Exception as e:
        return rule_pred


def hybrid_classify(client: OpenAI, window_data: Dict) -> Tuple[str, dict]:
    """Hybrid classification."""
    features = extract_features(window_data)
    fall_score, nofall_score, reasons = compute_scores(features)
    rule_pred, conf, _ = rule_classify(features)
    
    stats = {"rule_only": False, "fall_score": fall_score, "nofall_score": nofall_score}
    
    # High confidence: trust rule-based
    if conf >= 0.75:
        stats["rule_only"] = True
        return rule_pred, stats
    
    # LLM for uncertain cases
    pred = llm_classify(client, features, rule_pred, fall_score, nofall_score, reasons)
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
    rule_only = 0
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            pred, stats = hybrid_classify(client, data)
            
            if stats["rule_only"]:
                rule_only += 1
            else:
                time.sleep(0.1)
            
            window_preds[wid] = pred
            window_results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
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
    print("OPTIMIZED v2 RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    print(f"\nRule-only: {rule_only}/{len(window_ids)} ({rule_only*100/len(window_ids):.1f}%)")
    
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({"window": w_metrics, "video": v_metrics}, f, indent=2)
    
    return w_metrics, v_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.split, args.sample)
