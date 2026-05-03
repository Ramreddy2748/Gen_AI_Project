"""
Experimental Prompting Strategies for 90%+ Accuracy and Recall
===============================================================

Approaches:
1. Rule-based feature interpretation with LLM verification
2. Confidence-calibrated prompting
3. Two-stage classification
4. Feature-weighted explicit prompting
5. Ensemble voting
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from zero_shot import (
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "experimental"

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_RETRIES = 3


def call_api(client: OpenAI, system: str, user: str, temp: float = 0.0) -> str:
    """Call API with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temp,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


def extract_key_features(window_data: Dict) -> Dict:
    """Extract the most discriminative features."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    features = {
        "frames": [],
        "trajectory": analysis.get("trajectory_direction", "unknown"),
        "max_velocity": analysis.get("max_velocity", 0) or 0,
        "velocity_spike": analysis.get("has_velocity_spike", False),
    }
    
    for key in ["frame_1", "frame_2", "frame_3"]:
        frame = sequence.get(key, {})
        if frame.get("pose_detected", False):
            f = frame.get("features", {})
            e = frame.get("enhanced", {})
            features["frames"].append({
                "hip_y": f.get("hip_y", 0),
                "body_angle": f.get("body_angle_degrees", 90),
                "velocity": e.get("velocity_hip_y", 0),
                "posture": e.get("posture_state", "unknown"),
                "on_ground": e.get("is_on_ground", False),
                "rapid_descent": e.get("is_rapid_descent", False),
                "horizontal": e.get("is_horizontal", False),
            })
        else:
            features["frames"].append(None)
    
    return features


def compute_fall_score(features: Dict) -> Tuple[float, List[str]]:
    """
    Compute a rule-based fall score from 0-100.
    Returns score and list of reasons.
    """
    score = 0
    reasons = []
    
    frames = [f for f in features["frames"] if f is not None]
    if len(frames) < 2:
        return 0, ["Insufficient pose data"]
    
    # Check trajectory
    if features["trajectory"] == "descending":
        score += 15
        reasons.append("Descending trajectory")
    
    # Check velocity spike
    if features["velocity_spike"]:
        score += 20
        reasons.append("Velocity spike detected")
    
    # Check max velocity
    max_vel = features["max_velocity"]
    if max_vel > 0.15:
        score += 25
        reasons.append(f"High velocity ({max_vel:.2f})")
    elif max_vel > 0.08:
        score += 10
        reasons.append(f"Moderate velocity ({max_vel:.2f})")
    
    # Check final frame
    last = frames[-1]
    
    if last["on_ground"]:
        score += 20
        reasons.append("Person on ground")
    
    if last["horizontal"]:
        score += 15
        reasons.append("Body horizontal")
    
    if last["posture"] == "fallen":
        score += 20
        reasons.append("Fallen posture")
    
    if last["body_angle"] < 30:
        score += 15
        reasons.append(f"Low body angle ({last['body_angle']:.0f}°)")
    
    # Check for rapid descent in any frame
    if any(f["rapid_descent"] for f in frames):
        score += 15
        reasons.append("Rapid descent detected")
    
    # Check hip position change
    if len(frames) >= 2:
        hip_change = frames[-1]["hip_y"] - frames[0]["hip_y"]
        if hip_change > 0.15:  # Significant drop
            score += 15
            reasons.append(f"Hip dropped ({hip_change:.2f})")
    
    return min(score, 100), reasons


# =============================================================================
# STRATEGY 1: Rule-Based + LLM Verification
# =============================================================================

VERIFY_SYSTEM = """You are verifying a fall detection decision.

Given:
1. Pose data from 3 frames
2. A preliminary fall score (0-100) and reasons

Your task: Confirm or correct the classification.

Rules:
- Score >= 50 with strong indicators (on_ground, horizontal, rapid_descent) = FALL
- Score >= 70 = likely FALL
- Score < 30 = likely NO_FALL
- Controlled sitting/bending has gradual movement, no sudden velocity

Respond: "FALL" or "NO_FALL" """


def strategy_rule_based_verify(client: OpenAI, window_data: Dict) -> str:
    """Rule-based scoring with LLM verification."""
    features = extract_key_features(window_data)
    score, reasons = compute_fall_score(features)
    
    # Build prompt
    user = f"Fall Score: {score}/100\n"
    user += f"Reasons: {', '.join(reasons) if reasons else 'None'}\n\n"
    user += "Frame Data:\n"
    
    for i, f in enumerate(features["frames"], 1):
        if f:
            user += f"F{i}: hip_y={f['hip_y']:.2f}, angle={f['body_angle']:.0f}°, "
            user += f"vel={f['velocity']:.3f}, posture={f['posture']}\n"
    
    user += f"\nOverall: trajectory={features['trajectory']}, "
    user += f"max_vel={features['max_velocity']:.3f}\n"
    user += "\nClassify as FALL or NO_FALL:"
    
    response = call_api(client, VERIFY_SYSTEM, user)
    return parse_response(response)


# =============================================================================
# STRATEGY 2: Confidence-Calibrated Prompting
# =============================================================================

CONFIDENCE_SYSTEM = """You are an expert fall detection system.

Analyze the pose sequence and provide:
1. Your classification: FALL or NO_FALL
2. Your confidence: HIGH (>90%), MEDIUM (70-90%), or LOW (<70%)

FALL indicators (need 2+ for high confidence):
- Rapid downward velocity (>0.1)
- Body angle < 30° (horizontal)
- Hip position > 0.7 (low/ground level)
- Posture = "fallen"
- Flags: on_ground, rapid_descent, horizontal

NO_FALL indicators:
- Stable velocity (<0.05)
- Body angle > 60° (upright)
- Controlled movement pattern

Format: "CLASSIFICATION: [FALL/NO_FALL], CONFIDENCE: [HIGH/MEDIUM/LOW]"
"""


def strategy_confidence_calibrated(client: OpenAI, window_data: Dict) -> str:
    """Use confidence to adjust threshold."""
    features = extract_key_features(window_data)
    
    user = "Analyze:\n"
    for i, f in enumerate(features["frames"], 1):
        if f:
            user += f"F{i}: hip={f['hip_y']:.2f}, angle={f['body_angle']:.0f}°, "
            user += f"vel={f['velocity']:.3f}, posture={f['posture']}, "
            flags = []
            if f['on_ground']: flags.append("ON_GROUND")
            if f['rapid_descent']: flags.append("RAPID")
            if f['horizontal']: flags.append("HORIZONTAL")
            user += f"flags=[{','.join(flags)}]\n"
    
    user += f"Trajectory: {features['trajectory']}, MaxVel: {features['max_velocity']:.3f}"
    
    response = call_api(client, CONFIDENCE_SYSTEM, user)
    
    # Parse with confidence adjustment
    resp_upper = response.upper()
    
    # If low confidence on NO_FALL, lean toward FALL (safety)
    if "NO_FALL" in resp_upper and "LOW" in resp_upper:
        # Check if rule-based score is moderate
        score, _ = compute_fall_score(features)
        if score >= 40:
            return "fall"
    
    return parse_response(response)


# =============================================================================
# STRATEGY 3: Two-Stage Classification
# =============================================================================

STAGE1_SYSTEM = """Quick fall screening. Is there ANY indication of a fall?

Look for:
- Rapid movement
- Person going down
- Horizontal body position
- Ground contact

Respond: "POSSIBLE_FALL" or "UNLIKELY_FALL"
"""

STAGE2_SYSTEM = """Detailed fall analysis. Confirm if this is a real fall.

A REAL FALL has:
- UNCONTROLLED rapid descent (not sitting down slowly)
- Person ends up on ground or horizontal
- Velocity spike (sudden change)

NOT A FALL:
- Controlled sitting/lying down
- Slow, deliberate movement
- Person maintains control

Respond: "CONFIRMED_FALL" or "NOT_A_FALL"
"""


def strategy_two_stage(client: OpenAI, window_data: Dict) -> str:
    """Two-stage screening then confirmation."""
    features = extract_key_features(window_data)
    
    # Stage 1: Quick screening
    user1 = "Quick check:\n"
    for i, f in enumerate(features["frames"], 1):
        if f:
            user1 += f"F{i}: pos={f['hip_y']:.2f}, vel={f['velocity']:.3f}, "
            user1 += f"angle={f['body_angle']:.0f}°\n"
    
    response1 = call_api(client, STAGE1_SYSTEM, user1)
    
    if "UNLIKELY" in response1.upper():
        # Double-check with rule score
        score, _ = compute_fall_score(features)
        if score < 30:
            return "no_fall"
        # If rule score is moderate, proceed to stage 2
    
    # Stage 2: Detailed analysis
    user2 = "Detailed analysis:\n"
    for i, f in enumerate(features["frames"], 1):
        if f:
            user2 += f"Frame {i}:\n"
            user2 += f"  Position: hip_y={f['hip_y']:.3f} (1=bottom, 0=top)\n"
            user2 += f"  Body angle: {f['body_angle']:.1f}° (90=upright, 0=horizontal)\n"
            user2 += f"  Velocity: {f['velocity']:.3f} (>0 = downward)\n"
            user2 += f"  Posture: {f['posture']}\n"
            user2 += f"  Alerts: ground={f['on_ground']}, rapid={f['rapid_descent']}, horiz={f['horizontal']}\n"
    
    user2 += f"\nSequence: trajectory={features['trajectory']}, max_velocity={features['max_velocity']:.3f}"
    user2 += f", velocity_spike={features['velocity_spike']}\n"
    user2 += "\nIs this a CONFIRMED_FALL or NOT_A_FALL?"
    
    response2 = call_api(client, STAGE2_SYSTEM, user2)
    
    if "CONFIRMED" in response2.upper() or "FALL" in response2.upper().replace("NOT_A_FALL", ""):
        return "fall"
    return "no_fall"


# =============================================================================
# STRATEGY 4: Explicit Feature Checklist
# =============================================================================

CHECKLIST_SYSTEM = """Fall detection checklist analysis.

Go through each indicator and check if present:

□ Rapid downward velocity (vel > 0.1) 
□ High velocity spike in sequence
□ Body angle becoming horizontal (<30°)
□ Hip position low (>0.65)
□ Posture state = "fallen"
□ On ground flag = True
□ Rapid descent flag = True
□ Horizontal body flag = True
□ Descending trajectory

CLASSIFICATION RULES:
- 4+ indicators checked = FALL
- 2-3 indicators with (on_ground OR horizontal OR fallen) = FALL
- Otherwise = NO_FALL

After checking, state: "INDICATORS: X/9, CLASSIFICATION: [FALL/NO_FALL]"
"""


def strategy_checklist(client: OpenAI, window_data: Dict) -> str:
    """Explicit feature checklist approach."""
    features = extract_key_features(window_data)
    
    user = "Check each indicator:\n\n"
    
    frames = [f for f in features["frames"] if f]
    if not frames:
        return "no_fall"
    
    last = frames[-1]
    max_vel = max(abs(f["velocity"]) for f in frames) if frames else 0
    
    user += f"1. Rapid velocity: max={max_vel:.3f} (threshold: 0.1)\n"
    user += f"2. Velocity spike: {features['velocity_spike']}\n"
    user += f"3. Body angle low: final={last['body_angle']:.0f}° (threshold: 30°)\n"
    user += f"4. Hip position low: final={last['hip_y']:.2f} (threshold: 0.65)\n"
    user += f"5. Posture fallen: {last['posture']}\n"
    user += f"6. On ground: {last['on_ground']}\n"
    user += f"7. Rapid descent: {any(f['rapid_descent'] for f in frames)}\n"
    user += f"8. Horizontal body: {last['horizontal']}\n"
    user += f"9. Descending trajectory: {features['trajectory']}\n"
    
    user += "\nCount indicators and classify:"
    
    response = call_api(client, CHECKLIST_SYSTEM, user)
    return parse_response(response)


# =============================================================================
# STRATEGY 5: Ensemble Voting
# =============================================================================

def strategy_ensemble(client: OpenAI, window_data: Dict) -> str:
    """Combine multiple strategies with voting."""
    votes = []
    
    # Rule-based score
    features = extract_key_features(window_data)
    score, reasons = compute_fall_score(features)
    votes.append("fall" if score >= 50 else "no_fall")
    
    # Confidence-calibrated
    votes.append(strategy_confidence_calibrated(client, window_data))
    
    # Two-stage
    votes.append(strategy_two_stage(client, window_data))
    
    # Count votes
    fall_votes = sum(1 for v in votes if v == "fall")
    
    # Majority with bias toward fall (safety)
    return "fall" if fall_votes >= 2 else "no_fall"


def parse_response(response: str) -> str:
    """Parse response to get prediction."""
    resp = response.upper()
    if "NO_FALL" in resp or "NO FALL" in resp or "NOT_A_FALL" in resp or "NOT A FALL" in resp:
        return "no_fall"
    elif "FALL" in resp:
        return "fall"
    return "unknown"


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_strategy(client: OpenAI, strategy_fn, strategy_name: str, 
                      split: str = "test", sample_size: int = None):
    """Evaluate a strategy."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {strategy_name}")
    print('='*60)
    
    metadata = load_metadata(split)
    window_ids = list(metadata.keys())
    
    if sample_size:
        random.seed(42)
        window_ids = random.sample(window_ids, min(sample_size, len(window_ids)))
    
    print(f"Windows: {len(window_ids)}")
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[metadata[wid]["video_id"]].append(wid)
    
    window_results = []
    video_results = []
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            pred = strategy_fn(client, data)
            window_preds[wid] = pred
            
            window_results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"  Processed {processed}/{len(window_ids)}...")
            
            time.sleep(0.15)
        
        video_pred = aggregate_video_predictions(window_preds, "majority")
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
        })
    
    # Compute metrics
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    
    print(f"\n{strategy_name} Results:")
    print(f"  Window Accuracy: {w_metrics['overall']['accuracy']:.1%}")
    print(f"  Window Fall Recall: {w_metrics['fall']['recall']:.1%}")
    print(f"  Window Fall Precision: {w_metrics['fall']['precision']:.1%}")
    print(f"  Window Macro F1: {w_metrics['overall']['macro_f1']:.3f}")
    print(f"  Video Accuracy: {v_metrics['overall']['accuracy']:.1%}")
    print(f"  Video Fall Recall: {v_metrics['fall']['recall']:.1%}")
    
    return {
        "strategy": strategy_name,
        "window": w_metrics,
        "video": v_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", 
                        choices=["all", "rule_verify", "confidence", "two_stage", "checklist", "ensemble"])
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    strategies = {
        "rule_verify": ("Rule-Based + LLM Verify", strategy_rule_based_verify),
        "confidence": ("Confidence-Calibrated", strategy_confidence_calibrated),
        "two_stage": ("Two-Stage Classification", strategy_two_stage),
        "checklist": ("Feature Checklist", strategy_checklist),
        "ensemble": ("Ensemble Voting", strategy_ensemble),
    }
    
    results = []
    
    if args.strategy == "all":
        for key, (name, fn) in strategies.items():
            result = evaluate_strategy(client, fn, name, args.split, args.sample)
            results.append(result)
    else:
        name, fn = strategies[args.strategy]
        result = evaluate_strategy(client, fn, name, args.split, args.sample)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - All Strategies")
    print("="*70)
    print(f"{'Strategy':<30} {'W.Acc':>8} {'W.Recall':>10} {'V.Acc':>8} {'V.Recall':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['strategy']:<30} "
              f"{r['window']['overall']['accuracy']:>7.1%} "
              f"{r['window']['fall']['recall']:>9.1%} "
              f"{r['video']['overall']['accuracy']:>7.1%} "
              f"{r['video']['fall']['recall']:>9.1%}")
    
    # Save
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
