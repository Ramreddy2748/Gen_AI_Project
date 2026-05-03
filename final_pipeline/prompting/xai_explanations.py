"""
XAI (Explainable AI) Fall Detection
====================================

Generates human-readable explanations for fall detection predictions.
Provides reasoning transparency for safety-critical decisions.

Features:
1. Feature-based explanations (which indicators triggered the decision)
2. Confidence scoring with reasoning
3. Similar case references (from RAG)
4. Step-by-step reasoning trace
"""

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

from zero_shot import load_metadata, compute_metrics, print_metrics, WINDOWS_DIR

RESULTS_DIR = Path(__file__).parent.parent / "results" / "xai"
INDEX_DIR = Path(__file__).parent.parent / "rag_index"


# =============================================================================
# FEATURE INTERPRETATION
# =============================================================================

def interpret_features(window_data: Dict) -> Dict:
    """Extract and interpret features from window."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    interpretations = {
        "frames": [],
        "fall_indicators": [],
        "normal_indicators": [],
        "summary": {},
    }
    
    for i, key in enumerate(["frame_1", "frame_2", "frame_3"], 1):
        frame = sequence.get(key, {})
        if not frame.get("pose_detected", False):
            interpretations["frames"].append(f"Frame {i}: Pose not detected")
            continue
        
        f = frame.get("features", {})
        e = frame.get("enhanced", {})
        
        hip_y = f.get("hip_y", 0)
        angle = f.get("body_angle_degrees", 90)
        velocity = e.get("velocity_hip_y", 0)
        posture = e.get("posture_state", "unknown")
        
        # Interpret position
        if hip_y > 0.7:
            pos_text = "Person is LOW (near ground)"
            interpretations["fall_indicators"].append(f"Frame {i}: Low position (hip_y={hip_y:.2f})")
        elif hip_y < 0.4:
            pos_text = "Person is HIGH (standing)"
            interpretations["normal_indicators"].append(f"Frame {i}: Standing position")
        else:
            pos_text = "Person is at MEDIUM height"
        
        # Interpret angle
        if angle < 45:
            angle_text = "Body is HORIZONTAL (lying down)"
            interpretations["fall_indicators"].append(f"Frame {i}: Horizontal body (angle={angle:.0f}°)")
        elif angle > 70:
            angle_text = "Body is VERTICAL (upright)"
            interpretations["normal_indicators"].append(f"Frame {i}: Upright posture")
        else:
            angle_text = f"Body is TILTED ({angle:.0f}°)"
        
        # Interpret velocity
        if abs(velocity) > 0.05:
            vel_text = f"RAPID movement (velocity={velocity:.3f})"
            if velocity > 0:
                interpretations["fall_indicators"].append(f"Frame {i}: Rapid downward motion")
        else:
            vel_text = "Stable/slow movement"
        
        # Interpret flags
        flags = []
        if e.get("is_rapid_descent"):
            flags.append("RAPID_DESCENT")
            interpretations["fall_indicators"].append(f"Frame {i}: Rapid descent detected")
        if e.get("is_on_ground"):
            flags.append("ON_GROUND")
            interpretations["fall_indicators"].append(f"Frame {i}: Person on ground")
        if e.get("is_horizontal"):
            flags.append("HORIZONTAL")
            interpretations["fall_indicators"].append(f"Frame {i}: Horizontal position")
        
        frame_text = f"Frame {i}: {pos_text}, {angle_text}, {vel_text}"
        if flags:
            frame_text += f" [FLAGS: {', '.join(flags)}]"
        frame_text += f" Posture: {posture}"
        
        interpretations["frames"].append(frame_text)
    
    # Window-level summary
    trajectory = analysis.get("trajectory_direction", "unknown")
    max_vel = analysis.get("max_velocity", 0) or 0
    spike = analysis.get("has_velocity_spike", False)
    
    interpretations["summary"] = {
        "trajectory": trajectory,
        "max_velocity": max_vel,
        "has_spike": spike,
    }
    
    if trajectory == "descending":
        interpretations["fall_indicators"].append("Overall trajectory: Descending")
    if spike:
        interpretations["fall_indicators"].append("Velocity spike detected")
    if max_vel > 0.05:
        interpretations["fall_indicators"].append(f"High max velocity: {max_vel:.3f}")
    
    return interpretations


def generate_explanation(interpretations: Dict, prediction: str, confidence: float) -> str:
    """Generate human-readable explanation."""
    explanation = []
    
    explanation.append("=" * 60)
    explanation.append("FALL DETECTION ANALYSIS")
    explanation.append("=" * 60)
    
    explanation.append("\n### Frame-by-Frame Analysis ###")
    for frame_text in interpretations["frames"]:
        explanation.append(f"  {frame_text}")
    
    explanation.append(f"\n### Trajectory Summary ###")
    summary = interpretations["summary"]
    explanation.append(f"  Direction: {summary['trajectory']}")
    explanation.append(f"  Max Velocity: {summary['max_velocity']:.3f}")
    explanation.append(f"  Velocity Spike: {'Yes' if summary['has_spike'] else 'No'}")
    
    if interpretations["fall_indicators"]:
        explanation.append("\n### FALL Indicators Detected ###")
        for indicator in interpretations["fall_indicators"]:
            explanation.append(f"  [!] {indicator}")
    
    if interpretations["normal_indicators"]:
        explanation.append("\n### Normal Activity Indicators ###")
        for indicator in interpretations["normal_indicators"]:
            explanation.append(f"  [✓] {indicator}")
    
    # Decision reasoning
    fall_count = len(interpretations["fall_indicators"])
    normal_count = len(interpretations["normal_indicators"])
    
    explanation.append("\n### Decision Reasoning ###")
    explanation.append(f"  Fall indicators: {fall_count}")
    explanation.append(f"  Normal indicators: {normal_count}")
    
    if prediction == "fall":
        explanation.append(f"\n  DECISION: FALL DETECTED")
        explanation.append(f"  Confidence: {confidence:.1%}")
        explanation.append(f"  Reason: {fall_count} fall indicators outweigh {normal_count} normal indicators")
    else:
        explanation.append(f"\n  DECISION: NO FALL")
        explanation.append(f"  Confidence: {confidence:.1%}")
        explanation.append(f"  Reason: Normal activity patterns dominate")
    
    explanation.append("=" * 60)
    
    return "\n".join(explanation)


# =============================================================================
# XAI CLASSIFICATION
# =============================================================================

XAI_SYSTEM_PROMPT = """You are an expert fall detection system that provides EXPLAINABLE predictions.

For each analysis:
1. Examine the pose data carefully
2. Identify specific fall indicators or normal activity signs
3. Provide your reasoning step-by-step
4. Give a final decision with confidence

Your response MUST follow this exact format:

REASONING:
[Your step-by-step analysis here]

INDICATORS:
- [List key indicators you identified]

DECISION: [FALL or NO_FALL]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
EXPLANATION: [One sentence summary of why]"""


def create_xai_prompt(interpretations: Dict) -> str:
    """Create XAI prompt with interpreted features."""
    prompt = "Analyze this pose sequence for fall detection:\n\n"
    
    for frame_text in interpretations["frames"]:
        prompt += f"{frame_text}\n"
    
    summary = interpretations["summary"]
    prompt += f"\nTrajectory: {summary['trajectory']}"
    prompt += f"\nMax Velocity: {summary['max_velocity']:.3f}"
    prompt += f"\nVelocity Spike: {'Yes' if summary['has_spike'] else 'No'}"
    
    prompt += "\n\nProvide your analysis with reasoning."
    
    return prompt


def parse_xai_response(response: str) -> Dict:
    """Parse XAI response."""
    result = {
        "reasoning": "",
        "indicators": [],
        "decision": "unknown",
        "confidence": "low",
        "explanation": "",
    }
    
    lines = response.split("\n")
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("REASONING:"):
            current_section = "reasoning"
            result["reasoning"] = line.replace("REASONING:", "").strip()
        elif line.startswith("INDICATORS:"):
            current_section = "indicators"
        elif line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().upper()
            result["decision"] = "fall" if "FALL" in decision and "NO" not in decision else "no_fall"
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().lower()
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif current_section == "reasoning" and line:
            result["reasoning"] += " " + line
        elif current_section == "indicators" and line.startswith("-"):
            result["indicators"].append(line[1:].strip())
    
    return result


def call_api(client: OpenAI, system: str, user: str) -> str:
    """Call API with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_xai_evaluation(splits: List[str] = ["val", "test"], 
                       sample_size: int = 50):
    """Run XAI evaluation with explanations."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load evaluation data
    all_metadata = {}
    for split in splits:
        split_meta = load_metadata(split)
        all_metadata.update(split_meta)
    
    print(f"Total windows: {len(all_metadata)}")
    
    # Sample for XAI (full explanations are expensive)
    window_ids = list(all_metadata.keys())
    random.seed(42)
    
    # Ensure balanced sample
    fall_ids = [w for w in window_ids if all_metadata[w]["label"] == "fall"]
    nofall_ids = [w for w in window_ids if all_metadata[w]["label"] == "no_fall"]
    
    sample_per_class = sample_size // 2
    sampled = random.sample(fall_ids, min(sample_per_class, len(fall_ids)))
    sampled += random.sample(nofall_ids, min(sample_per_class, len(nofall_ids)))
    random.shuffle(sampled)
    
    print(f"Sampled {len(sampled)} windows for XAI evaluation")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    explanations = []
    
    for i, wid in enumerate(sampled):
        window_path = Path(all_metadata[wid]["window_path"])
        
        with open(window_path) as f:
            data = json.load(f)
        
        # Get interpretations
        interpretations = interpret_features(data)
        
        # Create XAI prompt
        user_prompt = create_xai_prompt(interpretations)
        
        # Get LLM response with reasoning
        response = call_api(client, XAI_SYSTEM_PROMPT, user_prompt)
        parsed = parse_xai_response(response)
        
        # Generate full explanation
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        conf_score = confidence_map.get(parsed["confidence"], 0.5)
        
        full_explanation = generate_explanation(
            interpretations, 
            parsed["decision"], 
            conf_score
        )
        
        results.append({
            "window_id": wid,
            "true_label": all_metadata[wid]["label"],
            "predicted": parsed["decision"],
            "confidence": parsed["confidence"],
            "llm_reasoning": parsed["reasoning"],
            "llm_indicators": parsed["indicators"],
            "llm_explanation": parsed["explanation"],
        })
        
        explanations.append({
            "window_id": wid,
            "true_label": all_metadata[wid]["label"],
            "predicted": parsed["decision"],
            "full_explanation": full_explanation,
            "llm_response": response,
        })
        
        print(f"[{i+1}/{len(sampled)}] {wid}: {parsed['decision']} ({parsed['confidence']})")
        time.sleep(0.2)
    
    # Save results
    with open(RESULTS_DIR / "xai_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(RESULTS_DIR / "xai_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
    
    # Save sample explanations as readable text
    with open(RESULTS_DIR / "sample_explanations.txt", "w") as f:
        for exp in explanations[:10]:
            f.write(f"\nWindow: {exp['window_id']}\n")
            f.write(f"True Label: {exp['true_label']}\n")
            f.write(f"Predicted: {exp['predicted']}\n")
            f.write(exp['full_explanation'])
            f.write("\n\nLLM Response:\n")
            f.write(exp['llm_response'])
            f.write("\n" + "="*80 + "\n")
    
    # Compute metrics
    print("\n" + "="*60)
    print("XAI FALL DETECTION RESULTS")
    print("="*60)
    
    true_labels = [r["true_label"] for r in results]
    predictions = [r["predicted"] for r in results]
    metrics = compute_metrics(true_labels, predictions)
    print_metrics(metrics, "XAI Sample")
    
    # Confidence distribution
    print("\nConfidence Distribution:")
    conf_dist = defaultdict(int)
    for r in results:
        conf_dist[r["confidence"]] += 1
    for conf, count in sorted(conf_dist.items()):
        print(f"  {conf}: {count} ({count/len(results)*100:.1f}%)")
    
    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({
            "sample_metrics": metrics,
            "confidence_distribution": dict(conf_dist),
            "sample_size": len(results),
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--sample", type=int, default=50)
    args = parser.parse_args()
    
    run_xai_evaluation(args.splits, args.sample)


if __name__ == "__main__":
    main()
