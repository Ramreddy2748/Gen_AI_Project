"""
Enhanced XAI Fall Detection
============================

Combines Safety-First prompting (97.9% recall) with explainable reasoning.
Target: 85-90%+ accuracy with full explanations.
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

from zero_shot import load_metadata, compute_metrics, print_metrics, WINDOWS_DIR

RESULTS_DIR = Path(__file__).parent.parent / "results" / "xai_enhanced"


# =============================================================================
# ENHANCED SAFETY-FIRST XAI PROMPT
# =============================================================================

XAI_SAFETY_SYSTEM = """You are a SAFETY-CRITICAL fall detection system with EXPLAINABLE AI capabilities.

Your task: Analyze pose data and provide BOTH a classification AND clear reasoning.

## FALL DETECTION RULES (Safety-Critical - err on side of caution)

STRONG FALL INDICATORS (any ONE = likely fall):
- hip_y > 0.65 (person low/on ground)
- body_angle < 50° (horizontal orientation)  
- velocity > 0.05 (rapid descent)
- posture = "fallen" or "transitioning"
- is_on_ground = true
- is_horizontal = true
- is_rapid_descent = true

NO_FALL requires ALL of:
- Upright posture (angle > 70°)
- Stable position (low velocity)
- No fall flags triggered
- hip_y < 0.5 (standing height)

## OUTPUT FORMAT (must follow exactly):

ANALYSIS:
[2-3 sentences analyzing the key indicators]

INDICATORS:
- [List 2-4 key observations]

CLASSIFICATION: [FALL or NO_FALL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [One sentence explaining the decision]"""


def extract_key_features(window_data: Dict) -> Dict:
    """Extract key features for analysis."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    features = {
        "frames": [],
        "trajectory": analysis.get("trajectory_direction", "unknown"),
        "max_velocity": analysis.get("max_velocity", 0) or 0,
        "has_spike": analysis.get("has_velocity_spike", False),
    }
    
    for key in ["frame_1", "frame_2", "frame_3"]:
        frame = sequence.get(key, {})
        if frame.get("pose_detected", False):
            f = frame.get("features", {})
            e = frame.get("enhanced", {})
            features["frames"].append({
                "hip_y": round(f.get("hip_y", 0), 3),
                "angle": round(f.get("body_angle_degrees", 90), 1),
                "velocity": round(e.get("velocity_hip_y", 0), 4),
                "posture": e.get("posture_state", "unknown"),
                "on_ground": e.get("is_on_ground", False),
                "horizontal": e.get("is_horizontal", False),
                "rapid_descent": e.get("is_rapid_descent", False),
            })
        else:
            features["frames"].append(None)
    
    return features


def create_xai_prompt(features: Dict) -> str:
    """Create detailed prompt for XAI analysis."""
    prompt = "Analyze this pose sequence for fall detection:\n\n"
    
    for i, f in enumerate(features["frames"], 1):
        if f:
            prompt += f"Frame {i}:\n"
            prompt += f"  hip_y={f['hip_y']}, body_angle={f['angle']}°\n"
            prompt += f"  velocity={f['velocity']}, posture={f['posture']}\n"
            flags = []
            if f['on_ground']: flags.append("ON_GROUND")
            if f['horizontal']: flags.append("HORIZONTAL")
            if f['rapid_descent']: flags.append("RAPID_DESCENT")
            if flags:
                prompt += f"  FLAGS: {', '.join(flags)}\n"
        else:
            prompt += f"Frame {i}: Pose not detected\n"
        prompt += "\n"
    
    prompt += f"Window Summary:\n"
    prompt += f"  Trajectory: {features['trajectory']}\n"
    prompt += f"  Max Velocity: {features['max_velocity']:.4f}\n"
    prompt += f"  Velocity Spike: {features['has_spike']}\n"
    
    return prompt


def parse_response(response: str) -> Dict:
    """Parse XAI response."""
    result = {
        "analysis": "",
        "indicators": [],
        "classification": "unknown",
        "confidence": "medium",
        "reason": "",
    }
    
    lines = response.split("\n")
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("ANALYSIS:"):
            current_section = "analysis"
            result["analysis"] = line.replace("ANALYSIS:", "").strip()
        elif line.startswith("INDICATORS:"):
            current_section = "indicators"
        elif line.startswith("CLASSIFICATION:"):
            current_section = None
            cls = line.replace("CLASSIFICATION:", "").strip().upper()
            if "NO_FALL" in cls or "NO FALL" in cls:
                result["classification"] = "no_fall"
            elif "FALL" in cls:
                result["classification"] = "fall"
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().lower()
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif current_section == "analysis" and line:
            result["analysis"] += " " + line
        elif current_section == "indicators" and line.startswith("-"):
            result["indicators"].append(line[1:].strip())
    
    return result


def call_api(client: OpenAI, prompt: str) -> str:
    """Call API with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": XAI_SAFETY_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                print(f"API Error: {e}")
                return "ERROR"
    return "ERROR"


def run_enhanced_xai(splits: List[str] = ["val", "test"], sample_size: int = 100):
    """Run enhanced XAI evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load data
    all_metadata = {}
    for split in splits:
        split_meta = load_metadata(split)
        all_metadata.update(split_meta)
    
    print(f"Total windows: {len(all_metadata)}")
    
    # Balanced sample
    window_ids = list(all_metadata.keys())
    random.seed(42)
    
    fall_ids = [w for w in window_ids if all_metadata[w]["label"] == "fall"]
    nofall_ids = [w for w in window_ids if all_metadata[w]["label"] == "no_fall"]
    
    per_class = sample_size // 2
    sampled = random.sample(fall_ids, min(per_class, len(fall_ids)))
    sampled += random.sample(nofall_ids, min(per_class, len(nofall_ids)))
    random.shuffle(sampled)
    
    print(f"Evaluating {len(sampled)} windows (balanced sample)")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    explanations = []
    
    for i, wid in enumerate(sampled):
        with open(all_metadata[wid]["window_path"]) as f:
            data = json.load(f)
        
        features = extract_key_features(data)
        prompt = create_xai_prompt(features)
        
        response = call_api(client, prompt)
        parsed = parse_response(response)
        
        results.append({
            "window_id": wid,
            "true_label": all_metadata[wid]["label"],
            "predicted": parsed["classification"],
            "confidence": parsed["confidence"],
        })
        
        explanations.append({
            "window_id": wid,
            "true_label": all_metadata[wid]["label"],
            "predicted": parsed["classification"],
            "analysis": parsed["analysis"],
            "indicators": parsed["indicators"],
            "reason": parsed["reason"],
            "confidence": parsed["confidence"],
            "full_response": response,
        })
        
        status = "✓" if parsed["classification"] == all_metadata[wid]["label"] else "✗"
        print(f"[{i+1}/{len(sampled)}] {status} {wid}: {parsed['classification']} ({parsed['confidence']})")
        
        time.sleep(0.15)
    
    # Save results
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(RESULTS_DIR / "explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("ENHANCED XAI RESULTS")
    print("="*60)
    
    true_labels = [r["true_label"] for r in results]
    predictions = [r["predicted"] for r in results]
    metrics = compute_metrics(true_labels, predictions)
    print_metrics(metrics, "Enhanced XAI")
    
    # Confidence distribution
    conf_dist = defaultdict(int)
    for r in results:
        conf_dist[r["confidence"]] += 1
    
    print("\nConfidence Distribution:")
    for conf in ["high", "medium", "low"]:
        count = conf_dist.get(conf, 0)
        print(f"  {conf}: {count} ({count/len(results)*100:.1f}%)")
    
    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "confidence_distribution": dict(conf_dist),
            "sample_size": len(results),
        }, f, indent=2)
    
    # Save sample explanations
    with open(RESULTS_DIR / "sample_explanations.txt", "w") as f:
        for exp in explanations[:15]:
            f.write(f"\n{'='*60}\n")
            f.write(f"Window: {exp['window_id']}\n")
            f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted']}\n")
            f.write(f"Confidence: {exp['confidence']}\n")
            f.write(f"\nAnalysis: {exp['analysis']}\n")
            f.write(f"\nIndicators:\n")
            for ind in exp['indicators']:
                f.write(f"  - {ind}\n")
            f.write(f"\nReason: {exp['reason']}\n")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()
    
    run_enhanced_xai(sample_size=args.sample)
