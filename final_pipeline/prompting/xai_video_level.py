"""
Video-Level XAI Fall Detection
===============================

Analyzes ALL windows from a video and provides a single video-level
classification with comprehensive explanation.

Target: 85-90% accuracy with full explainability.
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "xai_video"


VIDEO_XAI_SYSTEM = """You are a SAFETY-CRITICAL fall detection system analyzing a VIDEO sequence.

You will receive data from multiple time windows of a single video. Your task:
1. Analyze the temporal progression across ALL windows
2. Identify if a FALL occurred anywhere in the video
3. Provide clear explanation of your reasoning

## FALL DETECTION CRITERIA

A video contains a FALL if ANY window shows:
- Rapid descent (high velocity)
- Transition from upright to horizontal
- Person ending up on/near ground (hip_y > 0.65)
- "fallen" posture state
- Multiple fall flags (ON_GROUND, HORIZONTAL, RAPID_DESCENT)

A video is NO_FALL only if ALL windows show:
- Consistent upright posture
- Normal movement patterns
- No fall indicators

## OUTPUT FORMAT:

VIDEO ANALYSIS:
[Describe the overall pattern across all windows]

KEY OBSERVATIONS:
- [List 3-5 most important observations]

CLASSIFICATION: [FALL or NO_FALL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [2-3 sentences explaining your decision based on temporal analysis]"""


def extract_video_features(windows_data: List[Dict]) -> List[Dict]:
    """Extract features from all windows of a video."""
    features = []
    
    for i, window_data in enumerate(windows_data):
        sequence = window_data.get("sequence", {})
        analysis = window_data.get("window_analysis", {})
        
        window_feat = {
            "window_idx": i,
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
                window_feat["frames"].append({
                    "hip_y": round(f.get("hip_y", 0), 2),
                    "angle": round(f.get("body_angle_degrees", 90), 0),
                    "velocity": round(e.get("velocity_hip_y", 0), 3),
                    "posture": e.get("posture_state", "unknown"),
                    "flags": [
                        f for f in ["ON_GROUND", "HORIZONTAL", "RAPID_DESCENT"]
                        if e.get(f"is_{f.lower()}", False)
                    ]
                })
            else:
                window_feat["frames"].append(None)
        
        features.append(window_feat)
    
    return features


def create_video_prompt(video_features: List[Dict], video_id: str) -> str:
    """Create prompt for video-level analysis."""
    prompt = f"Analyze video '{video_id}' with {len(video_features)} time windows:\n\n"
    
    # Summarize each window
    for wf in video_features[:10]:  # Limit to first 10 windows for prompt size
        prompt += f"Window {wf['window_idx']}:\n"
        
        valid_frames = [f for f in wf["frames"] if f is not None]
        if valid_frames:
            avg_hip = sum(f["hip_y"] for f in valid_frames) / len(valid_frames)
            avg_angle = sum(f["angle"] for f in valid_frames) / len(valid_frames)
            max_vel = max(abs(f["velocity"]) for f in valid_frames)
            postures = [f["posture"] for f in valid_frames]
            all_flags = [flag for f in valid_frames for flag in f["flags"]]
            
            prompt += f"  Avg hip_y={avg_hip:.2f}, Avg angle={avg_angle:.0f}°\n"
            prompt += f"  Max velocity={max_vel:.3f}, Trajectory={wf['trajectory']}\n"
            prompt += f"  Postures: {', '.join(postures)}\n"
            if all_flags:
                prompt += f"  FLAGS: {', '.join(all_flags)}\n"
        else:
            prompt += "  No pose detected\n"
        prompt += "\n"
    
    if len(video_features) > 10:
        prompt += f"... and {len(video_features) - 10} more windows\n\n"
    
    # Add summary statistics
    all_valid = []
    for wf in video_features:
        for f in wf["frames"]:
            if f:
                all_valid.append(f)
    
    if all_valid:
        prompt += "VIDEO SUMMARY:\n"
        prompt += f"  Hip range: {min(f['hip_y'] for f in all_valid):.2f} - {max(f['hip_y'] for f in all_valid):.2f}\n"
        prompt += f"  Angle range: {min(f['angle'] for f in all_valid):.0f}° - {max(f['angle'] for f in all_valid):.0f}°\n"
        prompt += f"  Max velocity: {max(abs(f['velocity']) for f in all_valid):.3f}\n"
        
        fall_flags = sum(1 for f in all_valid if f["flags"])
        prompt += f"  Windows with fall flags: {fall_flags}/{len(video_features)}\n"
        
        posture_counts = defaultdict(int)
        for f in all_valid:
            posture_counts[f["posture"]] += 1
        prompt += f"  Posture distribution: {dict(posture_counts)}\n"
    
    return prompt


def parse_video_response(response: str) -> Dict:
    """Parse video-level XAI response."""
    result = {
        "analysis": "",
        "observations": [],
        "classification": "unknown",
        "confidence": "medium",
        "explanation": "",
    }
    
    lines = response.split("\n")
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("VIDEO ANALYSIS:"):
            current_section = "analysis"
            result["analysis"] = line.replace("VIDEO ANALYSIS:", "").strip()
        elif line.startswith("KEY OBSERVATIONS:"):
            current_section = "observations"
        elif line.startswith("CLASSIFICATION:"):
            current_section = None
            cls = line.replace("CLASSIFICATION:", "").strip().upper()
            if "NO_FALL" in cls or "NO FALL" in cls:
                result["classification"] = "no_fall"
            elif "FALL" in cls:
                result["classification"] = "fall"
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().lower()
        elif line.startswith("EXPLANATION:"):
            current_section = "explanation"
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif current_section == "analysis" and line:
            result["analysis"] += " " + line
        elif current_section == "observations" and line.startswith("-"):
            result["observations"].append(line[1:].strip())
        elif current_section == "explanation" and line:
            result["explanation"] += " " + line
    
    return result


def call_api(client: OpenAI, prompt: str) -> str:
    """Call API."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": VIDEO_XAI_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                print(f"Error: {e}")
                return "ERROR"
    return "ERROR"


def run_video_xai(splits: List[str] = ["val", "test"], sample_videos: int = 50):
    """Run video-level XAI evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load and group by video
    all_metadata = {}
    for split in splits:
        split_meta = load_metadata(split)
        all_metadata.update(split_meta)
    
    video_windows = defaultdict(list)
    for wid, meta in all_metadata.items():
        video_windows[meta["video_id"]].append((wid, meta))
    
    print(f"Total videos: {len(video_windows)}")
    
    # Sample videos (balanced)
    video_ids = list(video_windows.keys())
    random.seed(42)
    
    fall_videos = [v for v in video_ids if video_windows[v][0][1]["label"] == "fall"]
    nofall_videos = [v for v in video_ids if video_windows[v][0][1]["label"] == "no_fall"]
    
    per_class = sample_videos // 2
    sampled = random.sample(fall_videos, min(per_class, len(fall_videos)))
    sampled += random.sample(nofall_videos, min(per_class, len(nofall_videos)))
    random.shuffle(sampled)
    
    print(f"Evaluating {len(sampled)} videos")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    explanations = []
    
    for i, video_id in enumerate(sampled):
        windows_info = video_windows[video_id]
        true_label = windows_info[0][1]["label"]
        
        # Load all windows for this video
        windows_data = []
        for wid, meta in sorted(windows_info, key=lambda x: x[0]):
            with open(meta["window_path"]) as f:
                windows_data.append(json.load(f))
        
        # Extract features
        video_features = extract_video_features(windows_data)
        
        # Create prompt and get response
        prompt = create_video_prompt(video_features, video_id)
        response = call_api(client, prompt)
        parsed = parse_video_response(response)
        
        results.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted": parsed["classification"],
            "confidence": parsed["confidence"],
            "num_windows": len(windows_data),
        })
        
        explanations.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted": parsed["classification"],
            "analysis": parsed["analysis"],
            "observations": parsed["observations"],
            "explanation": parsed["explanation"],
            "confidence": parsed["confidence"],
        })
        
        status = "✓" if parsed["classification"] == true_label else "✗"
        print(f"[{i+1}/{len(sampled)}] {status} {video_id}: {parsed['classification']} ({parsed['confidence']})")
        
        time.sleep(0.2)
    
    # Save results
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(RESULTS_DIR / "explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("VIDEO-LEVEL XAI RESULTS")
    print("="*60)
    
    true_labels = [r["true_label"] for r in results]
    predictions = [r["predicted"] for r in results]
    metrics = compute_metrics(true_labels, predictions)
    print_metrics(metrics, "Video XAI")
    
    # Confidence
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
            "num_videos": len(results),
        }, f, indent=2)
    
    # Save explanations
    with open(RESULTS_DIR / "sample_explanations.txt", "w") as f:
        for exp in explanations[:10]:
            f.write(f"\n{'='*60}\n")
            f.write(f"Video: {exp['video_id']}\n")
            f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted']}\n")
            f.write(f"Confidence: {exp['confidence']}\n")
            f.write(f"\nAnalysis:\n{exp['analysis']}\n")
            f.write(f"\nKey Observations:\n")
            for obs in exp['observations']:
                f.write(f"  - {obs}\n")
            f.write(f"\nExplanation:\n{exp['explanation']}\n")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    args = parser.parse_args()
    
    run_video_xai(sample_videos=args.sample)
