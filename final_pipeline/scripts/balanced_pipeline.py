"""
Balanced Fall Detection Pipeline
=================================

Fixes class imbalance by:
1. Extracting MORE frames from fall videos (lower sample rate)
2. Using smaller stride for fall windows (stride=1)
3. Undersampling no-fall windows to match fall count

This ensures ~1:1 balance for fair GenAI evaluation.
"""

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "balanced"

# Asymmetric settings per class — intentional design to counteract imbalance.
# Fall videos: denser sampling (every 5th frame, stride=1) → more windows per video.
# No-fall videos: sparser sampling (every 15th frame, stride=3) → fewer windows per video.
# Both are then undersampled to achieve 1:1 balance (941 fall : 941 no-fall windows).
# Table 3 in the report reflects these actual asymmetric values.
SETTINGS = {
    "fall": {
        "frame_sample_rate": 5,    # Extract every 5th frame (more frames)
        "max_frames": 60,
        "window_stride": 1,        # Overlap more to get more windows
    },
    "no_fall": {
        "frame_sample_rate": 15,   # Extract every 15th frame (fewer frames)
        "max_frames": 30,
        "window_stride": 3,        # Less overlap
    }
}

WINDOW_SIZE = 3
FRAME_RESIZE = (640, 480)
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Calibrated thresholds
THRESHOLDS = {
    "upright_torso_min": 0.12,
    "fallen_torso_max": 0.06,
    "ground_hip_y_min": 0.65,
    "rapid_velocity_min": 0.15,
    "high_acceleration_min": 0.10,
    "significant_descent": 0.12,
    "horizontal_angle_max": 30.0,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_local_datasets():
    """Load all local datasets including GMNCSA24."""
    videos = []
    
    # Load URFD, Le2i, and GMNCSA24
    for dataset_name in ["URFD", "Le2i", "GMNCSA24"]:
        dataset_path = LOCAL_DATASET_ROOT / dataset_name
        if not dataset_path.exists():
            logger.warning(f"{dataset_name} not found at {dataset_path}")
            continue
        
        for label_folder in ["Fall", "No-Fall"]:
            label_path = dataset_path / label_folder
            if not label_path.exists():
                continue
            
            label = "fall" if label_folder == "Fall" else "no_fall"
            
            for video_file in sorted(label_path.glob("*.mp4")):
                videos.append({
                    "video_id": f"{dataset_name.lower()}_{video_file.stem}",
                    "video_path": str(video_file),
                    "dataset": dataset_name.lower(),
                    "label": label,
                })
        
        count = sum(1 for v in videos if v["dataset"] == dataset_name.lower())
        logger.info(f"Loaded {count} videos from {dataset_name}")
    
    return videos


# =============================================================================
# POSE EXTRACTION
# =============================================================================

_pose_model = None

def get_pose_model():
    global _pose_model
    if _pose_model is not None:
        return _pose_model
    
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    model_path = PROJECT_ROOT / "pose_landmarker_lite.task"
    if not model_path.exists():
        import urllib.request
        logger.info("Downloading pose model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, str(model_path))
    
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
    )
    _pose_model = vision.PoseLandmarker.create_from_options(options)
    return _pose_model


LANDMARKS = {
    "NOSE": 0, "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23, "RIGHT_HIP": 24, "LEFT_KNEE": 25, 
    "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}


def extract_pose(frame_path: str) -> Dict:
    """Extract pose from frame."""
    import mediapipe as mp
    
    image = cv2.imread(frame_path)
    if image is None:
        return {"pose_detected": False, "features": {}}
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    results = get_pose_model().detect(mp_image)
    
    if not results.pose_landmarks:
        return {"pose_detected": False, "features": {}}
    
    lm = results.pose_landmarks[0]
    
    shoulder_x = (lm[11].x + lm[12].x) / 2
    shoulder_y = (lm[11].y + lm[12].y) / 2
    hip_x = (lm[23].x + lm[24].x) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    
    torso_diff = abs(shoulder_y - hip_y)
    body_angle = math.degrees(math.atan2(abs(shoulder_y - hip_y), abs(shoulder_x - hip_x)))
    
    return {
        "pose_detected": True,
        "features": {
            "nose_x": round(lm[0].x, 6), "nose_y": round(lm[0].y, 6),
            "shoulder_x": round(shoulder_x, 6), "shoulder_y": round(shoulder_y, 6),
            "hip_x": round(hip_x, 6), "hip_y": round(hip_y, 6),
            "torso_vertical_diff": round(torso_diff, 6),
            "body_center_y": round((shoulder_y + hip_y) / 2, 6),
            "body_angle_degrees": round(body_angle, 2),
        }
    }


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def process_video(video_info: Dict, output_dir: Path) -> Tuple[Dict, List[Dict]]:
    """Process single video with label-aware settings."""
    video_path = video_info["video_path"]
    video_id = video_info["video_id"]
    label = video_info["label"]
    split = video_info.get("split", "train")
    dataset = video_info["dataset"]
    
    # Get label-specific settings
    settings = SETTINGS[label]
    sample_rate = settings["frame_sample_rate"]
    max_frames = settings["max_frames"]
    window_stride = settings["window_stride"]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Extract frames
    frames_dir = output_dir / "frames" / split / label / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frames = []
    frame_idx = 0
    extracted = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0 and extracted < max_frames:
            frame = cv2.resize(frame, FRAME_RESIZE)
            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Extract pose
            pose_result = extract_pose(str(frame_path))
            
            frames.append({
                "frame_index": frame_idx,
                "frame_name": frame_name,
                "frame_path": str(frame_path),
                "timestamp_seconds": round(frame_idx / fps, 4),
                **pose_result,
            })
            extracted += 1
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) < WINDOW_SIZE:
        return None, []
    
    # Enhance with velocity/acceleration
    enhanced_frames = enhance_frames(frames, fps)
    
    # Create windows with label-specific stride
    windows = create_windows(enhanced_frames, video_id, label, split, dataset, window_stride)
    
    # Save pose data
    pose_dir = output_dir / "poses" / split / label
    pose_dir.mkdir(parents=True, exist_ok=True)
    with open(pose_dir / f"{video_id}.json", "w") as f:
        json.dump({"video_id": video_id, "label": label, "frames": enhanced_frames}, f, indent=2)
    
    # Save windows
    window_dir = output_dir / "windows" / split / label / video_id
    window_dir.mkdir(parents=True, exist_ok=True)
    for w in windows:
        with open(window_dir / f"{w['window_id']}.json", "w") as f:
            json.dump(w, f, indent=2)
    
    summary = {
        "video_id": video_id, "dataset": dataset, "label": label, "split": split,
        "frame_count": len(frames), "pose_count": sum(1 for f in frames if f.get("pose_detected")),
        "window_count": len(windows),
    }
    
    return summary, windows


def enhance_frames(frames: List[Dict], fps: float) -> List[Dict]:
    """Add velocity, acceleration, posture."""
    enhanced = []
    prev_frame = None
    prev_vel = 0.0
    
    for frame in frames:
        ef = dict(frame)
        features = frame.get("features", {})
        
        enhanced_f = {
            "velocity_hip_y": 0.0, "velocity_shoulder_y": 0.0, "velocity_magnitude": 0.0,
            "acceleration_hip_y": 0.0, "acceleration_magnitude": 0.0, "jerk": 0.0,
            "posture_state": "unknown",
            "is_rapid_descent": False, "is_on_ground": False, 
            "is_horizontal": False, "is_high_acceleration": False,
        }
        
        if frame.get("pose_detected") and features:
            hip_y = features.get("hip_y")
            torso_diff = features.get("torso_vertical_diff")
            body_angle = features.get("body_angle_degrees", 90)
            
            # Posture
            on_ground = hip_y > THRESHOLDS["ground_hip_y_min"] if hip_y else False
            is_horizontal = body_angle < THRESHOLDS["horizontal_angle_max"] if body_angle else False
            is_upright = torso_diff > THRESHOLDS["upright_torso_min"] and body_angle > 60 if torso_diff else False
            
            if on_ground and is_horizontal:
                enhanced_f["posture_state"] = "fallen"
            elif is_upright:
                enhanced_f["posture_state"] = "upright"
            else:
                enhanced_f["posture_state"] = "transitioning"
            
            enhanced_f["is_on_ground"] = on_ground
            enhanced_f["is_horizontal"] = is_horizontal
            
            # Velocity
            if prev_frame and prev_frame.get("pose_detected"):
                prev_f = prev_frame.get("features", {})
                frame_gap = frame.get("frame_index", 0) - prev_frame.get("frame_index", 0)
                dt = max(frame_gap / fps, 0.001)
                
                if prev_f.get("hip_y") and hip_y:
                    vel = (hip_y - prev_f["hip_y"]) / dt
                    vel_s = (features.get("shoulder_y", 0) - prev_f.get("shoulder_y", 0)) / dt
                    
                    enhanced_f["velocity_hip_y"] = round(vel, 6)
                    enhanced_f["velocity_shoulder_y"] = round(vel_s, 6)
                    enhanced_f["velocity_magnitude"] = round(math.sqrt(vel**2 + vel_s**2), 6)
                    
                    accel = (vel - prev_vel) / dt
                    enhanced_f["acceleration_hip_y"] = round(accel, 6)
                    enhanced_f["acceleration_magnitude"] = round(abs(accel), 6)
                    
                    enhanced_f["is_rapid_descent"] = vel > THRESHOLDS["rapid_velocity_min"]
                    enhanced_f["is_high_acceleration"] = abs(accel) > THRESHOLDS["high_acceleration_min"]
                    
                    prev_vel = vel
            
            prev_frame = frame
        
        ef["enhanced"] = enhanced_f
        enhanced.append(ef)
    
    return enhanced


def create_windows(frames: List[Dict], video_id: str, label: str, 
                   split: str, dataset: str, stride: int) -> List[Dict]:
    """Create windows with given stride."""
    if len(frames) < WINDOW_SIZE:
        return []
    
    windows = []
    
    for i, start in enumerate(range(0, len(frames) - WINDOW_SIZE + 1, stride)):
        window_frames = frames[start:start + WINDOW_SIZE]
        
        sequence = {f"frame_{j+1}": f for j, f in enumerate(window_frames)}
        
        frames_with_pose = [f for f in window_frames if f.get("pose_detected")]
        
        analysis = {
            "pose_detection_count": len(frames_with_pose),
            "pose_detection_rate": round(len(frames_with_pose) / len(window_frames), 4),
        }
        
        if len(frames_with_pose) >= 2:
            first, last = frames_with_pose[0], frames_with_pose[-1]
            first_hip = first.get("features", {}).get("hip_y", 0)
            last_hip = last.get("features", {}).get("hip_y", 0)
            hip_delta = (last_hip or 0) - (first_hip or 0)
            
            analysis["hip_delta"] = round(hip_delta, 6)
            analysis["max_velocity"] = round(max(f.get("enhanced", {}).get("velocity_magnitude", 0) for f in frames_with_pose), 6)
            analysis["trajectory_direction"] = "descending" if hip_delta > 0.05 else "stable"
            analysis["has_velocity_spike"] = any(f.get("enhanced", {}).get("is_rapid_descent") for f in frames_with_pose)
            analysis["start_posture"] = first.get("enhanced", {}).get("posture_state", "unknown")
            analysis["end_posture"] = last.get("enhanced", {}).get("posture_state", "unknown")
            
            # Fall likelihood
            score = 0.0
            if analysis["has_velocity_spike"]: score += 0.25
            if hip_delta > THRESHOLDS["significant_descent"]: score += 0.20
            if analysis["end_posture"] == "fallen": score += 0.25
            if analysis["start_posture"] == "upright" and analysis["end_posture"] != "upright": score += 0.15
            if last.get("enhanced", {}).get("is_horizontal"): score += 0.15
            
            analysis["fall_likelihood_score"] = round(min(1.0, score), 3)
        else:
            analysis["fall_likelihood_score"] = 0.0
        
        # LLM descriptions
        llm_desc = {
            "frame_descriptions": [
                f"Frame {j+1}: {f.get('enhanced', {}).get('posture_state', 'unknown')}, vel={f.get('enhanced', {}).get('velocity_hip_y', 0):.3f}"
                for j, f in enumerate(window_frames) if f.get("pose_detected")
            ],
            "fall_assessment": f"Score: {analysis.get('fall_likelihood_score', 0):.0%}",
        }
        
        windows.append({
            "window_id": f"{video_id}_w{i:04d}",
            "video_id": video_id, "dataset": dataset,
            "label": label, "split": split,
            "window_index": i,
            "sequence": sequence,
            "window_analysis": analysis,
            "llm_descriptions": llm_desc,
        })
    
    return windows


# =============================================================================
# BALANCING
# =============================================================================

def balance_windows(all_windows: List[Dict], target_ratio: float = 1.0) -> List[Dict]:
    """Balance fall vs no-fall windows to 1:1 ratio."""
    fall_windows = [w for w in all_windows if w["label"] == "fall"]
    nofall_windows = [w for w in all_windows if w["label"] == "no_fall"]
    
    logger.info(f"Before balancing: Fall={len(fall_windows)}, No-Fall={len(nofall_windows)}")
    
    random.seed(42)
    
    # Balance to smaller class size for 1:1
    min_count = min(len(fall_windows), len(nofall_windows))
    target_count = int(min_count * target_ratio)
    
    if len(fall_windows) > target_count:
        fall_windows = random.sample(fall_windows, target_count)
    if len(nofall_windows) > target_count:
        nofall_windows = random.sample(nofall_windows, target_count)
    
    balanced = fall_windows + nofall_windows
    random.shuffle(balanced)
    
    logger.info(f"After balancing: Fall={len(fall_windows)}, No-Fall={len(nofall_windows)}")
    
    return balanced


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance-ratio", type=float, default=1.0, help="No-Fall:Fall ratio (1.0 = equal)")
    args = parser.parse_args()
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests_dir = OUTPUT_ROOT / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    # Load videos
    videos = load_local_datasets()
    logger.info(f"Loaded {len(videos)} videos")
    
    # Assign splits
    random.seed(42)
    random.shuffle(videos)
    
    fall_vids = [v for v in videos if v["label"] == "fall"]
    nofall_vids = [v for v in videos if v["label"] == "no_fall"]
    
    def assign_splits(vids):
        n = len(vids)
        for i, v in enumerate(vids):
            if i < int(n * 0.7):
                v["split"] = "train"
            elif i < int(n * 0.85):
                v["split"] = "val"
            else:
                v["split"] = "test"
        return vids
    
    videos = assign_splits(fall_vids) + assign_splits(nofall_vids)
    
    # Process
    all_summaries = []
    all_windows = []
    
    for i, video in enumerate(videos):
        try:
            summary, windows = process_video(video, OUTPUT_ROOT)
            if summary:
                all_summaries.append(summary)
                all_windows.extend(windows)
            
            if (i + 1) % 30 == 0:
                logger.info(f"Processed {i + 1}/{len(videos)} videos ({len(all_windows)} windows)")
        except Exception as e:
            logger.error(f"Error: {video['video_id']}: {e}")
    
    # Balance windows
    balanced_windows = balance_windows(all_windows, args.balance_ratio)
    
    # Save balanced window manifest
    balanced_dir = OUTPUT_ROOT / "windows_balanced"
    balanced_dir.mkdir(exist_ok=True)
    
    balanced_manifest = []
    for w in balanced_windows:
        # Copy to balanced directory
        src_dir = OUTPUT_ROOT / "windows" / w["split"] / w["label"] / w["video_id"]
        dst_dir = balanced_dir / w["split"] / w["label"] / w["video_id"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        src_file = src_dir / f"{w['window_id']}.json"
        dst_file = dst_dir / f"{w['window_id']}.json"
        
        if src_file.exists():
            import shutil
            shutil.copy(src_file, dst_file)
        
        balanced_manifest.append({
            "window_id": w["window_id"],
            "video_id": w["video_id"],
            "label": w["label"],
            "split": w["split"],
            "fall_likelihood_score": w["window_analysis"].get("fall_likelihood_score", 0),
        })
    
    # Save manifests
    with open(manifests_dir / "video_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "dataset", "label", "split", "frame_count", "pose_count", "window_count"])
        writer.writeheader()
        writer.writerows(all_summaries)
    
    with open(manifests_dir / "balanced_window_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "video_id", "label", "split", "fall_likelihood_score"])
        writer.writeheader()
        writer.writerows(balanced_manifest)
    
    # Stats
    fall_w = [w for w in balanced_windows if w["label"] == "fall"]
    nofall_w = [w for w in balanced_windows if w["label"] == "no_fall"]
    
    fall_scores = [w["window_analysis"].get("fall_likelihood_score", 0) for w in fall_w]
    nofall_scores = [w["window_analysis"].get("fall_likelihood_score", 0) for w in nofall_w]
    
    print("\n" + "=" * 70)
    print("BALANCED PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nVideos processed: {len(all_summaries)}")
    print(f"Total windows (before balance): {len(all_windows)}")
    print(f"Total windows (after balance): {len(balanced_windows)}")
    print(f"\n  Fall: {len(fall_w)}")
    print(f"  No-Fall: {len(nofall_w)}")
    print(f"  Ratio: 1:{len(nofall_w)/len(fall_w):.1f}")
    
    if fall_scores:
        print(f"\nFall Likelihood Scores:")
        print(f"  Fall avg: {sum(fall_scores)/len(fall_scores):.3f}")
    if nofall_scores:
        print(f"  No-Fall avg: {sum(nofall_scores)/len(nofall_scores):.3f}")
    
    print(f"\nOutput: {OUTPUT_ROOT}")
    print(f"Balanced windows: {balanced_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
