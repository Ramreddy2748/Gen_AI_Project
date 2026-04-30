import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


SAMPLED_ROOT = Path("data/pose_pipeline/sampled_frames")
OUTPUT_ROOT = Path("data/pose_pipeline/pose_landmarks")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path("pose_landmarker_lite.task")


def ensure_model() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def build_landmarker() -> mp_vision.PoseLandmarker:
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_from_frame(landmarker: mp_vision.PoseLandmarker, frame_path: Path) -> Dict[str, object]:
    image = cv2.imread(str(frame_path))
    if image is None:
        return {"frame": frame_path.name, "pose_detected": False, "reason": "image_not_read"}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {"frame": frame_path.name, "pose_detected": False}

    landmarks = result.pose_landmarks[0]
    return {
        "frame": frame_path.name,
        "pose_detected": True,
        "landmarks": [
            {
                "index": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
            for idx, landmark in enumerate(landmarks)
        ],
    }


def process_video_dir(landmarker: mp_vision.PoseLandmarker, split_name: str, video_dir: Path) -> None:
    frames = sorted(video_dir.glob("*.png"))
    results: List[Dict[str, object]] = []

    for frame_path in frames:
        results.append(extract_pose_from_frame(landmarker, frame_path))

    split_output_dir = OUTPUT_ROOT / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_output_dir / f"{video_dir.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": video_dir.name,
                "split": split_name,
                "frame_count": len(frames),
                "pose_frame_count": sum(1 for item in results if item.get("pose_detected")),
                "frames": results,
            },
            f,
            indent=2,
        )

    print(f"Saved pose landmarks for {video_dir.name} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks from sampled URFD frames."
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_model()

    selected_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    landmarker = build_landmarker()
    try:
        for split_name in selected_splits:
            split_dir = SAMPLED_ROOT / split_name
            if not split_dir.exists():
                print(f"Skipping missing split directory: {split_dir}")
                continue
            for video_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                process_video_dir(landmarker, split_name, video_dir)
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
