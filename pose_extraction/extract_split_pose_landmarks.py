import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


DEFAULT_FRAMES_ROOT = Path("data/processed/frames")
DEFAULT_OUTPUT_ROOT = Path("data/processed/poses")
DEFAULT_MODEL_PATH = Path("pose_landmarker_lite.task")
SPLIT_DIR_NAME = {"train": "train", "val": "valid", "test": "test"}


def build_pose_estimator() -> mp_vision.PoseLandmarker:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe pose model not found: {DEFAULT_MODEL_PATH}. "
            "Run a setup step that downloads pose_landmarker_lite.task first."
        )

    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(DEFAULT_MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_from_frame(
    pose_estimator: mp_vision.PoseLandmarker,
    frame_path: Path,
) -> Dict[str, object]:
    image = cv2.imread(str(frame_path))
    if image is None:
        return {"frame": frame_path.name, "pose_detected": False, "reason": "image_not_read"}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = pose_estimator.detect(mp_image)

    if not result.pose_landmarks:
        return {"frame": frame_path.name, "pose_detected": False}

    landmarks = result.pose_landmarks[0]
    features = {
        "hip_x": float(landmarks[23].x),
        "hip_y": float(landmarks[23].y),
        "shoulder_x": float(landmarks[11].x),
        "shoulder_y": float(landmarks[11].y),
        "nose_x": float(landmarks[0].x),
        "nose_y": float(landmarks[0].y),
        "torso_vertical_diff": float(abs(landmarks[11].y - landmarks[23].y)),
    }
    return {
        "frame": frame_path.name,
        "pose_detected": True,
        "features": features,
    }


def process_video_dir(
    pose_estimator: mp_vision.PoseLandmarker,
    split_name: str,
    label_name: str,
    video_dir: Path,
    output_root: Path,
) -> Dict[str, str]:
    frame_paths = sorted(video_dir.glob("*.jpg"))
    results: List[Dict[str, object]] = []

    for frame_path in frame_paths:
        results.append(extract_pose_from_frame(pose_estimator, frame_path))

    split_output_dir = output_root / SPLIT_DIR_NAME[split_name] / label_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_output_dir / f"{video_dir.name}.json"
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "video_id": video_dir.name,
                "split": split_name,
                "label": label_name,
                "frame_count": len(frame_paths),
                "pose_frame_count": sum(1 for item in results if item.get("pose_detected")),
                "frames": results,
            },
            file_obj,
            indent=2,
        )

    print(f"Saved pose landmarks for {video_dir.name} -> {output_path}")
    return {
        "video_id": video_dir.name,
        "split": split_name,
        "label": label_name,
        "frame_dir": str(video_dir),
        "pose_json": str(output_path),
        "frame_count": str(len(frame_paths)),
        "pose_frame_count": str(sum(1 for item in results if item.get("pose_detected"))),
    }


def write_summary(output_root: Path, split_name: str, rows: List[Dict[str, str]]) -> Path:
    summary_path = output_root / f"{SPLIT_DIR_NAME[split_name]}_pose_manifest.csv"
    fieldnames = [
        "video_id",
        "split",
        "label",
        "frame_dir",
        "pose_json",
        "frame_count",
        "pose_frame_count",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from sampled frame splits."
    )
    parser.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    split_dir = frames_root / SPLIT_DIR_NAME[args.split]

    if not split_dir.exists():
        raise FileNotFoundError(f"Frame split directory not found: {split_dir}")

    pose_estimator = build_pose_estimator()
    summary_rows: List[Dict[str, str]] = []

    try:
        for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for video_dir in sorted(path for path in label_dir.iterdir() if path.is_dir()):
                summary_rows.append(
                    process_video_dir(
                        pose_estimator=pose_estimator,
                        split_name=args.split,
                        label_name=label_dir.name,
                        video_dir=video_dir,
                        output_root=output_root,
                    )
                )
    finally:
        pose_estimator.close()

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = write_summary(output_root, args.split, summary_rows)
    print(f"Saved pose manifest to: {summary_path}")


if __name__ == "__main__":
    main()
