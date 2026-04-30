import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


DEFAULT_OUTPUT_ROOT = Path("data/upload_windows")
DEFAULT_MODEL_PATH = Path("pose_landmarker_lite.task")


def safe_video_id(video_path: Path, explicit_video_id: str) -> str:
    if explicit_video_id:
        return explicit_video_id
    return video_path.stem.replace(" ", "_").replace("-", "_")


def build_pose_estimator() -> mp_vision.PoseLandmarker:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe pose model not found: {DEFAULT_MODEL_PATH}"
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


def video_metadata(capture: cv2.VideoCapture) -> Dict[str, float]:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }


def timestamp_seconds(frame_index: int, fps: float) -> float:
    return round(frame_index / fps, 4) if fps > 0 else 0.0


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
) -> Dict[str, object]:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in output_dir.glob("*.jpg"):
        frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    metadata = video_metadata(capture)
    source_fps = float(metadata["fps"])
    interval_seconds = 1.0 / sample_fps
    next_sample_at_seconds = 0.0
    frame_index = 0
    sampled_frames: List[Dict[str, object]] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_time_raw = (frame_index / source_fps) if source_fps > 0 else 0.0
            if current_time_raw + 1e-9 >= next_sample_at_seconds:
                current_time = round(current_time_raw, 4)
                frame_name = f"frame_{frame_index:06d}.jpg"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                sampled_frames.append(
                    {
                        "frame": frame_name,
                        "frame_index": frame_index,
                        "timestamp_seconds": current_time,
                        "frame_path": str(frame_path),
                    }
                )
                next_sample_at_seconds += interval_seconds

            frame_index += 1
    finally:
        capture.release()

    metadata["sample_fps"] = sample_fps
    metadata["sampling_interval_seconds"] = interval_seconds
    metadata["sampled_frame_count"] = len(sampled_frames)
    metadata["frames"] = sampled_frames
    return metadata


def extract_pose_from_image(
    pose_estimator: mp_vision.PoseLandmarker,
    frame_path: Path,
) -> Dict[str, object]:
    image = cv2.imread(str(frame_path))
    if image is None:
        return {"pose_detected": False, "reason": "image_not_read", "features": {}}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = pose_estimator.detect(mp_image)
    if not result.pose_landmarks:
        return {"pose_detected": False, "features": {}}

    landmarks = result.pose_landmarks[0]
    return {
        "pose_detected": True,
        "features": {
            "hip_x": float(landmarks[23].x),
            "hip_y": float(landmarks[23].y),
            "shoulder_x": float(landmarks[11].x),
            "shoulder_y": float(landmarks[11].y),
            "nose_x": float(landmarks[0].x),
            "nose_y": float(landmarks[0].y),
            "torso_vertical_diff": float(abs(landmarks[11].y - landmarks[23].y)),
        },
    }


def extract_pose_frames(
    sampled_frames: List[Dict[str, object]],
    pose_json_path: Path,
) -> List[Dict[str, object]]:
    pose_estimator = build_pose_estimator()
    pose_frames: List[Dict[str, object]] = []
    try:
        for frame in sampled_frames:
            pose = extract_pose_from_image(pose_estimator, Path(str(frame["frame_path"])))
            pose_frames.append(
                {
                    "frame": frame["frame"],
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "frame_path": frame["frame_path"],
                    **pose,
                }
            )
    finally:
        pose_estimator.close()

    pose_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pose_json_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "frame_count": len(sampled_frames),
                "pose_frame_count": sum(1 for frame in pose_frames if frame["pose_detected"]),
                "frames": pose_frames,
            },
            file_obj,
            indent=2,
        )
    return pose_frames


def sliding_windows(
    frames: List[Dict[str, object]],
    window_size: int,
    stride: int,
) -> Iterable[Tuple[int, List[Dict[str, object]]]]:
    if window_size <= 0:
        raise ValueError("--window-size must be positive")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if len(frames) < window_size:
        return

    for start in range(0, len(frames) - window_size + 1, stride):
        yield start, frames[start : start + window_size]


def build_window_record(
    video_id: str,
    video_path: Path,
    duration_seconds: float,
    window_index: int,
    start_position: int,
    window_frames: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "window_id": f"{video_id}_w{window_index:04d}",
        "video_id": video_id,
        "source_video_path": str(video_path),
        "video_start_time_seconds": 0.0,
        "video_end_time_seconds": round(duration_seconds, 4),
        "window_start_time_seconds": window_frames[0]["timestamp_seconds"],
        "window_end_time_seconds": window_frames[-1]["timestamp_seconds"],
        "window_index": window_index,
        "window_start_position": start_position,
        "window_size": len(window_frames),
        "frames": [
            {
                "position": position,
                "frame": frame["frame"],
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_path": frame["frame_path"],
                "pose_detected": frame["pose_detected"],
                "features": frame["features"],
            }
            for position, frame in enumerate(window_frames)
        ],
    }


def csv_row(record: Dict[str, object]) -> Dict[str, object]:
    frames = record["frames"]
    if not isinstance(frames, list):
        frames = []
    return {
        "window_id": record["window_id"],
        "video_id": record["video_id"],
        "source_video_path": record["source_video_path"],
        "video_start_time_seconds": record["video_start_time_seconds"],
        "video_end_time_seconds": record["video_end_time_seconds"],
        "window_start_time_seconds": record["window_start_time_seconds"],
        "window_end_time_seconds": record["window_end_time_seconds"],
        "window_index": record["window_index"],
        "window_start_position": record["window_start_position"],
        "window_size": record["window_size"],
        "frame_names": "|".join(str(frame["frame"]) for frame in frames),
        "frame_indices": "|".join(str(frame["frame_index"]) for frame in frames),
        "frame_timestamps_seconds": "|".join(str(frame["timestamp_seconds"]) for frame in frames),
        "frame_paths": "|".join(str(frame["frame_path"]) for frame in frames),
        "pose_detected_count": sum(1 for frame in frames if frame.get("pose_detected")),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "window_id",
        "video_id",
        "source_video_path",
        "video_start_time_seconds",
        "video_end_time_seconds",
        "window_start_time_seconds",
        "window_end_time_seconds",
        "window_index",
        "window_start_position",
        "window_size",
        "frame_names",
        "frame_indices",
        "frame_timestamps_seconds",
        "frame_paths",
        "pose_detected_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")


def process_video(args: argparse.Namespace) -> Dict[str, object]:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_id = safe_video_id(video_path, args.video_id)
    output_dir = Path(args.output_root) / video_id
    frames_dir = output_dir / "frames"
    pose_json_path = output_dir / "pose_features.json"

    metadata = sample_video_frames(video_path, frames_dir, args.sample_fps)
    pose_frames = extract_pose_frames(
        sampled_frames=metadata["frames"],
        pose_json_path=pose_json_path,
    )

    records: List[Dict[str, object]] = []
    for window_index, (start, window_frames) in enumerate(
        sliding_windows(pose_frames, args.window_size, args.stride)
    ):
        records.append(
            build_window_record(
                video_id=video_id,
                video_path=video_path,
                duration_seconds=float(metadata["duration_seconds"]),
                window_index=window_index,
                start_position=start,
                window_frames=window_frames,
            )
        )

    csv_path = output_dir / "window_manifest.csv"
    jsonl_path = output_dir / "llm_windows.jsonl"
    summary_path = output_dir / "window_summary.json"
    write_csv(csv_path, [csv_row(record) for record in records])
    write_jsonl(jsonl_path, records)

    summary = {
        "video_id": video_id,
        "source_video_path": str(video_path),
        "output_dir": str(output_dir),
        "fps": round(float(metadata["fps"]), 4),
        "frame_count": int(metadata["frame_count"]),
        "duration_seconds": round(float(metadata["duration_seconds"]), 4),
        "sample_fps": args.sample_fps,
        "sampled_frame_count": int(metadata["sampled_frame_count"]),
        "pose_frame_count": sum(1 for frame in pose_frames if frame["pose_detected"]),
        "window_size": args.window_size,
        "stride": args.stride,
        "window_count": len(records),
        "pose_json": str(pose_json_path),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one uploaded video into timed, pre-LLM 3-frame windows."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = process_video(parse_args())
    print("Processed uploaded video through pre-LLM windows.")
    print(f"Video: {summary['source_video_path']}")
    print(f"Windows: {summary['window_count']}")
    print(f"CSV: {summary['csv_path']}")
    print(f"JSONL: {summary['jsonl_path']}")


if __name__ == "__main__":
    main()
