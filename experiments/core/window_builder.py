"""
Build sliding windows of arbitrary size from existing pose JSONs.

This is the cheap path that lets us run the window-size sweep (3, 5, 7, 9,
12, 15) without re-running MediaPipe on the raw videos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .config import EXP_DATA
from .data_loader import iter_pose_videos
from .features import enhance


def _window_analysis(window_frames: List[Dict], thresholds: Optional[Dict[str, float]] = None) -> Dict:
    pose_frames = [f for f in window_frames if f.get("pose_detected")]
    n = len(window_frames)
    analysis = {
        "pose_detection_count": len(pose_frames),
        "pose_detection_rate": round(len(pose_frames) / n, 4) if n else 0.0,
    }
    if len(pose_frames) < 2:
        analysis["fall_likelihood_score"] = 0.0
        return analysis

    first, last = pose_frames[0], pose_frames[-1]
    first_hip = (first.get("features") or {}).get("hip_y", 0) or 0
    last_hip = (last.get("features") or {}).get("hip_y", 0) or 0
    hip_delta = last_hip - first_hip
    max_vel = max(
        (f.get("enhanced", {}).get("velocity_magnitude", 0) or 0) for f in pose_frames
    )
    max_jerk = max(
        abs(f.get("enhanced", {}).get("jerk_hip_y", 0) or 0) for f in pose_frames
    )
    max_ang = max(
        abs(f.get("enhanced", {}).get("angular_velocity_deg", 0) or 0) for f in pose_frames
    )
    analysis.update({
        "hip_delta": round(hip_delta, 6),
        "max_velocity": round(max_vel, 6),
        "max_jerk": round(max_jerk, 6),
        "max_angular_velocity": round(max_ang, 6),
        "trajectory_direction": "descending" if hip_delta > 0.05 else "stable",
        "has_velocity_spike": any(
            f.get("enhanced", {}).get("is_rapid_descent") for f in pose_frames
        ),
        "start_posture": first.get("enhanced", {}).get("posture_state", "unknown"),
        "end_posture": last.get("enhanced", {}).get("posture_state", "unknown"),
        "end_is_horizontal": bool(last.get("enhanced", {}).get("is_horizontal", False)),
        "end_is_on_ground": bool(last.get("enhanced", {}).get("is_on_ground", False)),
    })
    return analysis


def build_windows_for_video(
    video: Dict,
    window_size: int,
    stride: int = 1,
    thresholds: Optional[Dict[str, float]] = None,
    fps: float = 30.0,
) -> List[Dict]:
    """Build sliding windows of `window_size` for one video."""
    frames = enhance(video["frames"], fps=fps, thresholds=thresholds)
    if len(frames) < window_size:
        return []

    windows: List[Dict] = []
    for i, start in enumerate(range(0, len(frames) - window_size + 1, stride)):
        wf = frames[start:start + window_size]
        sequence = {f"frame_{j + 1}": f for j, f in enumerate(wf)}
        analysis = _window_analysis(wf, thresholds)
        wid = f"{video['video_id']}_n{window_size:02d}_w{i:04d}"
        windows.append({
            "window_id": wid,
            "video_id": video["video_id"],
            "dataset": video.get("dataset", "unknown"),
            "label": video["label"],
            "window_size": window_size,
            "stride": stride,
            "window_index": i,
            "sequence": sequence,
            "window_analysis": analysis,
        })
    return windows


def build_window_set(
    split: str,
    window_size: int,
    stride: int = 1,
    thresholds: Optional[Dict[str, float]] = None,
    out_root: Optional[Path] = None,
    fps: float = 30.0,
) -> Path:
    """
    Build a full window set for one split and one window_size. Saved under
    `experiments/data/windows_n{N}/{split}/{label}/{video_id}/{wid}.json`.
    Returns the output root.
    """
    out_root = out_root or (EXP_DATA / f"windows_n{window_size:02d}")
    out_split = out_root / split
    out_split.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_videos = 0
    for v in iter_pose_videos(split):
        wins = build_windows_for_video(
            v, window_size=window_size, stride=stride,
            thresholds=thresholds, fps=fps,
        )
        if not wins:
            continue
        n_videos += 1
        for w in wins:
            d = out_split / w["label"] / w["video_id"]
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{w['window_id']}.json", "w") as f:
                json.dump(w, f, indent=2)
            manifest.append({
                "window_id": w["window_id"],
                "video_id": w["video_id"],
                "label": w["label"],
                "dataset": w["dataset"],
                "split": split,
                "window_size": window_size,
                "window_index": w["window_index"],
                "fall_likelihood_score": w["window_analysis"].get("fall_likelihood_score", 0.0),
            })

    manifest_path = out_root / f"manifest_{split}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[window_builder] split={split} n={window_size}: {n_videos} videos -> {len(manifest)} windows")
    return out_root
