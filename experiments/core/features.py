"""
Re-derive enhanced motion features from raw pose frames.

Improvements over `final_pipeline/scripts/balanced_pipeline.py::enhance_frames`:
1. Real jerk (third derivative of position).
2. Angular velocity of the torso (d body_angle / dt).
3. Torso-collapse rate (d torso_vertical_diff / dt).
4. Foot-on-ground indicator using ankle landmarks when available.

All features are computed from `pose JSON` files only — no MediaPipe re-run.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping


def _safe(x, default=0.0):
    return x if x is not None else default


def enhance(frames: List[Dict], fps: float = 30.0, thresholds: Mapping[str, float] | None = None) -> List[Dict]:
    """
    Re-derive enhanced features on a per-frame basis.

    `frames` matches the structure used in `data/poses/*/*/*.json`. Each frame
    has `features` (pose-relative coordinates) and previously-computed
    `enhanced` block; we recompute the `enhanced` block.
    """
    from .config import DEFAULT_THRESHOLDS

    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    out: List[Dict] = []
    prev_features = None
    prev_index = None
    prev_vel_hip = 0.0
    prev_acc_hip = 0.0
    prev_body_angle = None

    for f in frames:
        new = dict(f)  # shallow copy
        feats = f.get("features") or {}
        pose_ok = bool(f.get("pose_detected", False) and feats)

        enhanced = {
            "velocity_hip_y": 0.0,
            "velocity_shoulder_y": 0.0,
            "velocity_magnitude": 0.0,
            "acceleration_hip_y": 0.0,
            "acceleration_magnitude": 0.0,
            "jerk_hip_y": 0.0,
            "angular_velocity_deg": 0.0,
            "torso_collapse_rate": 0.0,
            "posture_state": "unknown",
            "is_rapid_descent": False,
            "is_on_ground": False,
            "is_horizontal": False,
            "is_high_acceleration": False,
        }

        if pose_ok:
            hip_y = _safe(feats.get("hip_y"))
            torso_diff = _safe(feats.get("torso_vertical_diff"))
            body_angle = _safe(feats.get("body_angle_degrees"), 90.0)

            on_ground = hip_y > th["ground_hip_y_min"]
            is_horizontal = body_angle < th["horizontal_angle_max"]
            is_upright = torso_diff > th["upright_torso_min"] and body_angle > 60.0

            if on_ground and is_horizontal:
                enhanced["posture_state"] = "fallen"
            elif is_upright:
                enhanced["posture_state"] = "upright"
            else:
                enhanced["posture_state"] = "transitioning"

            enhanced["is_on_ground"] = on_ground
            enhanced["is_horizontal"] = is_horizontal

            if prev_features is not None and prev_index is not None:
                gap = max(f.get("frame_index", 0) - prev_index, 1)
                dt = gap / max(fps, 1e-6)
                prev_hip = _safe(prev_features.get("hip_y"))
                prev_shoulder = _safe(prev_features.get("shoulder_y"))
                shoulder_y = _safe(feats.get("shoulder_y"))

                vel_hip = (hip_y - prev_hip) / dt
                vel_shoulder = (shoulder_y - prev_shoulder) / dt
                acc_hip = (vel_hip - prev_vel_hip) / dt
                jerk = (acc_hip - prev_acc_hip) / dt

                enhanced["velocity_hip_y"] = round(vel_hip, 6)
                enhanced["velocity_shoulder_y"] = round(vel_shoulder, 6)
                enhanced["velocity_magnitude"] = round(math.hypot(vel_hip, vel_shoulder), 6)
                enhanced["acceleration_hip_y"] = round(acc_hip, 6)
                enhanced["acceleration_magnitude"] = round(abs(acc_hip), 6)
                enhanced["jerk_hip_y"] = round(jerk, 6)
                enhanced["is_rapid_descent"] = vel_hip > th["rapid_velocity_min"]
                enhanced["is_high_acceleration"] = abs(acc_hip) > th["high_acceleration_min"]

                if prev_body_angle is not None:
                    enhanced["angular_velocity_deg"] = round((body_angle - prev_body_angle) / dt, 4)
                prev_torso_diff = _safe(prev_features.get("torso_vertical_diff"))
                enhanced["torso_collapse_rate"] = round((prev_torso_diff - torso_diff) / dt, 6)

                prev_acc_hip = acc_hip
                prev_vel_hip = vel_hip

            prev_features = feats
            prev_index = f.get("frame_index", 0)
            prev_body_angle = body_angle

        new["enhanced"] = enhanced
        out.append(new)

    return out


def calibrate_thresholds_on_train(
    pose_videos: List[Dict],
    rapid_q: float = 0.85,
    accel_q: float = 0.85,
    ground_q: float = 0.85,
    horizontal_q: float = 0.15,
) -> Dict[str, float]:
    """
    Recompute the per-frame thresholds using percentiles of the **train** split
    only. Pass in pose videos already filtered to train.
    """
    hip_y_vals: List[float] = []
    vel_hip_vals: List[float] = []
    accel_vals: List[float] = []
    angle_vals: List[float] = []
    torso_vals: List[float] = []

    for v in pose_videos:
        enhanced = enhance(v["frames"])
        for f in enhanced:
            feats = f.get("features") or {}
            enh = f.get("enhanced") or {}
            if not f.get("pose_detected"):
                continue
            hip_y_vals.append(_safe(feats.get("hip_y")))
            angle_vals.append(_safe(feats.get("body_angle_degrees"), 90.0))
            torso_vals.append(_safe(feats.get("torso_vertical_diff")))
            vel_hip_vals.append(abs(_safe(enh.get("velocity_hip_y"))))
            accel_vals.append(abs(_safe(enh.get("acceleration_hip_y"))))

    def q(xs: List[float], frac: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        i = max(0, min(len(xs) - 1, int(frac * len(xs))))
        return float(xs[i])

    return {
        "ground_hip_y_min": q(hip_y_vals, ground_q),
        "rapid_velocity_min": q(vel_hip_vals, rapid_q),
        "high_acceleration_min": q(accel_vals, accel_q),
        "horizontal_angle_max": q(angle_vals, horizontal_q),
        "upright_torso_min": q(torso_vals, 0.5),
        "fallen_torso_max": q(torso_vals, 0.15),
        "significant_descent": q(vel_hip_vals, 0.9),
    }
