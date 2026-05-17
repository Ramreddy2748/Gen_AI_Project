"""
Pinned prompts for the experiments folder.

Each prompt is referenced by an explicit name so we can A/B them without
copy-pasting and so the eventual paper can cite a stable identifier.
"""

# ---------------------------------------------------------------------------
# Stage 1 — high-recall safety-first prompt (same as Best Prompt in final_pipeline)
# ---------------------------------------------------------------------------

STAGE1_SAFETY_FIRST = """You are a SAFETY-CRITICAL fall detection system for elderly monitoring.

MISSION: Protect elderly people by detecting falls. Missing a real fall could be life-threatening.

Analyze the pose sequence for FALL INDICATORS:

STRONG FALL SIGNS (any 1 = likely fall):
- rapid_descent flag = True
- on_ground flag = True
- horizontal flag = True
- posture = "fallen"
- velocity > 0.1 (rapid movement)
- body angle < 30 degrees (horizontal)

MODERATE FALL SIGNS (2+ together = likely fall):
- hip_y > 0.6 (person low)
- body angle < 50 degrees (tilted)
- descending trajectory
- velocity spike

CLEAR NO-FALL SIGNS (all needed for NO_FALL):
- posture = "upright" throughout
- body angle > 60 degrees throughout
- velocity < 0.05 (stable)
- no fall flags triggered

When in doubt, choose FALL (safety first).

Respond with ONLY: FALL or NO_FALL"""


# ---------------------------------------------------------------------------
# Stage 2 — strict false-positive filter
# Only runs on Stage-1 fall candidates and is asked to reject sit/bend/lie events.
# ---------------------------------------------------------------------------

STAGE2_FP_FILTER = """You are a strict fall-detection auditor. The previous stage flagged
this sequence as a possible fall. Your job is to REJECT false positives.

Common false positives to REJECT (classify as NO_FALL):
- Sitting down on a chair, sofa, or bed (controlled descent, ends seated)
- Bending over to pick something up (body bends but feet stay planted)
- Lying down voluntarily (slow, controlled, no sudden velocity spike)
- Kneeling or crouching (downward motion but controlled)
- Stretching to reach the floor (brief horizontal-looking pose)

A real FALL has ALL of these together:
- Sudden/uncontrolled downward velocity (velocity spike during the sequence)
- The final frame(s) show the body horizontal AND low (hip near ground)
- The descent is NOT a smooth, monotonic motion of someone deliberately sitting

If you see ANY of the false-positive patterns and no clear evidence of
uncontrolled impact, answer NO_FALL.

Respond with ONLY: FALL or NO_FALL"""


# ---------------------------------------------------------------------------
# Helper formatter for window data (matches the convention used in
# final_pipeline/prompting/best_prompt.py)
# ---------------------------------------------------------------------------

def format_window_for_prompt(window_data: dict) -> str:
    """Render a window JSON into the text we feed to the model."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})

    lines = ["Pose Data:"]
    n = sum(1 for k in sequence if k.startswith("frame_"))
    for i in range(1, n + 1):
        frame = sequence.get(f"frame_{i}", {})
        if frame.get("pose_detected"):
            feats = frame.get("features", {}) or {}
            enh = frame.get("enhanced", {}) or {}
            hip_y = feats.get("hip_y", 0) or 0
            angle = feats.get("body_angle_degrees", 90) or 90
            vel = enh.get("velocity_hip_y", 0) or 0
            posture = enh.get("posture_state", "unknown")
            lines.append(
                f"Frame {i}: hip_y={hip_y:.2f}, angle={angle:.0f}deg, "
                f"velocity={vel:.3f}, posture={posture}"
            )
            lines.append(
                f"  Flags: rapid_descent={enh.get('is_rapid_descent', False)}, "
                f"on_ground={enh.get('is_on_ground', False)}, "
                f"horizontal={enh.get('is_horizontal', False)}"
            )
        else:
            lines.append(f"Frame {i}: pose not detected")

    lines.append(
        f"\nSequence: trajectory={analysis.get('trajectory_direction', 'unknown')}, "
        f"max_velocity={analysis.get('max_velocity', 0) or 0:.3f}, "
        f"velocity_spike={analysis.get('has_velocity_spike', False)}"
    )
    lines.append("\nClassification:")
    return "\n".join(lines)
