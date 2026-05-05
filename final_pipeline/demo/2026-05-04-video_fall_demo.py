"""
Single-Video Fall Detection Demo (Best-Prompt + GPT-4o)
========================================================

Takes one video file → extracts frames → runs MediaPipe pose → builds
3-frame sliding windows in the same schema as the training data → calls the
project's safety-first "Best-Prompt" classifier on each window with GPT-4o
→ aggregates the per-window predictions into a single video-level
FALL / NO_FALL verdict.

This is a *thin* CLI wrapper. It does not modify any existing pipeline file.
It only imports:

  • pose extraction + window construction
        from final_pipeline/scripts/balanced_pipeline.py
  • the BEST_PROMPT classifier
        from final_pipeline/prompting/best_prompt.py

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python 2026-05-04-video_fall_demo.py --video /path/to/clip.mp4

    # Pose + windows only, no API calls (cheap dry run):
    python 2026-05-04-video_fall_demo.py --video clip.mp4 --no-llm

    # Custom output dir, sample every 3rd frame, fire on 1+ fall windows:
    python 2026-05-04-video_fall_demo.py --video clip.mp4 \\
        --output-dir ./my_demo_out --sample-rate 3 --fall-threshold 0.10

Notes
-----
• Not real-time. Roughly 30–90 seconds per 10-second clip, dominated by
  MediaPipe pose detection (CPU) and serial GPT-4o calls (one per window).
• OPENAI_API_KEY must be set unless --no-llm is passed.
• Output: extracted frames, per-window JSONs, and a top-level result.json
  with the verdict and a per-window timeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

# -----------------------------------------------------------------------------
# Path wiring — let us import from sibling pipeline directories without
# packaging the project. We are NOT modifying those modules.
# -----------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent           # .../final_pipeline/demo
PIPELINE_DIR = DEMO_DIR.parent                       # .../final_pipeline
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))
sys.path.insert(0, str(PIPELINE_DIR / "prompting"))

from openai import OpenAI

# Pose + window pipeline (read-only re-use)
from balanced_pipeline import (
    extract_pose,        # frame_path -> {"pose_detected", "features"}
    enhance_frames,      # adds velocity / acceleration / posture
    create_windows,      # builds 3-frame windows w/ sequence + window_analysis
    WINDOW_SIZE,
    FRAME_RESIZE,
)

# Strongest classifier (read-only re-use)
from best_prompt import (
    BEST_PROMPT,         # safety-first system prompt
    extract_features,    # window dict -> compact feature dict
    format_data,         # features -> user prompt string
    parse_response,      # raw model output -> "fall" / "no_fall"
    aggregate_video,     # window preds + threshold -> video pred
    MODEL,               # "gpt-4o"
)


# =============================================================================
# CORE HELPERS
# =============================================================================

def call_gpt4o(client: OpenAI, system: str, user: str, retries: int = 3) -> str:
    """One GPT-4o call with simple 429 back-off. Returns uppercased text or 'ERROR'."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


def extract_frames_from_video(video_path: Path, frames_dir: Path,
                              sample_rate: int, max_frames: int) -> Tuple[List[Dict], float]:
    """
    Sample every Nth frame from the video, save it to disk, and run MediaPipe
    pose on it. Returns (frames_with_pose, fps).

    The returned dicts match the schema balanced_pipeline.process_video produces
    BEFORE enhance_frames runs:
        {frame_index, frame_name, frame_path, timestamp_seconds,
         pose_detected, features}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: List[Dict] = []
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
    return frames, fps


def pick_video_via_dialog() -> str:
    """
    Open a native macOS / Linux / Windows file-picker dialog and return the
    selected video path. Returns "" if the user cancels.

    Uses tkinter, which ships with the standard Python distribution on macOS,
    so no extra dependency is required.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Error: tkinter is not available in this Python build, so the GUI "
              "file picker can't open. Pass --video <path> instead.")
        return ""

    root = tk.Tk()
    root.withdraw()              # don't show the empty Tk window
    root.update()                # nudges the dialog to the front on macOS
    path = filedialog.askopenfilename(
        title="Select a video for fall detection",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or ""


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-video fall detection demo (Best-Prompt + GPT-4o)."
    )
    parser.add_argument("--video", default=None,
                        help="Path to input video (.mp4 / .mov / .avi). "
                             "If omitted, a file picker dialog will open.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write frames, per-window JSONs, and result.json. "
                             "Default: ./demo_out/<video_stem>/")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Sample every Nth frame from the video. Default: 5.")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Cap on total frames sampled. Default: 200.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding-window stride (in sampled frames). Default: 1.")
    parser.add_argument("--fall-threshold", type=float, default=0.25,
                        help="Fraction of windows predicting FALL needed to flag the "
                             "video as a fall. Default: 0.25 (matches best_prompt.py).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the GPT-4o classifier — only do pose + window "
                             "construction. Useful for offline checks.")
    args = parser.parse_args()

    # 0. Validate input + set up output
    chosen = args.video
    if not chosen:
        print("No --video passed. Opening file picker...")
        chosen = pick_video_via_dialog()
        if not chosen:
            print("No video selected. Exiting.")
            sys.exit(0)

    video_path = Path(chosen).expanduser().resolve()
    if not video_path.exists():
        print(f"Error: video not found at {video_path}")
        sys.exit(1)

    out_dir = (Path(args.output_dir).expanduser().resolve()
               if args.output_dir
               else Path.cwd() / "demo_out" / video_path.stem)
    frames_dir = out_dir / "frames"
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video        : {video_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Sample rate  : every {args.sample_rate} frames (max {args.max_frames})")
    print(f"Stride       : {args.stride}")
    print(f"Threshold    : {args.fall_threshold:.0%} of windows predicting FALL "
          f"flags the video")
    print()

    # 1. Frames + per-frame pose
    print("[1/4] Extracting frames and running MediaPipe pose...")
    frames, fps = extract_frames_from_video(
        video_path, frames_dir,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )
    detected = sum(1 for f in frames if f.get("pose_detected"))
    print(f"      fps={fps:.2f}, frames sampled={len(frames)}, pose detected on={detected}")

    if len(frames) < WINDOW_SIZE:
        print(f"Error: need at least {WINDOW_SIZE} sampled frames, got {len(frames)}.")
        print("Try a smaller --sample-rate or a longer video.")
        sys.exit(1)
    if detected == 0:
        print("Warning: MediaPipe did not detect a pose in any sampled frame.")
        print("The classifier will see 'pose not detected' on every window — "
              "predictions will be unreliable.")

    # 2. Add enhanced features (velocity / acceleration / posture)
    print("[2/4] Computing velocity, acceleration, and posture features...")
    enhanced = enhance_frames(frames, fps)

    # 3. Build 3-frame sliding windows in the training-data schema
    print(f"[3/4] Building 3-frame sliding windows (stride={args.stride})...")
    windows = create_windows(
        enhanced,
        video_id=video_path.stem,
        label="unknown",
        split="demo",
        dataset="user",
        stride=args.stride,
    )
    print(f"      {len(windows)} windows built")
    for w in windows:
        with open(windows_dir / f"{w['window_id']}.json", "w") as fh:
            json.dump(w, fh, indent=2)

    if args.no_llm:
        print("[4/4] --no-llm passed; skipping classification.")
        print(f"\nFrames + windows written to: {out_dir}")
        return

    # 4. Classify each window with GPT-4o using the safety-first prompt
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Export it before running, or rerun "
              "with --no-llm to skip classification.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    print(f"[4/4] Classifying {len(windows)} windows with {MODEL} (Best-Prompt strategy)...")
    print()

    window_preds: List[Dict] = []
    raw_preds: List[str] = []

    for i, window in enumerate(windows, 1):
        features = extract_features(window)
        user_text = format_data(features)
        raw = call_gpt4o(client, BEST_PROMPT, user_text)
        pred = parse_response(raw)
        raw_preds.append(pred)

        seq = window["sequence"]
        t_start = seq["frame_1"]["timestamp_seconds"]
        t_end = seq["frame_3"]["timestamp_seconds"]

        window_preds.append({
            "window_index": window["window_index"],
            "window_id": window["window_id"],
            "t_start_seconds": t_start,
            "t_end_seconds": t_end,
            "predicted": pred,
            "raw_response": raw,
        })

        marker = "FALL  " if pred == "fall" else "      "
        print(f"   window {i:3d}/{len(windows)}  "
              f"t={t_start:6.2f}s–{t_end:6.2f}s   {marker}  {pred}")

        time.sleep(0.12)  # gentle rate limiting

    # 5. Aggregate to a single video-level verdict
    video_pred = aggregate_video(raw_preds, threshold=args.fall_threshold)
    fall_count = sum(1 for p in raw_preds if p == "fall")

    print()
    print("=" * 64)
    print(f"  VIDEO VERDICT : {video_pred.upper()}")
    print(f"  {fall_count} of {len(raw_preds)} windows predicted FALL "
          f"(threshold = {args.fall_threshold:.0%})")
    print("=" * 64)

    # 6. Persist run summary
    summary = {
        "video_path": str(video_path),
        "model": MODEL,
        "strategy": "best_prompt_safety_first",
        "fps": fps,
        "frames_sampled": len(frames),
        "pose_detected_frames": detected,
        "window_count": len(windows),
        "windows_predicted_fall": fall_count,
        "fall_threshold": args.fall_threshold,
        "video_prediction": video_pred,
        "windows": window_preds,
    }
    result_path = out_dir / "result.json"
    with open(result_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results : {result_path}")


if __name__ == "__main__":
    main()
