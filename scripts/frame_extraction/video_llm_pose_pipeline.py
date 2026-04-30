import argparse
import base64
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets import Video
from huggingface_hub import hf_hub_download
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

HF_DATASET_ID = "ud-smart-city/fall-detection"
HF_LOCAL_DATASET_DIR = Path("data/hf_fall_detection/dataset")
OUTPUT_ROOT = Path("data/hf_fall_detection/llm_pose_pipeline")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path("pose_landmarker_lite.task")


def ensure_pose_model() -> None:
    if MODEL_PATH.exists():
        return
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video -> LLM useful frames -> pose feature extraction"
    )
    parser.add_argument(
        "--fall-episode-mode",
        action="store_true",
        help="Extract variable-length fall episode window: from first upright to last on ground (pose-based)",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=5,
        help="Sample one frame every N frames from video",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=24,
        help="Max sampled candidates scored by LLM per video",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top K useful frames selected by LLM",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llava",
        help="Ollama model name for frame usefulness scoring",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for LLM scoring (seconds)",
    )
    # ... add other arguments as needed ...
    return parser.parse_args()
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top K useful frames selected by LLM",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llava",
        help="Ollama model name for frame usefulness scoring",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="HTTP timeout for each LLM frame request",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on number of videos processed from the split",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use lightweight settings for a fast smoke test",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=3,
        help="Number of frames before the anchor frame to export",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=3,
        help="Number of frames after the anchor frame to export",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="event-window",
        choices=["event-window", "anchor-window"],
        help="Frame selection strategy",
    )
    parser.add_argument(
        "--before-person-start",
        type=int,
        default=1,
        help="Frames before person first appears",
    )
    parser.add_argument(
        "--before-motion-change",
        type=int,
        default=3,
        help="Frames before detected movement change",
    )
    parser.add_argument(
        "--before-fall",
        type=int,
        default=5,
        help="Frames before estimated fall frame",
    )
    parser.add_argument(
        "--after-fall",
        type=int,
        default=5,
        help="Frames after estimated fall frame",
    )
    return parser.parse_args()


def load_hf_dataset() -> DatasetDict:
    if HF_LOCAL_DATASET_DIR.exists():
        print(f"Loading local dataset from {HF_LOCAL_DATASET_DIR}")
        ds = load_from_disk(str(HF_LOCAL_DATASET_DIR))
    else:
        print(f"Local dataset not found. Loading from Hub: {HF_DATASET_ID}")
        ds = load_dataset(HF_DATASET_ID)

    # Keep video column as metadata/path only; avoid runtime video decoding dependency.
    for split_name in ds.keys():
        if "video" in ds[split_name].column_names:
            ds[split_name] = ds[split_name].cast_column("video", Video(decode=False))

    return ds


def get_video_path(row: Dict[str, Any]) -> Optional[str]:
    video_value = row.get("video")
    if isinstance(video_value, dict):
        return video_value.get("path")
    if isinstance(video_value, str):
        return video_value
    return None


def resolve_video_path(raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None

    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return str(p)

    candidates = [
        Path.cwd() / raw_path,
        HF_LOCAL_DATASET_DIR / raw_path,
        HF_LOCAL_DATASET_DIR.parent / raw_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        matches = list(hf_cache.rglob(Path(raw_path).name))
        if matches:
            return str(matches[0])

    # Final fallback: try downloading this file directly from the dataset repo.
    try:
        downloaded = hf_hub_download(repo_id=HF_DATASET_ID, filename=raw_path, repo_type="dataset")
        if downloaded and Path(downloaded).exists():
            return downloaded
    except (OSError, ValueError, RuntimeError):
        pass

    return None


def materialize_video_bytes(row: Dict[str, Any]) -> Optional[str]:
    video_value = row.get("video")
    if not isinstance(video_value, dict):
        return None

    data = video_value.get("bytes")
    if not data:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def sample_candidate_frames(
    video_path: str,
    sample_every_n: int,
    candidate_limit: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = -1
    prev_gray = None
    candidates: List[Dict[str, Any]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
        heuristic = (0.55 * motion) + (0.45 * sharpness)

        candidates.append(
            {
                "frame_index": frame_idx,
                "frame": frame,
                "motion": motion,
                "sharpness": sharpness,
                "heuristic_score": heuristic,
            }
        )
        prev_gray = gray

    cap.release()

    if not candidates:
        return fps, []

    candidates = sorted(candidates, key=lambda x: x["heuristic_score"], reverse=True)
    return fps, candidates[:candidate_limit]


def sample_timeline_frames(
    video_path: str,
    sample_every_n: int,
    pose: mp_vision.PoseLandmarker,
) -> Tuple[float, List[Dict[str, Any]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = -1
    prev_gray = None
    samples: List[Dict[str, Any]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
        prev_gray = gray

        pose_feats = extract_pose_features(frame, pose)
        samples.append(
            {
                "frame_index": frame_idx,
                "frame": frame,
                "motion": motion,
                "pose_features": pose_feats,
                "pose_found": bool(pose_feats.get("pose_found", False)),
                "torso_vertical_diff": float(pose_feats.get("torso_vertical_diff", 1.0)),
            }
        )

    cap.release()
    return fps, samples


def detect_event_indices(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    if not samples:
        return {
            "person_start": 0,
            "motion_change": 0,
            "fall": 0,
        }

    person_start = next(
        (i for i, s in enumerate(samples) if s.get("pose_found", False)),
        0,
    )

    motion_values = [float(s.get("motion", 0.0)) for s in samples]
    motion_change = person_start
    if len(samples) > person_start + 2:
        deltas: List[Tuple[int, float]] = []
        for i in range(person_start + 1, len(motion_values)):
            deltas.append((i, motion_values[i] - motion_values[i - 1]))
        if deltas:
            motion_change = max(deltas, key=lambda t: t[1])[0]

    pose_candidates = [
        (i, s) for i, s in enumerate(samples)
        if s.get("pose_found", False)
    ]
    if pose_candidates:
        filtered = [(i, s) for i, s in pose_candidates if i >= motion_change]
        if not filtered:
            filtered = pose_candidates
        fall = min(filtered, key=lambda t: float(t[1].get("torso_vertical_diff", 1.0)))[0]
    else:
        fall = max(range(len(motion_values)), key=lambda i: motion_values[i])

    return {
        "person_start": person_start,
        "motion_change": motion_change,
        "fall": fall,
    }


def add_range_indices(start: int, end: int, label: str, labeled: Dict[int, List[str]]) -> None:
    if end < start:
        return
    for idx in range(start, end + 1):
        labeled.setdefault(idx, []).append(label)


def build_event_window_indices(
    events: Dict[str, int],
    total: int,
    before_person_start: int,
    before_motion_change: int,
    before_fall: int,
    after_fall: int,
) -> Dict[int, List[str]]:
    labeled: Dict[int, List[str]] = {}

    person_start = events["person_start"]
    motion_change = events["motion_change"]
    fall = events["fall"]

    add_range_indices(
        max(0, person_start - before_person_start),
        max(-1, person_start - 1),
        "before_person_start",
        labeled,
    )
    add_range_indices(
        max(0, motion_change - before_motion_change),
        max(-1, motion_change - 1),
        "before_motion_change",
        labeled,
    )
    add_range_indices(
        max(0, fall - before_fall),
        max(-1, fall - 1),
        "before_fall",
        labeled,
    )
    add_range_indices(
        min(total - 1, fall + 1),
        min(total - 1, fall + after_fall),
        "after_fall",
        labeled,
    )

    if 0 <= fall < total:
        labeled.setdefault(fall, []).append("fall_anchor")

    return labeled


def frame_to_base64_jpg(frame_bgr: Any, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def llm_usefulness_score(
    frame_bgr: Any,
    model: str,
    ollama_url: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    image_b64 = frame_to_base64_jpg(frame_bgr)
    prompt = (
        "You are scoring a single video frame for FALL DETECTION usefulness. "
        "A useful frame should clearly show body posture, potential instability, "
        "or near-fall/fall evidence. Return strict JSON only with keys: "
        "useful (boolean), score (0-100 integer), reason (short string)."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        body = response.json()
        raw_text = str(body.get("response", "")).strip()
        parsed = extract_json_object(raw_text)
        if not parsed:
            return {
                "useful": False,
                "score": 0,
                "reason": "llm_json_parse_failed",
                "raw": raw_text,
            }

        score = int(parsed.get("score", 0))
        score = max(0, min(100, score))
        useful = bool(parsed.get("useful", score >= 50))
        reason = str(parsed.get("reason", ""))[:200]
        return {"useful": useful, "score": score, "reason": reason}

    except requests.RequestException as exc:
        return {
            "useful": False,
            "score": 0,
            "reason": f"llm_request_failed: {exc}",
        }


def build_pose_detector() -> mp_vision.PoseLandmarker:
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_features(frame_bgr: Any, pose: mp_vision.PoseLandmarker) -> Dict[str, Any]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect(mp_image)

    if not result.pose_landmarks:
        return {"pose_found": False}

    lm = result.pose_landmarks[0]
    return {
        "pose_found": True,
        "hip_x": lm[23].x,
        "hip_y": lm[23].y,
        "shoulder_x": lm[11].x,
        "shoulder_y": lm[11].y,
        "nose_x": lm[0].x,
        "nose_y": lm[0].y,
        "torso_vertical_diff": abs(lm[11].y - lm[23].y),
    }


def choose_anchor_frame(scored_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    with_positive_llm = [item for item in scored_frames if item.get("llm_score", 0) > 0]
    ranked = with_positive_llm if with_positive_llm else scored_frames
    ranked.sort(
        key=lambda item: (item.get("llm_score", 0), item.get("heuristic_score", 0.0)),
        reverse=True,
    )
    return ranked[0]


def extract_frame_window(
    video_path: str,
    anchor_index: int,
    before_count: int,
    after_count: int,
) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    start_index = max(0, anchor_index - before_count)
    end_index = anchor_index + after_count
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)

    frames: List[Dict[str, Any]] = []
    current_index = start_index

    while current_index <= end_index:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(
            {
                "frame_index": current_index,
                "relative_offset": current_index - anchor_index,
                "is_anchor": current_index == anchor_index,
                "frame": frame,
            }
        )
        current_index += 1

    cap.release()
    return frames


def process_video(
    video_path: str,
    out_dir: Path,
    args: argparse.Namespace,
    pose: mp_vision.PoseLandmarker,
) -> Dict[str, Any]:

    if args.fall_episode_mode:
        # Variable-length window: from first upright to last on ground (pose-based)
        fps, timeline = sample_timeline_frames(
            video_path=video_path,
            sample_every_n=args.sample_every_n,
            pose=pose,
        )
        if not timeline:
            return {
                "video_path": video_path,
                "status": "empty",
                "reason": "no_timeline_frames",
                "fps": fps,
            }
        # Heuristic: upright if torso_vertical_diff > 0.25, on ground if < 0.13
        upright_thresh = 0.25
        ground_thresh = 0.13
        first_upright = next((i for i, s in enumerate(timeline) if s.get("pose_found") and s.get("torso_vertical_diff", 1.0) > upright_thresh), 0)
        last_ground = 0
        for i in range(len(timeline)-1, -1, -1):
            s = timeline[i]
            if s.get("pose_found") and s.get("torso_vertical_diff", 1.0) < ground_thresh:
                last_ground = i
                break
        if last_ground <= first_upright:
            last_ground = len(timeline) - 1
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_meta: List[Dict[str, Any]] = []
        pose_json: List[Dict[str, Any]] = []
        for timeline_idx in range(first_upright, last_ground + 1):
            item = timeline[timeline_idx]
            frame_index = int(item["frame_index"])
            frame_name = f"frame_{frame_index:06d}_fall_episode.jpg"
            frame_path = out_dir / frame_name
            cv2.imwrite(str(frame_path), item["frame"])
            pose_record = {
                "frame_index": frame_index,
                "timeline_index": timeline_idx,
                "file": frame_name,
                "motion": round(float(item.get("motion", 0.0)), 4),
                **item["pose_features"],
            }
            pose_json.append(pose_record)
            selected_meta.append(
                {
                    "frame_index": frame_index,
                    "timeline_index": timeline_idx,
                    "file": frame_name,
                }
            )
        with open(out_dir / "pose_features.json", "w", encoding="utf-8") as f:
            json.dump(pose_json, f, indent=2)
        return {
            "video_path": video_path,
            "status": "ok",
            "selection_mode": "fall_episode",
            "fps": fps,
            "timeline_frames": len(timeline),
            "fall_episode_start_timeline_index": first_upright,
            "fall_episode_end_timeline_index": last_ground,
            "fall_episode_start_frame_index": int(timeline[first_upright]["frame_index"]),
            "fall_episode_end_frame_index": int(timeline[last_ground]["frame_index"]),
            "selected_count": len(selected_meta),
            "selected_frames": selected_meta,
        }
    if args.selection_mode == "event-window":
        # ...existing code...
        pass
    for item in candidates:
        llm = llm_usefulness_score(
            frame_bgr=item["frame"],
            model=args.llm_model,
            ollama_url=args.ollama_url,
            timeout_seconds=args.timeout_seconds,
        )
        llm_scored.append(
            {
                **item,
                "llm_useful": llm.get("useful", False),
                "llm_score": int(llm.get("score", 0)),
                "llm_reason": llm.get("reason", ""),
            }
        )

    # Try each candidate as anchor (best first) until we find a window
    # where at least one frame has a detectable pose.
    llm_ranked = sorted(
        llm_scored,
        key=lambda x: (x.get("llm_score", 0), x.get("heuristic_score", 0.0)),
        reverse=True,
    )

    anchor = None
    selected = []
    for candidate in llm_ranked:
        window = extract_frame_window(
            video_path=video_path,
            anchor_index=int(candidate["frame_index"]),
            before_count=args.context_before,
            after_count=args.context_after,
        )
        if not window:
            continue
        # Check if MediaPipe can find a person in at least one frame of the window.
        has_pose = any(
            extract_pose_features(f["frame"], pose).get("pose_found", False)
            for f in window
        )
        if has_pose:
            anchor = candidate
            selected = window
            break

    # Fallback: use best-scored candidate even if no pose found.
    if anchor is None:
        anchor = llm_ranked[0]
        selected = extract_frame_window(
            video_path=video_path,
            anchor_index=int(anchor["frame_index"]),
            before_count=args.context_before,
            after_count=args.context_after,
        )

    if not selected:
        return {
            "video_path": video_path,
            "status": "error",
            "reason": "unable_to_extract_anchor_window",
            "fps": fps,
            "anchor_frame_index": int(anchor["frame_index"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_meta: List[Dict[str, Any]] = []
    pose_json: List[Dict[str, Any]] = []

    for item in selected:
        offset = int(item["relative_offset"])
        offset_label = f"{offset:+d}"
        frame_name = f"frame_{item['frame_index']:06d}_offset{offset_label}.jpg"
        frame_path = out_dir / frame_name
        cv2.imwrite(str(frame_path), item["frame"])

        pose_feats = extract_pose_features(item["frame"], pose)
        pose_record = {
            "frame_index": item["frame_index"],
            "relative_offset": item["relative_offset"],
            "is_anchor": item["is_anchor"],
            "file": frame_name,
            "anchor_llm_score": anchor["llm_score"],
            "anchor_llm_useful": anchor["llm_useful"],
            "anchor_llm_reason": anchor["llm_reason"],
            **pose_feats,
        }
        pose_json.append(pose_record)

        selected_meta.append(
            {
                "frame_index": item["frame_index"],
                "relative_offset": item["relative_offset"],
                "is_anchor": item["is_anchor"],
                "file": frame_name,
                "anchor_llm_score": anchor["llm_score"],
                "anchor_llm_useful": anchor["llm_useful"],
                "anchor_llm_reason": anchor["llm_reason"],
            }
        )

    with open(out_dir / "pose_features.json", "w", encoding="utf-8") as f:
        json.dump(pose_json, f, indent=2)

    return {
        "video_path": video_path,
        "status": "ok",
        "selection_mode": args.selection_mode,
        "fps": fps,
        "candidate_count": len(candidates),
        "anchor_frame_index": int(anchor["frame_index"]),
        "context_before": args.context_before,
        "context_after": args.context_after,
        "selected_count": len(selected_meta),
        "selected_frames": selected_meta,
    }


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    args = parse_args()

    if args.quick_test:
        # Fast smoke-test profile to avoid long LLM runs.
        args.sample_every_n = max(args.sample_every_n, 10)
        args.candidate_limit = min(args.candidate_limit, 6)
        args.top_k = min(args.top_k, 3)
        # Keep timeout high enough for llava on CPU (never below 90s).
        args.timeout_seconds = max(args.timeout_seconds, 90)
        if args.max_videos is None:
            args.max_videos = 1

    ensure_pose_model()

    ds = load_hf_dataset()
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    split_ds = ds[args.split]
    summary: Dict[str, Any] = {
        "split": args.split,
        "selection_mode": args.selection_mode,
        "sample_every_n": args.sample_every_n,
        "candidate_limit": args.candidate_limit,
        "top_k": args.top_k,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "before_person_start": args.before_person_start,
        "before_motion_change": args.before_motion_change,
        "before_fall": args.before_fall,
        "after_fall": args.after_fall,
        "llm_model": args.llm_model,
        "videos": [],
    }

    total_rows = len(split_ds)
    rows_to_process = total_rows if args.max_videos is None else min(total_rows, args.max_videos)

    print(
        f"Processing split '{args.split}' with {rows_to_process}/{total_rows} rows"
    )
    pose = build_pose_detector()
    temp_video_files: List[str] = []

    for idx in range(rows_to_process):
        row = split_ds[idx]
        raw_path = get_video_path(row)
        video_path = resolve_video_path(raw_path)

        if not video_path:
            byte_video = materialize_video_bytes(row)
            if byte_video:
                temp_video_files.append(byte_video)
                video_path = byte_video

        if not video_path:
            summary["videos"].append(
                {
                    "row_index": idx,
                    "status": "error",
                    "reason": "missing_or_unresolved_video_path",
                    "raw_video_path": raw_path,
                }
            )
            continue

        out_dir = OUTPUT_ROOT / args.split / f"sample_{idx:04d}"
        result = process_video(video_path=video_path, out_dir=out_dir, args=args, pose=pose)
        result["row_index"] = idx
        summary["videos"].append(result)
        print(f"[{idx + 1}/{rows_to_process}] {result['status']} - {video_path}")

    close_method = getattr(pose, "close", None)
    if close_method is not None:
        close_method()

    for tmp_path in temp_video_files:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / f"{args.split}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Selected frames and pose features saved under: {OUTPUT_ROOT / args.split}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()