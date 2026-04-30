import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from datasets import DatasetDict, Video, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

HF_DATASET_ID = "ud-smart-city/fall-detection"
HF_LOCAL_DATASET_DIR = Path("data/hf_fall_detection/dataset")
OUTPUT_ROOT = Path("data/hf_fall_detection/useful_frames")


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
    except Exception:
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


def frame_scores(
    prev_gray: Optional[np.ndarray],
    current_bgr: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness: higher Laplacian variance means less blur.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Motion: mean absolute pixel difference vs previous sampled frame.
    motion = 0.0
    if prev_gray is not None:
        motion = float(cv2.absdiff(gray, prev_gray).mean())

    # Brightness sanity check to de-prioritize very dark/washed frames.
    brightness = float(gray.mean())

    return gray, sharpness, motion, brightness


def combined_score(sharpness: float, motion: float, brightness: float) -> float:
    brightness_center = 127.0
    brightness_penalty = abs(brightness - brightness_center) / brightness_center
    brightness_weight = max(0.0, 1.0 - min(1.0, brightness_penalty))

    return (0.45 * motion) + (0.45 * sharpness) + (0.10 * (brightness_weight * 100.0))


def select_top_frames(
    video_path: str,
    out_dir: Path,
    sample_every_n: int,
    top_k: int,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video_path": video_path,
            "status": "error",
            "reason": "cannot_open_video",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = -1
    prev_sampled_gray = None
    candidates: List[Dict[str, Any]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every_n != 0:
            continue

        gray, sharpness, motion, brightness = frame_scores(prev_sampled_gray, frame)
        score = combined_score(sharpness, motion, brightness)

        candidates.append(
            {
                "frame_index": frame_idx,
                "score": score,
                "sharpness": sharpness,
                "motion": motion,
                "brightness": brightness,
                "frame": frame,
            }
        )
        prev_sampled_gray = gray

    cap.release()

    if not candidates:
        return {
            "video_path": video_path,
            "status": "empty",
            "reason": "no_frames_sampled",
            "fps": fps,
        }

    top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
    top = sorted(top, key=lambda x: x["frame_index"])

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_meta: List[Dict[str, Any]] = []

    for rank, item in enumerate(top):
        frame_name = f"frame_{item['frame_index']:06d}_rank{rank + 1}.jpg"
        frame_path = out_dir / frame_name
        cv2.imwrite(str(frame_path), item["frame"])

        selected_meta.append(
            {
                "frame_index": item["frame_index"],
                "file": frame_name,
                "score": round(float(item["score"]), 4),
                "sharpness": round(float(item["sharpness"]), 4),
                "motion": round(float(item["motion"]), 4),
                "brightness": round(float(item["brightness"]), 4),
            }
        )

    return {
        "video_path": video_path,
        "status": "ok",
        "fps": fps,
        "sampled_candidates": len(candidates),
        "selected_count": len(selected_meta),
        "selected_frames": selected_meta,
    }


def load_hf_dataset() -> DatasetDict:
    if HF_LOCAL_DATASET_DIR.exists():
        print(f"Loading local dataset from {HF_LOCAL_DATASET_DIR}")
        ds = load_from_disk(str(HF_LOCAL_DATASET_DIR))
    else:
        print(f"Local dataset not found. Loading from Hub: {HF_DATASET_ID}")
        ds = load_dataset(HF_DATASET_ID)

    # Keep video column as path metadata only to avoid torchcodec decode errors.
    for split_name in ds.keys():
        if "video" in ds[split_name].column_names:
            ds[split_name] = ds[split_name].cast_column("video", Video(decode=False))

    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract useful frames from HF videos")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=5,
        help="Sample one frame every N frames",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top K useful frames to keep per video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = load_hf_dataset()
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    split_ds = ds[args.split]
    summary: Dict[str, Any] = {
        "split": args.split,
        "sample_every_n": args.sample_every_n,
        "top_k": args.top_k,
        "videos": [],
    }

    print(f"Processing split '{args.split}' with {len(split_ds)} rows")
    temp_video_files: List[str] = []
    for idx, row in enumerate(split_ds):
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
        result = select_top_frames(
            video_path=video_path,
            out_dir=out_dir,
            sample_every_n=args.sample_every_n,
            top_k=args.top_k,
        )

        result["row_index"] = idx
        summary["videos"].append(result)
        print(f"[{idx + 1}/{len(split_ds)}] {result['status']} - {video_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / f"{args.split}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for tmp_path in temp_video_files:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    print(f"\nDone. Useful frames saved under: {OUTPUT_ROOT / args.split}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()