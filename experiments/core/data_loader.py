"""
Load pose JSONs and existing window predictions from the final_pipeline.

We never write back into final_pipeline. We only read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import POSES_DIR, RESULTS_DIR_FINAL, WINDOWS_BALANCED_DIR


def list_pose_files(split: str, label: Optional[str] = None) -> List[Path]:
    """Return all pose JSON paths for a split (optionally filtered by label)."""
    base = POSES_DIR / split
    if not base.exists():
        return []
    labels = [label] if label else ["fall", "no_fall"]
    out: List[Path] = []
    for lbl in labels:
        d = base / lbl
        if d.exists():
            out.extend(sorted(d.glob("*.json")))
    return out


def load_pose(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def iter_pose_videos(split: str, label: Optional[str] = None) -> Iterable[Dict]:
    """Yield {video_id, label, dataset, frames:[...]} dicts for each video."""
    for p in list_pose_files(split, label):
        data = load_pose(p)
        lbl = "fall" if "fall" in p.parent.name else "no_fall"
        vid = data.get("video_id", p.stem)
        dataset = vid.split("_", 1)[0] if "_" in vid else "unknown"
        yield {
            "video_id": vid,
            "label": lbl,
            "dataset": dataset,
            "frames": data.get("frames", []),
        }


def video_id_to_dataset(video_id: str) -> str:
    return video_id.split("_", 1)[0] if "_" in video_id else "unknown"


# ---------------------------------------------------------------------------
# Existing predictions (from final_pipeline/results/<strategy>/)
# ---------------------------------------------------------------------------

def load_existing_window_predictions(strategy: str, split: str = "test") -> List[Dict]:
    """
    Load window-level predictions from a previously-run strategy.

    Returns a list of {window_id, video_id, true_label, predicted, ...}.
    """
    path = RESULTS_DIR_FINAL / strategy / f"window_results_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"No window results at {path}")
    with open(path) as f:
        data = json.load(f)
    # Some scripts forget to include video_id; recover from window_id prefix.
    for r in data:
        if "video_id" not in r or not r["video_id"]:
            wid = r.get("window_id", "")
            r["video_id"] = wid.rsplit("_w", 1)[0]
    return data


def load_existing_video_predictions(strategy: str, split: str = "test") -> List[Dict]:
    path = RESULTS_DIR_FINAL / strategy / f"video_results_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"No video results at {path}")
    with open(path) as f:
        return json.load(f)


def load_window(window_id: str, split: str, label: str) -> Optional[Dict]:
    """Load a single balanced-window JSON by id.

    `split` may be a single split name ('train'/'val'/'test') or a combination
    like 'val_test'; we try each component split in order.
    """
    video_id = window_id.rsplit("_w", 1)[0]
    candidate_splits = split.split("_") if "_" in split else [split]
    for s in candidate_splits:
        p = WINDOWS_BALANCED_DIR / s / label / video_id / f"{window_id}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def video_window_index(strategy: str, split: str) -> Dict[str, List[Dict]]:
    """
    Build {video_id: [window_record sorted by window_index]} for a strategy.

    Each window_record carries window_id, true_label, predicted, window_index,
    end_posture, is_horizontal_last, is_on_ground_last (pulled from window JSON).
    """
    preds = load_existing_window_predictions(strategy, split)
    by_video: Dict[str, List[Dict]] = {}
    for r in preds:
        wid = r["window_id"]
        vid = r["video_id"]
        true_label = r["true_label"]
        window = load_window(wid, split, true_label)
        idx = window.get("window_index", 0) if window else 0
        analysis = (window or {}).get("window_analysis", {})
        sequence = (window or {}).get("sequence", {})
        last_frame_key = max(sequence.keys()) if sequence else None
        last_enh = (sequence.get(last_frame_key, {}) or {}).get("enhanced", {}) if last_frame_key else {}
        rec = {
            "window_id": wid,
            "video_id": vid,
            "true_label": true_label,
            "predicted": r["predicted"],
            "window_index": idx,
            "end_posture": analysis.get("end_posture") or last_enh.get("posture_state", "unknown"),
            "is_horizontal_last": bool(last_enh.get("is_horizontal", False)),
            "is_on_ground_last": bool(last_enh.get("is_on_ground", False)),
            "max_velocity": analysis.get("max_velocity", 0.0) or 0.0,
            "has_velocity_spike": bool(analysis.get("has_velocity_spike", False)),
        }
        by_video.setdefault(vid, []).append(rec)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["window_index"])
    return by_video
