"""
Video-level aggregation strategies.

All aggregators take a list of per-window records sorted by window_index and
return "fall" or "no_fall" for the video.

Each record is a dict with at least:
    predicted        : "fall" | "no_fall" | "unknown"
    end_posture      : "fallen" | "upright" | "transitioning" | "unknown"
    is_horizontal_last
    is_on_ground_last
    confidence       : float in [0,1]   (optional; defaults to 1.0 if predicted=='fall')

Strategies implemented:
    - ratio_threshold        (the current "≥25% windows" rule)
    - consecutive_k          (require k consecutive fall windows)
    - end_state              (require the last fall-window to end fallen/horizontal/on_ground)
    - consecutive_and_end    (combine the previous two)
    - confidence_weighted    (sum confidences; threshold = T)
    - any_fall               (any single fall window flags the video)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence


END_FALL_STATES = {"fallen"}


def _confidence(rec: Dict) -> float:
    if "confidence" in rec and rec["confidence"] is not None:
        return float(rec["confidence"])
    # Fall-back: 1.0 if the LLM said fall, else 0.0
    return 1.0 if rec.get("predicted") == "fall" else 0.0


def ratio_threshold(records: List[Dict], threshold: float = 0.25) -> str:
    if not records:
        return "no_fall"
    n = len(records)
    falls = sum(1 for r in records if r["predicted"] == "fall")
    return "fall" if (falls / n) >= threshold else "no_fall"


def consecutive_k(records: List[Dict], k: int = 2) -> str:
    run = 0
    for r in records:
        if r["predicted"] == "fall":
            run += 1
            if run >= k:
                return "fall"
        else:
            run = 0
    return "no_fall"


def end_state(records: List[Dict], min_falls: int = 1) -> str:
    """
    Require at least one fall-predicted window whose end-state is consistent
    with someone actually being down (fallen posture OR horizontal OR
    on_ground in the last frame of the window).
    """
    matches = [
        r for r in records
        if r["predicted"] == "fall"
        and (
            r.get("end_posture") in END_FALL_STATES
            or r.get("is_horizontal_last")
            or r.get("is_on_ground_last")
        )
    ]
    return "fall" if len(matches) >= min_falls else "no_fall"


def consecutive_and_end(records: List[Dict], k: int = 2) -> str:
    """Need k consecutive fall predictions AND at least one of those windows
    ends in a fallen/horizontal/on-ground state."""
    in_run: List[Dict] = []
    best_run: List[Dict] = []
    for r in records:
        if r["predicted"] == "fall":
            in_run.append(r)
            if len(in_run) > len(best_run):
                best_run = list(in_run)
        else:
            in_run = []
    if len(best_run) < k:
        return "no_fall"
    if any(
        r.get("end_posture") in END_FALL_STATES
        or r.get("is_horizontal_last")
        or r.get("is_on_ground_last")
        for r in best_run
    ):
        return "fall"
    return "no_fall"


def confidence_weighted(records: List[Dict], threshold: float = 1.5) -> str:
    """Sum confidences of fall-predicted windows; threshold on val."""
    total = sum(_confidence(r) for r in records if r["predicted"] == "fall")
    return "fall" if total >= threshold else "no_fall"


def any_fall(records: List[Dict]) -> str:
    return "fall" if any(r["predicted"] == "fall" for r in records) else "no_fall"


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

AggregatorFn = Callable[[List[Dict]], str]


def make_ratio(threshold: float) -> AggregatorFn:
    return lambda recs: ratio_threshold(recs, threshold)


def make_consecutive(k: int) -> AggregatorFn:
    return lambda recs: consecutive_k(recs, k)


def make_consecutive_end(k: int) -> AggregatorFn:
    return lambda recs: consecutive_and_end(recs, k)


def make_confidence_weighted(threshold: float) -> AggregatorFn:
    return lambda recs: confidence_weighted(recs, threshold)


def predict_video(records: List[Dict], strategy: AggregatorFn) -> str:
    return strategy(records)


def predict_all(by_video: Dict[str, List[Dict]], strategy: AggregatorFn) -> Dict[str, str]:
    return {vid: strategy(recs) for vid, recs in by_video.items()}


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_ratio_threshold(
    by_video: Dict[str, List[Dict]],
    thresholds: Sequence[float],
) -> List[Dict]:
    out = []
    for t in thresholds:
        preds = predict_all(by_video, make_ratio(t))
        out.append({"strategy": f"ratio>={t:.2f}", "threshold": t, "preds": preds})
    return out


def sweep_confidence_threshold(
    by_video: Dict[str, List[Dict]],
    thresholds: Sequence[float],
) -> List[Dict]:
    out = []
    for t in thresholds:
        preds = predict_all(by_video, make_confidence_weighted(t))
        out.append({"strategy": f"conf>={t:.2f}", "threshold": t, "preds": preds})
    return out
