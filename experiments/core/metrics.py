"""
Video-level metrics + bootstrap confidence intervals.

Window-level metrics are intentionally omitted: this project is evaluated at
the video level only.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def confusion(y_true: Sequence[str], y_pred: Sequence[str], positive: str = "fall") -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def video_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """Return video-level metrics. Positive class is 'fall'."""
    cm = confusion(y_true, y_pred, positive="fall")
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]

    fall_prec = _safe_div(tp, tp + fp)
    fall_rec = _safe_div(tp, tp + fn)
    fall_f1 = _safe_div(2 * fall_prec * fall_rec, fall_prec + fall_rec)

    nofall_prec = _safe_div(tn, tn + fn)
    nofall_rec = _safe_div(tn, tn + fp)
    nofall_f1 = _safe_div(2 * nofall_prec * nofall_rec, nofall_prec + nofall_rec)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    macro_f1 = (fall_f1 + nofall_f1) / 2

    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "fall_precision": round(fall_prec, 4),
        "fall_recall": round(fall_rec, 4),
        "fall_f1": round(fall_f1, 4),
        "no_fall_precision": round(nofall_prec, 4),
        "no_fall_recall": round(nofall_rec, 4),
        "no_fall_f1": round(nofall_f1, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "n_videos": tp + tn + fp + fn,
    }


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """
    Resample videos with replacement and report 95% CIs for the headline
    video-level metrics.
    """
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return {}

    boots: Dict[str, List[float]] = {
        "fall_recall": [],
        "fall_precision": [],
        "fall_f1": [],
        "macro_f1": [],
        "accuracy": [],
    }
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        m = video_metrics(yt, yp)
        for k in boots:
            boots[k].append(m[k])

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2
    out: Dict[str, Tuple[float, float]] = {}
    for k, vs in boots.items():
        vs_sorted = sorted(vs)
        lo = vs_sorted[max(0, int(math.floor(lo_q * len(vs_sorted))))]
        hi = vs_sorted[min(len(vs_sorted) - 1, int(math.ceil(hi_q * len(vs_sorted)) - 1))]
        out[k] = (round(lo, 4), round(hi, 4))
    return out


def mcnemar(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
) -> Dict[str, float]:
    """
    McNemar's test (binary, with continuity correction) to compare two
    classifiers on the same set of videos.
    """
    b = sum(1 for t, a, c in zip(y_true, y_pred_a, y_pred_b) if a == t and c != t)
    c = sum(1 for t, a, d in zip(y_true, y_pred_a, y_pred_b) if a != t and d == t)
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = ((abs(b - c) - 1) ** 2) / (b + c)
    # Approximate p-value using chi-squared survival with 1 dof.
    # P(X > stat) for chi^2_1 has closed form: erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2))
    return {"b": b, "c": c, "statistic": round(stat, 4), "p_value": round(p, 6)}


def format_metrics(m: Dict[str, float], ci: Dict[str, Tuple[float, float]] | None = None) -> str:
    lines = [
        f"  Videos: {m['n_videos']}  | TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']}",
        f"  Fall recall    : {m['fall_recall']:.4f}"
        + (f"  CI95=[{ci['fall_recall'][0]:.3f}, {ci['fall_recall'][1]:.3f}]" if ci else ""),
        f"  Fall precision : {m['fall_precision']:.4f}"
        + (f"  CI95=[{ci['fall_precision'][0]:.3f}, {ci['fall_precision'][1]:.3f}]" if ci else ""),
        f"  Fall F1        : {m['fall_f1']:.4f}"
        + (f"  CI95=[{ci['fall_f1'][0]:.3f}, {ci['fall_f1'][1]:.3f}]" if ci else ""),
        f"  Macro F1       : {m['macro_f1']:.4f}"
        + (f"  CI95=[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]" if ci else ""),
        f"  Accuracy       : {m['accuracy']:.4f}"
        + (f"  CI95=[{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]" if ci else ""),
    ]
    return "\n".join(lines)
