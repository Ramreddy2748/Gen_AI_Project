"""
Thin OpenAI wrapper that returns both a label and a calibrated confidence.

Confidence is derived from the token logprobs of the response. For each call
we request `top_logprobs=5` and read the probability of the first token that
starts with "FALL" vs "NO_FALL".
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def make_client(api_key: Optional[str] = None):
    if OpenAI is None:
        raise RuntimeError("Install the OpenAI SDK first: pip install openai")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _parse(text: str) -> str:
    t = (text or "").upper().strip()
    if "NO_FALL" in t or "NO FALL" in t or "NOT FALL" in t:
        return "no_fall"
    if "FALL" in t:
        return "fall"
    return "unknown"


def call_with_confidence(
    client,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 12,
    retries: int = 3,
) -> Tuple[str, float, str]:
    """
    Return (label, confidence_in_fall, raw_response).

    Confidence is derived from the first-token logprobs:
        p(FALL token) / (p(FALL) + p(NO_FALL))
    If logprobs aren't returned for either token, we fall back to a hard 0/1.
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
            )
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            label = _parse(text)

            fall_p, nofall_p = 0.0, 0.0
            try:
                # First-token logprobs.
                first_tok = choice.logprobs.content[0]
                for tlp in first_tok.top_logprobs:
                    tok_up = (tlp.token or "").upper().strip()
                    p = math.exp(tlp.logprob)
                    if tok_up.startswith("NO") or "NO_FALL" in tok_up or "NO FALL" in tok_up:
                        nofall_p += p
                    elif tok_up.startswith("FALL"):
                        fall_p += p
            except Exception:
                pass

            if fall_p + nofall_p > 0:
                conf = fall_p / (fall_p + nofall_p)
            else:
                conf = 1.0 if label == "fall" else 0.0
            return label, conf, text
        except Exception as e:  # pragma: no cover
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "unknown", 0.0, f"ERROR: {e}"
    return "unknown", 0.0, "ERROR: retries exhausted"
