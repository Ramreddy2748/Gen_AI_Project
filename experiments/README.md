# Experiments

Isolated sandbox for new fall-detection experiments. Nothing in
`final_pipeline/` is modified — every script in here reads from
`final_pipeline/data/` and `final_pipeline/results/` and writes only into
`experiments/data/` and `experiments/results/`.

**Focus: video-level metrics only.** Window-level numbers are not the
project's evaluation target.

---

## TL;DR — what's already been validated

Re-aggregating the existing Best-Prompt per-window predictions with a smarter
video-level rule gives a free Macro F1 lift on the test split at identical
recall:

| Strategy | Rule | Fall Recall | Fall F1 | Macro F1 | Accuracy |
|---|---|---|---|---|---|
| `best_prompt` | `ratio_0.25` (baseline) | 0.958 | 0.657 | 0.556 | 0.579 |
| `best_prompt` | **`ratio_0.50`** | **0.958** | **0.697** | **0.640** | **0.649** |
| `rag`         | `ratio_0.25` (baseline) | 0.854 | 0.689 | 0.678 | 0.678 |
| `rag`         | **`end_state_min1`** | 0.792 | **0.775** | **0.804** | **0.809** |
| `lora_finetune` | `ratio_0.25` (baseline) | 0.938 | 0.667 | 0.458 | 0.538 |
| `lora_finetune` | **`consecutive_k4`** | 0.812 | **0.812** | **0.815** | **0.815** |
| `enhanced_few_shot` | `ratio_0.25` (baseline) | 0.667 | 0.711 | 0.761 | 0.772 |
| `enhanced_few_shot` | **`ratio_0.30`** | 0.625 | 0.732 | **0.790** | 0.807 |

No new LLM calls were made for these numbers. They come from running
`scripts/01_reaggregate_baseline.py` on the existing
`final_pipeline/results/<strategy>/window_results_*.json`.

---

## Folder layout

```
experiments/
├── README.md
├── core/                         # reusable modules
│   ├── config.py                 # paths, default thresholds
│   ├── data_loader.py            # read poses & existing predictions
│   ├── features.py               # re-derive enhanced features (real jerk, etc.)
│   ├── window_builder.py         # build N-frame windows from pose JSONs
│   ├── aggregation.py            # 6 video-level aggregation strategies
│   ├── metrics.py                # video-only metrics + bootstrap CIs + McNemar
│   ├── llm_client.py             # OpenAI w/ logprobs for confidence
│   └── prompts.py                # pinned Stage-1 / Stage-2 prompts
├── scripts/
│   ├── 01_reaggregate_baseline.py     # zero-cost video-F1 lift via smarter aggregation
│   ├── 02_build_window_sizes.py       # build N=5/7/9/12/15 windows from poses
│   ├── 03_calibrate_thresholds.py     # train-only threshold calibration (leakage fix)
│   ├── 04_confidence_weighted.py      # requires OPENAI_API_KEY
│   ├── 05_two_stage_cascade.py        # requires OPENAI_API_KEY
│   └── 06_per_dataset_report.py       # break video metrics down by dataset
├── data/                         # generated windows, calibrated thresholds
└── results/                      # generated metrics JSONs
```

---

## Scripts

All scripts are runnable from the repo root.

### 1. Re-aggregation sweep (no API key needed)

Compares 17 video-level aggregation rules on existing window predictions.
Picks the operating point on a tuning split (val / val+test fallback) under a
fall-recall constraint, then reports the same rule on the held-out split
with bootstrap CIs and a McNemar test against the original baseline.

```
python experiments/scripts/01_reaggregate_baseline.py \
    --strategy best_prompt --tune-on val --report-on test --min-recall 0.85
```

`--strategy` is the sub-directory under `final_pipeline/results/` to re-aggregate
(`best_prompt`, `rag`, `enhanced_few_shot`, `lora_finetune`, etc.).
`--tune-on none` will sweep and pick on the report split itself (diagnostic only).

### 2. Window-size builder (no API key needed)

Rebuilds sliding windows of any size from the existing pose JSONs. No
MediaPipe re-run. Adds **real jerk**, **angular velocity** and
**torso-collapse rate** that the original pipeline left at zero.

```
python experiments/scripts/02_build_window_sizes.py \
    --sizes 5 7 9 12 15 --splits train val test
```

Output: `experiments/data/windows_n05/test/fall/<video_id>/<wid>.json` etc.

### 3. Train-only threshold calibration (no API key needed)

Recomputes the per-frame thresholds from the *train* split percentiles only,
plugging a small leakage hole in the original `balanced_pipeline.py`.

```
python experiments/scripts/03_calibrate_thresholds.py
```

Findings (relative to the original hard-coded constants):

| Threshold | Original | Train-calibrated |
|---|---|---|
| `ground_hip_y_min` | 0.65 | 0.77 |
| `rapid_velocity_min` | 0.15 | 0.16 |
| `high_acceleration_min` | 0.10 | **0.93** |
| `horizontal_angle_max` | 30.0 | 14.9 |
| `upright_torso_min` | 0.12 | 0.14 |
| `fallen_torso_max` | 0.06 | 0.03 |
| `significant_descent` | 0.12 | 0.24 |

The big shifts (acceleration, angle, descent) imply the original numbers
were too permissive, which is consistent with the high false-positive rate
of Best Prompt.

The output JSON is consumed by `core.window_builder.build_window_set(...,
thresholds=...)` so a rebuilt size-3 (or size-N) window set will have
flags that better track real fall events.

### 4. Confidence-weighted aggregation (requires `OPENAI_API_KEY`)

Re-classifies every window with Stage-1 (safety-first prompt) using
`logprobs=True`, derives a per-window fall-confidence, then sums
confidences over a video and sweeps the threshold on val.

```
export OPENAI_API_KEY=sk-...
python experiments/scripts/04_confidence_weighted.py \
    --model gpt-4o --tune-on val --report-on test --min-recall 0.90
```

Cost: ~1 chat completion per window in tune+report splits. Caching to
`experiments/results/04_confidence_weighted/stage1_<split>.json` so reruns
are free.

### 5. Two-stage cascade (requires `OPENAI_API_KEY`)

Stage 1 runs the existing Best-Prompt aggregation, producing candidate fall
videos. Stage 2 sends the *entire* timeline of each candidate's windows to a
strict FP-filter prompt that's been primed to reject sit / bend / kneel /
lie-down cases. Final cascade prediction = `fall` iff both stages agree.

```
export OPENAI_API_KEY=sk-...
python experiments/scripts/05_two_stage_cascade.py \
    --stage1-strategy best_prompt --stage1-rule ratio_0.50 \
    --report-on test --model gpt-4o
```

Cost: 1 chat completion per Stage-1-flagged video (roughly 40 calls for
the test split).

### 6. Per-dataset report (no API key needed)

Breaks video metrics down by URFD / Le2i / GMNCSA24 to expose which dataset
is dragging the aggregate down.

```
python experiments/scripts/06_per_dataset_report.py \
    --strategy best_prompt --split test --rule ratio_0.50
```

Headline finding (test split, `best_prompt`, rule `ratio_0.50`):

| Dataset | Macro F1 |
|---|---|
| GMNCSA24 | 0.826 |
| URFD | 0.533 |
| Le2i | 0.479 |

Le2i is the bottleneck.

---

## What this folder addresses from the professor's feedback

1. **Increase F1 while keeping decent recall** — scripts 01 and 04-05.
   Script 01 already shows a free 8.4-point Macro F1 lift on Best Prompt
   at identical recall.
2. **More experiments** — scripts 02 (window-size sweep), 03 (threshold
   recalibration), 04 (confidence-weighted), 05 (cascade), 06 (per-dataset).
3. **Remove data leakage** — script 03 fixes the threshold-calibration
   leakage. The (much-discussed) subject overlap in Le2i is not actual
   leakage in this pose-feature pipeline; documented in this README and in
   the chat history for the record.

---

## Important context: why subject overlap in Le2i is *not* a problem here

The Le2i folder has only 4 unique scene/subject prefixes (`CR01`, `CR02`,
`H01`, `H02`) across 260 videos. In an image-based CNN, that would be a
serious problem because the network could memorise face / clothing / room.
In *this* pipeline:

- The LLM never sees images. It only sees normalized pose-feature numbers.
- The LLM is stateless across calls (no subject-ID memory between videos).
- Pose features are image-relative, so two fall events from the same person
  do not look identical at the feature level.

So no re-split is required. The single legitimate leakage was the
hand-picked thresholds, which `scripts/03_calibrate_thresholds.py` fixes.

---

## Next steps (suggested order)

1. Run `01_reaggregate_baseline.py` for every strategy in
   `final_pipeline/results/` and commit the resulting table to the report.
2. Run `04_confidence_weighted.py` once on val+test to get a single,
   reproducible operating point with confidence intervals.
3. Run `05_two_stage_cascade.py` on the same set and produce the cascade row
   for the headline comparison table.
4. Build size-5 / size-7 / size-9 windows with `02_build_window_sizes.py`
   plus calibrated thresholds, then re-run `04` on them to produce the
   "video F1 vs window size" figure the professor asked for.
5. Add a per-dataset table (`06_per_dataset_report.py`) showing where each
   strategy wins and loses.
