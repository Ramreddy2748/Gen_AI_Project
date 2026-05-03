# DATA 266 GenAI Fall Detection

This repository contains a fall-detection data pipeline for DATA 266. It prepares local video datasets, samples frames, extracts MediaPipe pose features, and builds short pose-window records that can be passed into later LLM, RAG, or fine-tuning experiments.

The current pipeline focuses on structured pre-LLM artifacts:

- video manifests and train/validation/test splits
- sampled frame manifests
- per-video pose JSON files
- sliding-window JSONL/CSV inputs for multimodal fall reasoning
- one-shot processing for an uploaded/local video

## Project Layout

```text
.
├── data/
│   ├── raw/                          # local raw datasets, ignored by Git
│   │   ├── urfd_videos/
│   │   └── voxel_huggingface_data/
│   ├── manifests/                    # compact dataset/video manifests
│   ├── splits/
│   │   └── videos/                   # symlinked train/valid/test video folders
│   ├── processed/
│   │   ├── frames/                   # sampled frame manifests; JPG frames are ignored
│   │   ├── poses/                    # per-video pose JSON files and pose manifests
│   │   ├── windows/                  # pre-LLM sliding-window JSONL/CSV files
│   │   ├── window_poses/             # one JSON per pose window
│   │   └── window_poses_readable/    # readable window-pose export layout
│   └── uploads/
│       └── windows/                  # single uploaded-video window outputs
├── pose_extraction/                  # MediaPipe pose extraction scripts
├── scripts/
│   ├── dataset_setup/                # dataset download, verification, manifest, split utilities
│   ├── frame_extraction/             # video/frame sampling utilities
│   └── window_generation/            # pose-window generation and uploaded-video pipeline
├── pose_landmarker_lite.task         # MediaPipe pose landmarker model, if available locally
├── requirements.txt
└── PROJECT_ROADMAP.md
```

Large raw datasets, video split folders, frame JPGs, runtime outputs, caches, and virtual environments are intentionally ignored by Git.

## Setup

Use Python 3.11 if possible, because the local environment and generated cache names in this project have mostly used `.venv311`.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some MediaPipe scripts expect the model file at the repo root:

```text
pose_landmarker_lite.task
```

If the model is missing, download the MediaPipe Pose Landmarker Lite task file and place it at that path.

For local matplotlib/MediaPipe runs on macOS, this repo uses `.mplconfig`:

```bash
export MPLCONFIGDIR=.mplconfig
```

## End-To-End Data Pipeline

The typical training-data flow is:

1. Build a video manifest.
2. Materialize train/valid/test video folders.
3. Sample frames from a split.
4. Extract pose landmarks from sampled frames.
5. Build sliding pose windows for later LLM/RAG/fine-tuning work.

### 1. Create a Video Manifest

```bash
python scripts/dataset_setup/create_video_manifest.py \
  --output data/manifests/video_manifest.csv \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42
```

This scans local video roots such as:

- `data/raw/voxel_huggingface_data/`
- `data/raw/urfd_videos/`

The manifest includes video id, dataset source, label, target, group id, camera, source path, split, and split strategy.

### 2. Materialize Video Splits

```bash
python scripts/dataset_setup/materialize_video_splits.py \
  --manifest data/manifests/video_manifest.csv \
  --output-root data/splits/videos \
  --mode symlink
```

Use `--mode symlink` to avoid duplicating videos. Use `--mode copy` only when you need physically copied split folders.

### 3. Sample Frames

```bash
python scripts/frame_extraction/extract_split_frames.py \
  --manifest data/manifests/video_manifest.csv \
  --split-root data/splits/videos \
  --output-root data/processed/frames \
  --split train \
  --sample-fps 3
```

Output:

- `data/processed/frames/train_frame_manifest.csv`
- `data/processed/frames/train/<label>/<video_id>/frame_*.jpg`

Frame JPGs can be large and are ignored by Git. The manifest is safe to track.

### 4. Extract Pose Features

```bash
MPLCONFIGDIR=.mplconfig python pose_extraction/extract_split_pose_landmarks.py \
  --frames-root data/processed/frames \
  --output-root data/processed/poses \
  --split train
```

Output:

- `data/processed/poses/train_pose_manifest.csv`
- `data/processed/poses/train/fall/*.json`
- `data/processed/poses/train/no_fall/*.json`

Each pose JSON contains frame-level detection status and a compact set of landmark-derived features:

- hip position
- shoulder position
- nose position
- torso vertical difference

### 5. Build Pre-LLM Sliding Windows

```bash
python scripts/window_generation/build_pose_windows.py \
  --pose-manifest data/processed/poses/train_pose_manifest.csv \
  --frame-manifest data/processed/frames/train_frame_manifest.csv \
  --output-root data/processed/windows \
  --split train \
  --window-size 3 \
  --stride 2
```

Output:

- `data/processed/windows/train/train_window_manifest.csv`
- `data/processed/windows/train/train_llm_windows.jsonl`
- `data/processed/windows/train/train_window_summary.json`

Each JSONL record includes the video id, label, source video path, window timing, frame paths, timestamps, pose-detection flags, and per-frame pose features.

### 6. Export One JSON Per Window

```bash
python scripts/window_generation/extract_window_pose_splits.py \
  --window-jsonl data/processed/windows/train/train_llm_windows.jsonl \
  --output-root data/processed/window_poses_readable \
  --split train \
  --naming descriptive
```

Output:

- `data/processed/window_poses_readable/train/train_window_pose_manifest.csv`
- `data/processed/window_poses_readable/train/train_window_pose_summary.json`
- readable per-window JSON files grouped by label and source video

## Uploaded Video Pipeline

For one local or uploaded video, run frame sampling, pose extraction, and window generation in one command:

```bash
MPLCONFIGDIR=.mplconfig python scripts/window_generation/process_video_to_windows.py \
  --video path/to/video.mp4 \
  --output-root data/uploads/windows \
  --sample-fps 3 \
  --window-size 3 \
  --stride 2
```

Output is written to:

```text
data/uploads/windows/<video_id>/
├── frames/
├── pose_features.json
├── window_manifest.csv
├── llm_windows.jsonl
└── window_summary.json
```

## Useful Scripts

### Dataset Setup

- `scripts/dataset_setup/data.py` downloads and extracts source data.
- `scripts/dataset_setup/hf_data.py` downloads the Hugging Face video dataset.
- `scripts/dataset_setup/extract_zips.py` extracts remaining ZIP files.
- `scripts/dataset_setup/repair_falls_data.py` repairs missing or corrupt fall files.
- `scripts/dataset_setup/verify_falls_data.py` validates fall dataset files.
- `scripts/dataset_setup/create_video_manifest.py` creates grouped train/val/test metadata.
- `scripts/dataset_setup/materialize_video_splits.py` creates split folders by symlink or copy.

### Frame Extraction

- `scripts/frame_extraction/extract_split_frames.py` samples frames from split videos.
- `scripts/frame_extraction/sample_urfd_frames.py` samples URFD frame sequences.
- `scripts/frame_extraction/extract_useful_frames.py` selects useful frames from video data.
- `scripts/frame_extraction/video_llm_pose_pipeline.py` runs an older end-to-end frame/pose pipeline.

### Pose Extraction

- `pose_extraction/extract_split_pose_landmarks.py` extracts pose JSON from sampled frame split folders.
- `pose_extraction/extract_urfd_pose.py` extracts landmarks from URFD sampled frames.
- `pose_extraction/sample_video_frames_and_landmarks.py` processes one video into sampled frames and landmarks.
- `pose_extraction/batch_video_landmarks.py` processes all videos in a folder.
- `pose_extraction/pose_features.py` contains pose feature extraction helpers.

### Window Generation

- `scripts/window_generation/build_pose_windows.py` builds pre-LLM sliding-window JSONL and CSV files.
- `scripts/window_generation/extract_window_pose_splits.py` materializes one JSON file per window.
- `scripts/window_generation/process_video_to_windows.py` processes a single uploaded/local video end to end.

## Tracked vs. Local Artifacts

Tracked or intended to be trackable:

- source scripts
- manifests
- compact pose/window JSON and CSV artifacts
- project documentation

Ignored by design:

- `.venv/`, `.venv311/`, caches, and `__pycache__/`
- raw downloaded datasets
- split video folders
- sampled frame JPG directories
- runtime outputs such as `results/`
- large re-downloadable model files, except `pose_landmarker_lite.task` if explicitly needed

Before pushing, inspect the exact commit scope:

```bash
git status --short
git diff --cached --stat
git diff --cached --name-status
```

## Current Project Direction

The prepared window records are the bridge into the GenAI part of the project. They are designed to support:

- zero-shot and chain-of-thought prompting over selected frames
- retrieval-augmented prompting using similar annotated pose/window cases
- LoRA or other fine-tuning experiments using structured frame/pose context
- evaluation with accuracy, precision, recall, F1, and false-positive rate

See `PROJECT_ROADMAP.md` for the broader course plan and milestone breakdown.
