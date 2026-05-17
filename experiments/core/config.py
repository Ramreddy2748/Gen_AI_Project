"""
Shared configuration for the experiments folder.

This folder is intentionally isolated from `final_pipeline/`. We never write
back into `final_pipeline/`; we only read its existing pose JSONs, window
JSONs, and result JSONs.
"""

from pathlib import Path

# Repository roots
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_PIPELINE = REPO_ROOT / "final_pipeline"

# Read-only sources (do not write here)
POSES_DIR = FINAL_PIPELINE / "data" / "poses"
WINDOWS_BALANCED_DIR = FINAL_PIPELINE / "data" / "windows_balanced"
WINDOWS_RAW_DIR = FINAL_PIPELINE / "data" / "windows"
RESULTS_DIR_FINAL = FINAL_PIPELINE / "results"

# Experiment outputs live here
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
EXP_DATA = EXPERIMENTS_ROOT / "data"
EXP_RESULTS = EXPERIMENTS_ROOT / "results"

# Frame-level thresholds copied from balanced_pipeline.py for reference.
# core/thresholds.py rewrites these from train-split percentiles.
DEFAULT_THRESHOLDS = {
    "upright_torso_min": 0.12,
    "fallen_torso_max": 0.06,
    "ground_hip_y_min": 0.65,
    "rapid_velocity_min": 0.15,
    "high_acceleration_min": 0.10,
    "significant_descent": 0.12,
    "horizontal_angle_max": 30.0,
}

LABELS = ("fall", "no_fall")
SPLITS = ("train", "val", "test")
DATASETS = ("urfd", "le2i", "gmncsa24")
