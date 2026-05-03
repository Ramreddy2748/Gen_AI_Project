# Fall Detection Data Pipeline

## Dataset Summary

| Dataset | Source | Fall Videos | No-Fall Videos | Total |
|---------|--------|-------------|----------------|-------|
| **URFD** | University of Rochester | 30 | 40 | 70 |
| **Le2i** | Lab Electronics & Image | 130 | 59* | 189 |
| **GMNCSA24** | GitHub/HuggingFace | 79 | 81 | 160 |
| **Total** | | **239** | **180** | **419** |

*Some Le2i videos are corrupted (moov atom not found)

## Processed Data

| Metric | Value |
|--------|-------|
| Videos processed | 419 |
| Total windows (before balance) | 4,665 |
| **Balanced windows** | **1,882** |
| Fall windows | 941 |
| No-Fall windows | 941 |
| **Ratio** | **1:1** |

## Data Splits

| Split | Percentage | Windows |
|-------|------------|---------|
| Train | 70% | ~1,317 |
| Val | 15% | ~282 |
| Test | 15% | ~283 |

## Features Per Window (3 frames)

### Position Features
- `hip_y`, `shoulder_y`, `nose_y` - Normalized coordinates (0-1)
- `body_center_y` - Center of body mass
- `torso_vertical_diff` - Shoulder-hip distance
- `body_angle_degrees` - 90°=upright, 0°=horizontal

### Motion Features
- `velocity_hip_y`, `velocity_shoulder_y`, `velocity_magnitude`
- `acceleration_hip_y`, `acceleration_magnitude`
- `jerk` - Rate of acceleration change

### Posture Classification
- `posture_state`: "upright" | "transitioning" | "fallen"

### Fall Indicator Flags
- `is_rapid_descent` - Fast downward movement
- `is_on_ground` - Body near ground level
- `is_horizontal` - Body angle < 30°
- `is_high_acceleration` - Sudden movement change

### Window Analysis
- `fall_likelihood_score` - 0.0 to 1.0
- `trajectory_direction` - "descending" | "stable" | "ascending"
- `has_velocity_spike` - Rapid movement detected

## Directory Structure

```
final_pipeline/
├── data/                    # Processed data
│   ├── windows/             # All sliding windows
│   ├── windows_balanced/    # Balanced 1:1 dataset
│   ├── poses/               # Pose features per video
│   ├── frames/              # Extracted frames
│   └── manifests/           # CSV manifests
├── scripts/                 # Pipeline scripts
│   └── balanced_pipeline.py # Main pipeline script
└── README.md
```

## Usage

### Run Pipeline
```bash
python scripts/balanced_pipeline.py
```

### Load Data for GenAI
```python
import json
from pathlib import Path

# Load a window
window_path = Path("data/windows_balanced/train/fall/video_id/video_id_w0000.json")
with open(window_path) as f:
    window = json.load(f)

# Access features
sequence = window["sequence"]  # frame_1, frame_2, frame_3
analysis = window["window_analysis"]
llm_desc = window["llm_descriptions"]
```

## Feature Discrimination

| Label | Avg Fall Likelihood |
|-------|---------------------|
| Fall | 0.183 |
| No-Fall | 0.037 |
| **Ratio** | **5x** |
