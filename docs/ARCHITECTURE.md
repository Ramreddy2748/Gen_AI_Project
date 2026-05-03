# System Architecture

## Overview

The GenAI Fall Detection system uses a multi-stage pipeline that combines computer vision (pose estimation) with Large Language Models (LLMs) for temporal reasoning.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│   │   URFD     │    │   Le2i     │    │  GMNCSA24  │    │  Custom    │     │
│   │  Dataset   │    │  Dataset   │    │  Dataset   │    │  Videos    │     │
│   └─────┬──────┘    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘     │
│         │                 │                 │                 │             │
│         └─────────────────┴─────────────────┴─────────────────┘             │
│                                   │                                          │
│                                   ▼                                          │
│                        ┌──────────────────┐                                  │
│                        │  Video Manifest  │                                  │
│                        │   (419 videos)   │                                  │
│                        └────────┬─────────┘                                  │
│                                 │                                            │
└─────────────────────────────────┼────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Frame Extraction                               │   │
│   │  • Fall videos: 5 FPS (capture fall dynamics)                       │   │
│   │  • No-fall videos: 15 FPS (normal activity)                         │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     MediaPipe Pose Estimation                        │   │
│   │  • 33 body landmarks                                                 │   │
│   │  • Normalized coordinates (0-1)                                      │   │
│   │  • Confidence scores                                                 │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Enhanced Feature Engineering                      │   │
│   │                                                                      │   │
│   │  Position:     hip_y, shoulder_y, nose_y, body_center_y             │   │
│   │  Orientation:  body_angle_degrees, torso_vertical_diff              │   │
│   │  Motion:       velocity_hip_y, velocity_shoulder_y, acceleration    │   │
│   │  Posture:      upright / transitioning / fallen                     │   │
│   │  Flags:        is_rapid_descent, is_on_ground, is_horizontal        │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SLIDING WINDOW LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Temporal Window Creation                          │   │
│   │                                                                      │   │
│   │   Window Size: 3 consecutive frames                                  │   │
│   │   Fall Stride: 1 (overlap for dense coverage)                       │   │
│   │   No-Fall Stride: 3 (reduce redundancy)                             │   │
│   │                                                                      │   │
│   │   ┌─────┬─────┬─────┐                                               │   │
│   │   │  F1 │  F2 │  F3 │  → Window Analysis                            │   │
│   │   └─────┴─────┴─────┘                                               │   │
│   │                                                                      │   │
│   │   Window Features:                                                   │   │
│   │   • trajectory_direction                                            │   │
│   │   • max_velocity                                                    │   │
│   │   • has_velocity_spike                                              │   │
│   │   • fall_likelihood_score                                           │   │
│   └───────────────────────────────┬─────────────────────────────────────┘   │
│                                   │                                          │
│                          1,882 total windows                                 │
│                       (941 fall + 941 no-fall)                               │
│                                                                              │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GENAI CLASSIFICATION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│   │   Zero-Shot     │ │   Few-Shot      │ │      CoT        │              │
│   │   Prompting     │ │   Prompting     │ │   Reasoning     │              │
│   │                 │ │                 │ │                 │              │
│   │  No examples    │ │  6 examples     │ │  Step-by-step   │              │
│   │  Direct query   │ │  from training  │ │  analysis       │              │
│   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘              │
│            │                   │                   │                        │
│   ┌────────┴───────────────────┴───────────────────┴────────┐              │
│   │                                                          │              │
│   │   ┌─────────────────┐ ┌─────────────────┐               │              │
│   │   │ Self-Consistency│ │   RAG Pipeline  │               │              │
│   │   │                 │ │                 │               │              │
│   │   │  5 samples      │ │  Semantic       │               │              │
│   │   │  Majority vote  │ │  retrieval      │               │              │
│   │   │                 │ │  Top-4 similar  │               │              │
│   │   └────────┬────────┘ └────────┬────────┘               │              │
│   │            │                   │                        │              │
│   └────────────┴───────────────────┴────────────────────────┘              │
│                            │                                                │
│                            ▼                                                │
│              ┌─────────────────────────────┐                               │
│              │    Best Prompt (Safety)     │                               │
│              │                             │                               │
│              │  • Safety-critical framing  │                               │
│              │  • Strong fall indicators   │                               │
│              │  • 25% threshold            │                               │
│              │  • 97.9% fall recall        │                               │
│              └──────────────┬──────────────┘                               │
│                             │                                               │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGGREGATION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Window Predictions → Video-Level Decision                                  │
│                                                                              │
│   Strategies:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • Majority (>50%): Standard voting                                 │   │
│   │  • Any: Single fall window = fall video                             │   │
│   │  • Threshold 25%: ≥25% fall windows = fall (RECOMMENDED)           │   │
│   │  • Consecutive: 2+ consecutive falls = fall                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│                    ┌──────────────────┐                                     │
│                    │   FALL / NO_FALL │                                     │
│                    │    + Confidence  │                                     │
│                    │    + Explanation │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion
- **Input**: Video files (MP4, AVI)
- **Output**: Video manifest with labels
- **Datasets**: URFD, Le2i, GMNCSA24

### 2. Feature Extraction
- **Pose Estimation**: MediaPipe Pose (33 landmarks)
- **Feature Types**: Position, Orientation, Motion, Posture, Flags
- **Frame Rate**: Adaptive (5 FPS fall, 15 FPS no-fall)

### 3. Sliding Window
- **Window Size**: 3 frames
- **Stride**: 1 (fall), 3 (no-fall)
- **Balance**: 1:1 fall/no-fall ratio

### 4. GenAI Classification
- **Models**: GPT-4o, GPT-4o-mini
- **Strategies**: Zero-Shot, Few-Shot, CoT, Self-Consistency, RAG
- **Best**: Safety-First prompt (97.9% recall)

### 5. RAG Pipeline
```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Architecture                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  OFFLINE (Build Knowledge Base)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Training Windows → Text → Embeddings → Vector DB   │    │
│  │  (1,305 windows)   (text-embedding-3-small)         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ONLINE (Inference)                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Query → Embed → Retrieve Top-4 → RAG Prompt → LLM  │    │
│  │                  (2 fall + 2 no-fall)               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6. Aggregation
- Multiple strategies for window → video aggregation
- **Recommended**: 25% threshold for safety applications

## Data Flow

```
Video → Frames → Poses → Features → Windows → Prompt → LLM → Prediction
  │        │        │        │         │         │       │        │
  └────────┴────────┴────────┴─────────┴─────────┴───────┴────────┘
                              │
                        All stored as JSON
                        for reproducibility
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Video Processing | OpenCV |
| Pose Estimation | MediaPipe |
| Feature Storage | JSON |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Language | Python 3.8+ |

## Scalability Considerations

1. **Batch Processing**: Process multiple videos in parallel
2. **Caching**: Store embeddings for RAG knowledge base
3. **API Rate Limiting**: Retry logic with exponential backoff
4. **Memory**: Stream frames instead of loading entire video

## Future Enhancements

1. **Real-time Processing**: Stream video analysis
2. **Edge Deployment**: On-device pose estimation
3. **Multi-person Detection**: Track multiple subjects
4. **Alert System**: Integrate with notification services
