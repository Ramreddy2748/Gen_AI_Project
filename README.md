# GenAI-Based Multimodal Fall Detection using Sliding Window Temporal Reasoning

A comprehensive fall detection system leveraging Generative AI with multiple prompting strategies, RAG (Retrieval-Augmented Generation), and fine-tuning techniques for safety-critical elderly care applications.

## Project Overview

This project implements a multimodal fall detection pipeline that:
1. **Extracts pose features** from video frames using MediaPipe
2. **Creates sliding windows** of temporal pose sequences
3. **Classifies fall events** using various GenAI prompting strategies
4. **Achieves 97.9% fall recall** for safety-critical applications

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fall Detection Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │  Video   │───►│    Pose      │───►│   Sliding   │───►│   GenAI    │ │
│  │  Input   │    │  Extraction  │    │   Windows   │    │ Classifier │ │
│  │ (URFD,   │    │ (MediaPipe)  │    │ (3 frames)  │    │            │ │
│  │  Le2i,   │    │              │    │             │    │ - Zero-Shot│ │
│  │ GMNCSA)  │    │              │    │             │    │ - Few-Shot │ │
│  └──────────┘    └──────────────┘    └─────────────┘    │ - CoT      │ │
│                                                          │ - RAG      │ │
│                                                          │ - LoRA     │ │
│                                                          └────────────┘ │
│                                                                 │        │
│                                                                 ▼        │
│                                                          ┌────────────┐ │
│                                                          │   Fall /   │ │
│                                                          │  No-Fall   │ │
│                                                          │ Prediction │ │
│                                                          └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Gen_AI_Project/
├── README.md                      # This file
├── PROJECT_ROADMAP.md             # Detailed project roadmap
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── docs/                          # Documentation
│   ├── PROJECT_PROPOSAL.md        # Project proposal
│   ├── ARCHITECTURE.md            # System architecture details
│   └── EVALUATION_RESULTS.md      # Detailed results
│
├── final_pipeline/                # Main pipeline (recommended)
│   ├── README.md                  # Pipeline documentation
│   ├── scripts/                   # Data processing scripts
│   │   ├── balanced_pipeline.py   # Main data pipeline
│   │   └── download_gmncsa24_github.py
│   │
│   ├── prompting/                 # GenAI prompting strategies
│   │   ├── zero_shot.py           # Zero-shot baseline
│   │   ├── few_shot.py            # Few-shot learning
│   │   ├── chain_of_thought.py    # CoT reasoning
│   │   ├── self_consistency.py    # Self-consistency
│   │   ├── enhanced_few_shot.py   # Enhanced with GPT-4o
│   │   ├── best_prompt.py         # Safety-first prompt
│   │   ├── rag_fall_detection.py  # RAG pipeline
│   │   └── lora_finetune.py       # LoRA fine-tuning
│   │
│   ├── data/                      # Processed data
│   │   ├── windows/               # Sliding window JSONs
│   │   ├── windows_balanced/      # Balanced dataset
│   │   ├── metadata/              # Data splits & labels
│   │   └── manifests/             # Video manifests
│   │
│   └── results/                   # Evaluation results
│       ├── FINAL_REPORT.md        # Comprehensive report
│       ├── zero_shot/             # Zero-shot results
│       ├── few_shot/              # Few-shot results
│       ├── cot/                   # CoT results
│       └── rag/                   # RAG results
│
├── pose_extraction/               # Pose feature extraction
│   ├── pose_features.py           # Feature definitions
│   └── extract_*.py               # Extraction scripts
│
└── scripts/                       # Legacy scripts
    ├── dataset_setup/             # Dataset preparation
    ├── frame_extraction/          # Frame extraction
    └── window_generation/         # Window generation
```

## Datasets

| Dataset | Videos | Description |
|---------|--------|-------------|
| URFD | 70 | University of Rochester Fall Detection |
| Le2i | 189 | Laboratoire Electronique, Informatique et Image |
| GMNCSA24 | 160 | Fall detection from GitHub/HuggingFace |
| **Total** | **419** | Combined multimodal dataset |

### Dataset Download Links

**Note:** Datasets are not included in this repository due to size (1.1GB). Please download from the original sources:

1. **URFD (UR Fall Detection)**
   - Download: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
   - Place in: `Dataset/URFD/`

2. **Le2i**
   - Download: http://le2i.cnrs.fr/Fall-detection-Dataset
   - Place in: `Dataset/Le2i/`

3. **GMNCSA24**
   - Download: https://huggingface.co/datasets/GMNCSA/Fall-Detection
   - Or use: `python final_pipeline/scripts/download_gmncsa24_github.py`
   - Place in: `Dataset/GMNCSA24/`

### Directory Structure After Download
```
Dataset/
├── URFD/
│   ├── Fall/           # 30 fall videos
│   └── ADL/            # 40 ADL videos
├── Le2i/
│   ├── Coffee_room/    # Fall & ADL videos
│   ├── Home_01/
│   └── Home_02/
└── GMNCSA24/
    ├── fall/           # 80 fall videos
    └── adl/            # 80 ADL videos
```

### Data Splits

| Split | Fall Videos | No-Fall Videos | Total | Windows |
|-------|-------------|----------------|-------|---------|
| Train | 111 | 149 | 260 | 1,305 |
| Val | 24 | 34 | 58 | 305 |
| Test | 24 | 33 | 57 | 272 |
| **Total** | **159** | **216** | **375** | **1,882** |

## Features Extracted

### Per-Frame Features (MediaPipe Pose)
- **Position**: hip_y, shoulder_y, nose_y, body_center_y
- **Orientation**: body_angle_degrees, torso_vertical_diff
- **Velocity**: velocity_hip_y, velocity_shoulder_y, velocity_magnitude
- **Acceleration**: acceleration_hip_y, acceleration_magnitude, jerk
- **Posture**: posture_state (upright/transitioning/fallen)
- **Flags**: is_rapid_descent, is_on_ground, is_horizontal

### Window-Level Features
- trajectory_direction (ascending/stable/descending)
- max_velocity, has_velocity_spike
- fall_likelihood_score

## GenAI Prompting Strategies

### 1. Zero-Shot Prompting
- Direct classification without examples
- Baseline performance

### 2. Few-Shot Prompting
- 3 fall + 3 no-fall examples from training set
- Improved contextual understanding

### 3. Chain-of-Thought (CoT)
- Step-by-step reasoning
- Position → Motion → Posture → Decision

### 4. Self-Consistency
- 5 samples per window with temperature=0.7
- Majority voting for robustness

### 5. Enhanced Few-Shot (GPT-4o)
- Rich textual descriptions instead of raw numbers
- More capable model

### 6. Best Prompt (Safety-First)
- Optimized for fall recall
- 25% threshold aggregation
- **97.9% video-level fall recall**

### 7. RAG (Retrieval-Augmented Generation)
- 1,305 training windows embedded
- Semantic similarity retrieval
- Top-4 similar examples as context
- **85.4% recall with explainability**

## Results Summary

| Strategy | Video Recall | Video Accuracy | Key Advantage |
|----------|--------------|----------------|---------------|
| Zero-Shot | 20.8% | 66.7% | Baseline |
| Few-Shot | 58.3% | 66.7% | Simple |
| CoT | 29.2% | 66.7% | Interpretable |
| Self-Consistency | 41.7% | 70.2% | Robust |
| Enhanced (GPT-4o) | 41.7% | **75.4%** | Highest accuracy |
| **RAG** | **85.4%** | 67.8% | Explainable |
| **Best Prompt** | **97.9%** | 58.3% | Highest recall |

## Installation

```bash
# Clone the repository
git clone https://github.com/Ramreddy2748/Gen_AI_Project.git
cd Gen_AI_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

## Usage

### 1. Data Pipeline (if starting fresh)
```bash
cd final_pipeline/scripts
python balanced_pipeline.py
```

### 2. Run Prompting Strategies
```bash
cd final_pipeline/prompting

# Zero-shot baseline
python zero_shot.py

# Few-shot
python few_shot.py

# Chain-of-Thought
python chain_of_thought.py

# Self-Consistency
python self_consistency.py

# Best Prompt (Safety-First)
python best_prompt.py --splits val test

# RAG Pipeline
python rag_fall_detection.py --splits val test
```

### 3. LoRA Fine-Tuning
```bash
python lora_finetune.py --mode prepare   # Prepare data
python lora_finetune.py --mode train     # Start training
python lora_finetune.py --mode evaluate  # Evaluate model
```

## Requirements

- Python 3.8+
- OpenAI API key (GPT-4o, GPT-4o-mini)
- MediaPipe
- OpenCV
- NumPy

See `requirements.txt` for full dependencies.

## Key Findings

1. **Safety-First Design**: Prioritizing recall over precision is crucial for elderly care
2. **97.9% Fall Recall**: Best Prompt strategy misses only 1 in 48 falls
3. **RAG Provides Explainability**: Similar cases explain decisions
4. **Rich Text > Raw Numbers**: Converting features to natural language improves LLM understanding
5. **Aggregation Matters**: Lower thresholds (25%) catch more falls at video level

## Project Team

- Term Project for GenAI Course

## License

This project is for educational purposes.

## Acknowledgments

- URFD, Le2i, and GMNCSA24 dataset creators
- OpenAI for GPT models
- Google MediaPipe for pose estimation
