# GenAI Fall Detection - Comprehensive Evaluation Report

## Executive Summary

This report presents a complete evaluation of multiple GenAI prompting strategies for fall detection using sliding window temporal reasoning on pose estimation data.

**Key Achievement**: **97.9% video-level fall recall** with Best Prompt (Safety-First) strategy on Val+Test combined dataset.

---

## 1. Dataset Overview

### 1.1 Data Sources
| Dataset | Videos | Description |
|---------|--------|-------------|
| URFD | 70 | University of Rochester Fall Detection |
| Le2i | 189 | Laboratoire Electronique, Informatique et Image |
| GMNCSA24 | 160 | Fall detection dataset from GitHub |
| **Total** | **419** | Combined multimodal dataset |

### 1.2 Data Splits (After Balancing)
| Split | Fall Videos | No-Fall Videos | Total Videos | Fall Windows | No-Fall Windows | Total Windows |
|-------|-------------|----------------|--------------|--------------|-----------------|---------------|
| **Train** | 111 | 149 | 260 | 660 | 645 | 1,305 |
| **Val** | 24 | 34 | 58 | 163 | 142 | 305 |
| **Test** | 24 | 33 | 57 | 118 | 154 | 272 |
| **Total** | **159** | **216** | **375** | **941** | **941** | **1,882** |

### 1.3 Sliding Window Configuration
- **Window size**: 3 consecutive frames
- **Frame sampling**: 5 FPS for fall videos, 15 FPS for no-fall
- **Window stride**: 1 for fall videos, 3 for no-fall
- **Balance**: 1:1 ratio achieved via undersampling

---

## 2. Feature Extraction

### 2.1 Per-Frame Features (MediaPipe Pose)
| Category | Features |
|----------|----------|
| **Position** | hip_y, shoulder_y, nose_y, body_center_y |
| **Orientation** | body_angle_degrees, torso_vertical_diff |
| **Velocity** | velocity_hip_y, velocity_shoulder_y, velocity_magnitude |
| **Acceleration** | acceleration_hip_y, acceleration_magnitude, jerk |
| **Posture** | posture_state (upright/transitioning/fallen) |
| **Flags** | is_rapid_descent, is_on_ground, is_horizontal |

### 2.2 Window-Level Features
- trajectory_direction (ascending/stable/descending)
- max_velocity, has_velocity_spike
- fall_likelihood_score (0-1)

---

## 3. GenAI Techniques Implemented

### 3.1 Prompting Strategies (4 Required)
| # | Strategy | Description | Model |
|---|----------|-------------|-------|
| 1 | **Zero-Shot** | Direct classification without examples | GPT-4o-mini |
| 2 | **Few-Shot** | 6 examples (3 fall + 3 no-fall) from training | GPT-4o-mini |
| 3 | **Chain-of-Thought** | Step-by-step reasoning before decision | GPT-4o-mini |
| 4 | **Self-Consistency** | 5 samples with majority voting | GPT-4o-mini |

### 3.2 RAG Pipeline
| Component | Details |
|-----------|---------|
| **Knowledge Base** | 1,305 training windows embedded |
| **Embedding Model** | text-embedding-3-small |
| **Retrieval** | Top-4 similar (balanced 2 fall + 2 no-fall) |
| **Similarity** | Cosine similarity |
| **Classification** | GPT-4o with retrieved context |

### 3.3 Fine-Tuning (LoRA)
| Component | Details |
|-----------|---------|
| **Base Model** | gpt-4o-mini-2024-07-18 |
| **Training Data** | 1,305 windows from training set |
| **Status** | Validating files (OpenAI processing) |

### 3.4 Explainable AI (XAI)
| Component | Details |
|-----------|---------|
| **Feature Interpretation** | Converts pose data to human-readable text |
| **Reasoning** | Step-by-step LLM analysis |
| **Confidence** | HIGH/MEDIUM/LOW with explanation |
| **Output** | Full reasoning trace for each prediction |

---

## 4. Evaluation Results

### 4.1 Test Set Results (272 windows, 57 videos)

| Strategy | Window Acc | Window Recall | Video Acc | Video Recall |
|----------|------------|---------------|-----------|--------------|
| Zero-Shot | 65.4% | 22.9% | 66.7% | 20.8% |
| Few-Shot | 60.7% | 48.3% | 66.7% | 58.3% |
| Chain-of-Thought | 60.5% | 39.1% | 66.7% | 29.2% |
| Self-Consistency | 66.2% | 45.8% | 70.2% | 41.7% |
| Enhanced (GPT-4o) | 73.9% | 55.1% | 75.4% | 41.7% |
| **Best Prompt** | 67.3% | **92.4%** | 57.9% | **95.8%** |

### 4.2 Val+Test Combined Results (577 windows, 115 videos)

| Strategy | Window Acc | Window Recall | Video Acc | Video Recall |
|----------|------------|---------------|-----------|--------------|
| **RAG Pipeline** | 71.4% | 64.8% | 67.8% | **85.4%** |
| **Best Prompt** | ~68% | ~81% | ~58% | **97.9%** |

### 4.3 Best Prompt Confusion Matrix (Val+Test Video-Level)
```
                 Predicted
                 FALL    NO_FALL
  Actual FALL      47       1      (97.9% recall - only 1 miss!)
  Actual NO_FALL   47      20      (29.9% specificity)
```

### 4.4 RAG Pipeline Confusion Matrix (Val+Test Video-Level)
```
                 Predicted
                 FALL    NO_FALL
  Actual FALL      41       7      (85.4% recall)
  Actual NO_FALL   30      37      (55.2% specificity)
```

---

## 5. XAI Explanations

### 5.1 Sample Evaluation (50 windows)
| Metric | Value |
|--------|-------|
| Accuracy | 72.0% |
| Fall Recall | 48.0% |
| Confidence Distribution | High: 74%, Medium: 10%, Low: 16% |

### 5.2 Explanation Structure
```
============================================================
FALL DETECTION ANALYSIS
============================================================

### Frame-by-Frame Analysis ###
  Frame 1: Person is at MEDIUM height, Body is VERTICAL (upright)
  Frame 2: Person is LOW (near ground), Body is HORIZONTAL [ON_GROUND]
  Frame 3: Person is LOW, Body is HORIZONTAL, Posture: fallen

### FALL Indicators Detected ###
  [!] Frame 2: Low position (hip_y=0.78)
  [!] Frame 2: Horizontal body (angle=32°)
  [!] Frame 2: Person on ground
  [!] Frame 3: Rapid descent detected

### Decision Reasoning ###
  Fall indicators: 4
  Normal indicators: 1

  DECISION: FALL DETECTED
  Confidence: 90.0%
  Reason: Multiple strong fall indicators detected
============================================================
```

---

## 6. Technique Comparison

### 6.1 Accuracy vs Recall Trade-off
```
High Recall (Safety-Critical):
  Best Prompt  → 97.9% recall, 58.3% accuracy
  RAG Pipeline → 85.4% recall, 67.8% accuracy

High Accuracy:
  Enhanced GPT-4o → 75.4% accuracy, 41.7% recall

Balanced:
  RAG Pipeline → Best balance with explainability
```

### 6.2 Summary Comparison Table

| Strategy | Video Recall | Video Accuracy | Key Advantage |
|----------|--------------|----------------|---------------|
| Zero-Shot | 20.8% | 66.7% | Baseline, no examples needed |
| Few-Shot | 58.3% | 66.7% | Simple, uses examples |
| CoT | 29.2% | 66.7% | Interpretable reasoning |
| Self-Consistency | 41.7% | 70.2% | Robust via voting |
| Enhanced GPT-4o | 41.7% | **75.4%** | Highest accuracy |
| **RAG** | **85.4%** | 67.8% | Explainable + balanced |
| **Best Prompt** | **97.9%** | 58.3% | **Highest recall** |

---

## 7. RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Fall Detection                        │
├─────────────────────────────────────────────────────────────┤
│  KNOWLEDGE BASE (Training Data)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1,305 windows → Feature Text → Embeddings          │    │
│  │  Balanced: 660 fall + 645 no-fall examples          │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  INFERENCE                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Query Window → Embedding → Cosine Similarity       │    │
│  │  Retrieve Top-4 Similar (2 fall + 2 no-fall)        │    │
│  │  Build RAG Prompt with Retrieved Examples           │    │
│  │  GPT-4o Classification with Context                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusions

### 8.1 For Safety-Critical Applications (Elderly Care)
- **Recommended**: Best Prompt (Safety-First) with 25% threshold
- **Result**: 97.9% fall recall (misses only 1 in 48 falls)
- **Trade-off**: More false alarms, but acceptable for safety

### 8.2 For Balanced Applications with Explainability
- **Recommended**: RAG Pipeline
- **Result**: 85.4% recall, 67.8% accuracy
- **Advantage**: Explains decisions via similar cases

### 8.3 For Highest Accuracy
- **Recommended**: Enhanced Few-Shot with GPT-4o
- **Result**: 75.4% accuracy
- **Trade-off**: Some falls may be missed (41.7% recall)

### 8.4 Key Insights
1. **Prompt design matters**: Safety-first framing improves recall dramatically
2. **RAG provides explainability**: Retrieved similar cases explain decisions
3. **Rich text descriptions help**: Converting numbers to natural language improves LLM understanding
4. **Aggregation threshold is key**: Lower thresholds (25%) catch more falls
5. **GPT-4o outperforms GPT-4o-mini**: Worth the extra cost for safety applications

---

## 9. Project Completion Status

| Requirement | Status | Details |
|-------------|--------|---------|
| 4 Prompting Strategies | ✅ Complete | Zero-Shot, Few-Shot, CoT, Self-Consistency |
| RAG Pipeline | ✅ Complete | 1,305 examples, semantic retrieval |
| LoRA Fine-Tuning | ⏳ Processing | Job submitted, validating files |
| Explainable AI | ✅ Complete | XAI with reasoning traces |
| Data Pipeline | ✅ Complete | 419 videos, 1,882 windows |
| Evaluation | ✅ Complete | All techniques on Test, key ones on Val+Test |
| Documentation | ✅ Complete | README, Proposal, Architecture, Report |

---

## 10. Files and Scripts

### 10.1 Main Scripts
| Script | Purpose |
|--------|---------|
| `balanced_pipeline.py` | Data processing and window generation |
| `zero_shot.py` | Zero-shot baseline |
| `few_shot.py` | Few-shot with examples |
| `chain_of_thought.py` | CoT reasoning |
| `self_consistency.py` | Multi-sample voting |
| `rag_fall_detection.py` | RAG pipeline |
| `lora_finetune.py` | LoRA fine-tuning |
| `xai_explanations.py` | Explainable AI |
| `best_prompt.py` | Safety-first prompting |

### 10.2 Data Files
| Directory | Contents |
|-----------|----------|
| `data/windows/` | 1,882 sliding window JSONs |
| `data/metadata/` | Ground truth labels |
| `data/manifests/` | Video manifests |
| `data/poses/` | Extracted pose features |

---

*Report generated: May 2, 2026*
*Project: GenAI-Based Multimodal Fall Detection using Sliding Window Temporal Reasoning*
*Repository: https://github.com/Ramreddy2748/Gen_AI_Project*
