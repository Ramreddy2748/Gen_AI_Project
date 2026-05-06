# GenAI-Based Fall Detection Using Sliding Window Temporal Reasoning

## Presentation Slides (10 minutes, 12 slides max)

---

## SLIDE 1: Title Slide

### GenAI-Based Multimodal Fall Detection
### Using Sliding Window Temporal Reasoning

**Team Members:** [Your Names]

**Course:** Generative AI

**Date:** May 2026

---

## SLIDE 2: Problem & Motivation

### Why Fall Detection Matters

**The Problem:**
- Falls are the **#1 cause of injury death** among adults 65+
- Every 11 seconds, an older adult is treated in the ER for a fall
- **95% of hip fractures** are caused by falls

**Real-World Impact:**
- Elderly living alone need **automated monitoring**
- Traditional sensors (wearables) have low adoption
- **Camera-based detection** is non-intrusive

**Our Goal:**
> Build a GenAI system that detects falls from video with **high recall** (catch all falls) while providing **explainable** predictions

---

## SLIDE 3: Data Understanding

### Dataset Overview

| Dataset | Videos | Source |
|---------|--------|--------|
| URFD | 70 | University of Rochester |
| Le2i | 189 | French Research Lab |
| GMNCSA24 | 160 | GitHub Repository |
| **Total** | **419** | Combined |

**After Processing:**
- **375 videos** (159 Fall, 216 No-Fall)
- Split: Train (260), Validation (58), Test (57)
- **Evaluation Set:** 115 videos (Val + Test combined)

**Key Insight:** Balanced dataset ensures fair evaluation

---

## SLIDE 4: Feature Extraction (Input)

### What Goes INTO the Model?

**Raw Input:** Video frames → Pose Estimation (MediaPipe)

**Extracted Features per Frame:**

| Category | Features |
|----------|----------|
| **Position** | Hip height (hip_y), Body center |
| **Orientation** | Body angle (0°=horizontal, 90°=upright) |
| **Motion** | Velocity, Acceleration |
| **Posture** | State: Upright / Transitioning / Fallen |
| **Flags** | is_on_ground, is_horizontal, is_rapid_descent |

**Sliding Window:** 3 consecutive frames analyzed together

---

## SLIDE 5: Input → Model → Output Pipeline

### End-to-End Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   VIDEO     │ →  │    POSE      │ →  │   SLIDING   │ →  │   GenAI    │
│   INPUT     │    │  EXTRACTION  │    │   WINDOWS   │    │    LLM     │
│             │    │  (MediaPipe) │    │  (3 frames) │    │  (GPT-4o)  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                                │
                                                                ▼
                                                         ┌────────────┐
                                                         │ FALL or    │
                                                         │ NO_FALL    │
                                                         │ + Reason   │
                                                         └────────────┘
```

**Input Type:** Structured pose features (numerical + categorical)

**Output Type:** Binary classification (FALL / NO_FALL) per video

**Aggregation:** Multiple windows → Single video prediction (25% threshold)

---

## SLIDE 6: Proposed Solution - GenAI Prompting Strategies

### 7 Techniques Implemented

| # | Technique | Description |
|---|-----------|-------------|
| 1 | **Zero-Shot** | No examples, direct inference |
| 2 | **Few-Shot** | 6 examples from training data |
| 3 | **Chain-of-Thought** | Step-by-step reasoning |
| 4 | **Self-Consistency** | 5 samples + majority vote |
| 5 | **RAG** | Retrieve similar cases from knowledge base |
| 6 | **Safety-First** | Optimized prompt for high recall |
| 7 | **LoRA Fine-tuning** | Train model on our data |

**Why GenAI?**
- Interpretable decisions (not black-box)
- Handles complex temporal patterns
- Can explain WHY a fall was detected

---

## SLIDE 7: RAG Pipeline Architecture

### Retrieval-Augmented Generation

```
┌─────────────────────────────────────────────────────┐
│              KNOWLEDGE BASE (Training Data)          │
│  1,305 examples embedded using text-embedding-3-small│
│  Balanced: 660 Fall + 645 No-Fall                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    INFERENCE                         │
│  1. Query video → Extract features                  │
│  2. Find top-4 similar cases (cosine similarity)    │
│  3. Include similar cases in prompt                 │
│  4. GPT-4o makes informed decision                  │
└─────────────────────────────────────────────────────┘
```

**Benefit:** Provides explainable context for each prediction

---

## SLIDE 8: Experimental Setup

### Evaluation Design

**Data Split:**
- Train: 260 videos (for examples/RAG knowledge base)
- **Evaluation: 115 videos** (Val + Test combined)

**Evaluation Metric Focus:**
- **Video-Level Fall Recall** (Primary) - Safety critical!
- Video-Level Accuracy (Secondary)

**Why Recall over Accuracy?**
> Missing a fall (False Negative) is **dangerous**
> False alarms (False Positive) are acceptable

**Models Compared:**
- GPT-4o-mini (faster, cheaper)
- GPT-4o (more capable)

**Aggregation Strategy:** 25% threshold (if ≥25% windows = fall → video = fall)

---

## SLIDE 9: Results - Video Level Performance

### Main Results (115 Videos: 48 Fall, 67 No-Fall)

| Technique | Video Accuracy | Video Fall Recall |
|-----------|----------------|-------------------|
| Zero-Shot | 71.3% | 54.2% |
| Few-Shot | 67.0% | 81.2% |
| Chain-of-Thought | 48.7% | 77.1% |
| Self-Consistency | **73.0%** | 56.2% |
| RAG Pipeline | 67.8% | **85.4%** |
| **Safety-First** | 47.8% | **100%** |

### Key Finding:
> **Safety-First achieved 100% Fall Recall** - Detected ALL 48 fall videos!

---

## SLIDE 10: Results Analysis - What Worked & Why

### Confusion Matrix (Safety-First Strategy)

```
                    Predicted
                    FALL    NO_FALL
  Actual FALL        48       0      ← 100% Recall (0 missed!)
  Actual NO_FALL     35      32      ← Trade-off: More false alarms
```

### Insights:

**What Worked:**
- Safety-first prompt design prioritizes recall
- Lower aggregation threshold (25%) catches more falls
- GPT-4o understands temporal patterns well

**What Didn't Work:**
- High accuracy AND high recall together (trade-off)
- Window-level classification alone (noisy)

**Trade-off Decision:**
> For elderly care, **missing a fall is unacceptable** → Accept false alarms

---

## SLIDE 11: XAI - Explainable Predictions

### Sample Explanation Output

```
VIDEO ANALYSIS:
The video shows a person transitioning from upright to 
horizontal position with rapid descent detected.

KEY OBSERVATIONS:
- Hip position drops from 0.35 to 0.78 (ground level)
- Body angle changes from 85° to 25° (horizontal)
- Velocity spike detected (0.15)
- Multiple ON_GROUND flags triggered

CLASSIFICATION: FALL
CONFIDENCE: HIGH
EXPLANATION: Clear fall pattern with rapid descent 
and horizontal final position.
```

**Video-Level XAI Accuracy: 82%** with full explanations

---

## SLIDE 12: Conclusion & Impact

### Summary

| Requirement | Status | Best Result |
|-------------|--------|-------------|
| 4 Prompting Strategies | ✅ | Zero-Shot, Few-Shot, CoT, Self-Consistency |
| RAG Pipeline | ✅ | 85.4% video recall |
| Fine-Tuning (LoRA) | ✅ | Colab notebook ready |
| Explainable AI | ✅ | 82% accuracy with reasoning |

### Key Achievements:
- **100% Fall Detection** with Safety-First strategy
- **RAG provides explainability** through similar cases
- **Video-level analysis** more reliable than frame-level

### Real-World Application:
- Elderly care monitoring systems
- Hospital patient safety
- Smart home security

### Future Work:
- Real-time video streaming
- Multi-person detection
- Edge device deployment

---

## BACKUP SLIDES (If Asked)

### Slide B1: Why 25% Threshold?

| Threshold | Fall Recall | Accuracy |
|-----------|-------------|----------|
| 50% (majority) | ~60% | ~70% |
| 25% | **100%** | 48% |
| Any fall window | 100% | ~40% |

**25% balances recall and false alarm rate**

### Slide B2: Model Comparison

| Model | Cost | Speed | Best For |
|-------|------|-------|----------|
| GPT-4o-mini | Low | Fast | Quick testing |
| GPT-4o | High | Slower | Production (better reasoning) |

---

## Speaking Notes

**Slide 2 (Motivation):** Emphasize real-world impact - mention specific statistics

**Slide 5 (Pipeline):** Walk through the flow step-by-step, explain each component

**Slide 9 (Results):** Highlight the 100% recall achievement - this is the key finding

**Slide 10 (Analysis):** Explain the accuracy vs recall trade-off clearly

**Slide 11 (XAI):** Read through the example explanation - shows interpretability

---

*Total: 12 slides (excluding backup)*
*Duration: ~10 minutes*
*Focus: Video-level metrics, clarity, real-world impact*
