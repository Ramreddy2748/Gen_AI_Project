# GenAI-Based Multimodal Fall Detection using Sliding Window Temporal Reasoning

## Project Proposal

### 1. Problem Statement

Falls are a leading cause of injury and death among elderly populations. Timely detection of falls can significantly reduce the severity of injuries and save lives. This project aims to develop a **GenAI-powered fall detection system** that uses temporal reasoning over pose estimation data to accurately identify fall events in video streams.

### 2. Objectives

1. **Implement multiple GenAI prompting strategies** for fall detection:
   - Zero-Shot Prompting
   - Few-Shot Prompting
   - Chain-of-Thought (CoT) Reasoning
   - Self-Consistency

2. **Develop a full RAG (Retrieval-Augmented Generation) pipeline** for explainable predictions

3. **Implement LoRA fine-tuning** for improved accuracy

4. **Build a sliding window temporal reasoning system** that:
   - Extracts pose features using MediaPipe
   - Creates temporal windows of 3 consecutive frames
   - Feeds pose sequences to LLMs for classification

### 3. Datasets

| Dataset | Source | Videos | Description |
|---------|--------|--------|-------------|
| URFD | University of Rochester | 70 | Indoor fall scenarios |
| Le2i | French research lab | 189 | Multiple camera views |
| GMNCSA24 | HuggingFace/GitHub | 160 | Diverse fall types |

**Total: 419 videos** (159 fall, 216 no-fall after processing)

### 4. Methodology

#### 4.1 Data Pipeline

```
Video → Frame Extraction → Pose Estimation → Feature Extraction → Sliding Windows
                              (MediaPipe)
```

#### 4.2 Feature Engineering

**Per-Frame Features:**
- Position metrics (hip_y, shoulder_y, body_center_y)
- Orientation (body_angle_degrees)
- Motion (velocity, acceleration, jerk)
- Posture state (upright/transitioning/fallen)
- Fall indicators (rapid_descent, on_ground, horizontal)

**Window-Level Features:**
- Trajectory direction
- Maximum velocity
- Velocity spike detection
- Fall likelihood score

#### 4.3 GenAI Strategies

| Strategy | Description | Expected Strength |
|----------|-------------|-------------------|
| Zero-Shot | No examples, direct inference | Baseline |
| Few-Shot | 6 examples from training set | Better context |
| CoT | Step-by-step reasoning | Interpretability |
| Self-Consistency | Multiple samples, majority vote | Robustness |
| RAG | Retrieved similar examples | Explainability |
| LoRA | Fine-tuned model | Highest accuracy |

### 5. Evaluation Metrics

- **Window-Level**: Accuracy, Precision, Recall, F1-Score
- **Video-Level**: Aggregated predictions with configurable thresholds
- **Primary Focus**: **Fall Recall** (safety-critical metric)

### 6. Expected Outcomes

1. **High Fall Recall (>90%)**: Critical for safety applications
2. **Comparative Analysis**: Performance across all prompting strategies
3. **Explainable Predictions**: RAG provides similar case references
4. **Deployable Pipeline**: End-to-end system from video to prediction

### 7. Technical Stack

| Component | Technology |
|-----------|------------|
| Video Processing | OpenCV |
| Pose Estimation | MediaPipe Pose |
| LLM | GPT-4o, GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Fine-tuning | OpenAI LoRA API |
| Language | Python 3.8+ |

### 8. Timeline

| Phase | Tasks | Status |
|-------|-------|--------|
| 1 | Dataset collection & preprocessing | ✅ Complete |
| 2 | Feature extraction & window generation | ✅ Complete |
| 3 | Zero-Shot & Few-Shot implementation | ✅ Complete |
| 4 | CoT & Self-Consistency | ✅ Complete |
| 5 | RAG Pipeline | ✅ Complete |
| 6 | LoRA Fine-tuning | ⏳ In Progress |
| 7 | Final evaluation & documentation | ✅ Complete |

### 9. Results Achieved

| Strategy | Video Recall | Video Accuracy |
|----------|--------------|----------------|
| Zero-Shot | 20.8% | 66.7% |
| Few-Shot | 58.3% | 66.7% |
| CoT | 29.2% | 66.7% |
| Self-Consistency | 41.7% | 70.2% |
| Enhanced (GPT-4o) | 41.7% | 75.4% |
| **RAG** | **85.4%** | 67.8% |
| **Best Prompt** | **97.9%** | 58.3% |

### 10. Key Contributions

1. **Safety-First Fall Detection**: 97.9% recall prioritizes catching all falls
2. **RAG for Explainability**: Similar cases provide interpretable decisions
3. **Comprehensive Comparison**: 7 prompting strategies evaluated
4. **Multimodal Integration**: Pose features + LLM reasoning
5. **Temporal Reasoning**: Sliding window captures fall dynamics

### 11. References

1. OpenAI GPT-4 Technical Report
2. MediaPipe Pose Estimation Documentation
3. URFD Dataset Paper
4. Le2i Fall Detection Dataset
5. Retrieval-Augmented Generation (Lewis et al., 2020)
6. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

---

*Project completed as part of GenAI Course Term Project*
*Date: May 2026*
