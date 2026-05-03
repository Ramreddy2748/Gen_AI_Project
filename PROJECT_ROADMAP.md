# DATA 266 — Multimodal Fall Detection: Phased Project Roadmap

**Project**: GenAI-Based Multimodal Fall Detection using Prompting, RAG & LoRA  
**Team**: Shristi Kumar · Harshitha Boinepally · Venkata Ramireddy Seelam (Ram)  
**Course**: DATA 266 — Generative AI  
**Final deadline**: ~Week of May 5 | **Midpoint presentation**: April 8

---

## Overview of Phases

| Phase | Name | Dates | Gate |
|-------|------|-------|------|
| **Phase 1** | Foundation + Presentation | Mar 26 – Apr 8 | 10-slide deck, initial results |
| **Phase 2** | Full Pipeline Implementation | Apr 9 – Apr 22 | RAG + LoRA working end-to-end |
| **Phase 3** | Evaluation + Final Report | Apr 23 – May 5 | Complete metrics, write-up |

---

---

## PHASE 1: Foundation + April 8 Presentation
**Deadline: April 8 | Duration: 13 days**

### Objective
Get the environment running, obtain initial baseline numbers, and build a compelling 10-slide deck that satisfies every requirement Dr. Masum listed. Every member must have a speaking role and concrete results to show.

---

### 1.1 — Environment Setup *(Day 1–2 | Owner: All)*

**Goal**: Every teammate can run code locally or on Colab without blockers.

```bash
# Step 1: Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Step 2: Install core dependencies
pip install torch torchvision transformers sentence-transformers \
            faiss-cpu chromadb mediapipe opencv-python \
            pandas numpy scikit-learn matplotlib seaborn \
            Pillow requests tqdm

# Step 3: Install Ollama (free local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava          # Vision-Language model
ollama pull llama3.1       # Text reasoning model

# Step 4: Verify GPU (Colab) or CPU fallback
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Deliverable**: A working `requirements.txt` and a shared Google Colab notebook all three members can open.

---

### 1.2 — Dataset Acquisition & Preprocessing *(Day 1–3 | Owner: Ram)*

**Goal**: URFD dataset downloaded, split, and preprocessed into a usable format.

**Dataset**: [URFD — UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html)
- 70 sequences: 30 falls + 40 ADL (Activities of Daily Living)
- Modalities: RGB video, depth (Kinect), accelerometer signals
- **Split**: 70% train / 15% val / 15% test (subject-based, no leakage)

**Steps:**
```
1. Download URFD → organise into /data/falls/ and /data/adl/
2. Extract frames from each video at 10 FPS using OpenCV
3. Label each sequence: FALL=1, NO_FALL=0
4. Log sequence metadata: subject_id, duration, modality availability
5. Document dataset statistics (class balance, frame count)
```

**Deliverable**: `data_stats.json` with sequence count, class distribution, modality breakdown — this feeds directly into Slide 3 of the presentation.

---

### 1.3 — Pose Feature Extraction *(Day 3–5 | Owner: Ram)*

**Goal**: Extract structured pose keypoints from all video sequences using MediaPipe.

**Why**: Pose features give the LLM geometric context (torso angle, velocity, joint positions) that raw pixels cannot efficiently convey.

```python
import mediapipe as mp
import cv2, json

mp_pose = mp.solutions.pose

def extract_pose_features(video_path):
    """Returns per-frame keypoint dict with torso_angle, hip_height, velocity."""
    cap = cv2.VideoCapture(video_path)
    features = []
    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                features.append({
                    "hip_y":    lm[mp_pose.PoseLandmark.LEFT_HIP].y,
                    "shoulder_y": lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    "nose_y":   lm[mp_pose.PoseLandmark.NOSE].y,
                    # torso_angle: angle between shoulder-hip vector and vertical
                })
    cap.release()
    return features
```

**Deliverable**: `/data/pose_features/` folder with one JSON per sequence, capturing keypoint trajectories. Summary stats (avg torso angle at fall vs. ADL).

---

### 1.4 — Baseline 1: Zero-Shot Prompting *(Day 4–7 | Owner: Shristi)*

**Goal**: Run the simplest possible baseline — zero-shot visual prompting — and record accuracy, precision, recall, F1.

**Objective**: Establish the floor. Show that raw prompting without grounding is insufficient, which motivates RAG and LoRA.

**Implementation**:
```python
import ollama, base64

def encode_frame(frame_path):
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def zero_shot_classify(frame_path):
    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": (
                "Analyze this image carefully. "
                "Is the person experiencing a fall? "
                "Answer with exactly one word: FALL or NO_FALL. "
                "Then explain your reasoning in one sentence."
            ),
            "images": [encode_frame(frame_path)]
        }]
    )
    return response["message"]["content"]
```

**Evaluation Protocol**:
- Sample 1 representative frame per sequence (the most critical frame — peak motion)
- Run zero-shot on all 70 sequences
- Compare predicted label vs. ground truth
- Compute: Accuracy, Precision, Recall (priority), F1, False Positive Rate

**Deliverable**: `results/zero_shot_results.csv` with per-sequence predictions + `baseline_metrics.json`.

---

### 1.5 — Baseline 2: Chain-of-Thought Prompting *(Day 5–7 | Owner: Shristi)*

**Goal**: Improve over zero-shot using structured reasoning steps. Quantify the delta.

```python
def cot_classify(frame_path, pose_features):
    prompt = f"""
You are analyzing a video frame for fall detection. Use the following reasoning steps:

Step 1 — Person Detection: Is a person visible? Describe their position.
Step 2 — Posture Analysis: Is the body upright, tilted, or horizontal?
           Pose data: torso_angle={pose_features['torso_angle']:.1f}°
Step 3 — Motion Context: Does the posture suggest rapid descent or loss of balance?
Step 4 — Final Decision: Based on steps 1-3, is this a FALL or NO_FALL?

Provide your answer as: DECISION: [FALL/NO_FALL] | CONFIDENCE: [HIGH/MEDIUM/LOW]
"""
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": prompt,
                   "images": [encode_frame(frame_path)]}]
    )
    return response["message"]["content"]
```

**Deliverable**: `results/cot_results.csv`. Side-by-side comparison table: Zero-Shot vs. CoT.

---

### 1.6 — Skeleton RAG Knowledge Base *(Day 6–8 | Owner: Harshitha)*

**Goal**: Build the initial version of the RAG knowledge base from existing annotated cases. This gives you something to demo on April 8, even if retrieval is not fully optimized yet.

**Knowledge Base Structure**:
```python
# Each entry in the knowledge base
case_entry = {
    "case_id": "fall_001",
    "sequence_id": "fall-01-cam0",
    "label": "FALL",
    "description": "Person walking, sudden loss of balance, rapid descent to floor.",
    "pose_summary": "torso_angle: 82°, hip_height: 0.15 (normalized), velocity: high",
    "key_features": ["horizontal_body", "rapid_descent", "arms_extended"],
    "outcome": "FALL",
    "notes": "Classic forward fall, hands did not brace in time."
}
```

**Build the index**:
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(cases):
    texts = [f"{c['description']} {c['pose_summary']}" for c in cases]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings
```

**Minimum viable knowledge base for April 8**: Manually annotate 20–30 cases (from the 70 URFD sequences) with textual descriptions. This is enough to demonstrate the concept.

**Deliverable**: `knowledge_base/cases.json` + `knowledge_base/faiss.index`. Demo retrieval for 3 test queries.

---

### 1.7 — April 8 Presentation (10 Slides) *(Day 9–13 | Owner: All)*

**Each slide maps to a requirement from Dr. Masum:**

| Slide | Title | Content | Owner |
|-------|-------|---------|-------|
| 1 | **Title + Team** | Project name, names, roles | All |
| 2 | **Problem Definition** | Why fall detection matters; limitations of vision-only systems; 300K hospitalizations/yr; our solution angle | Shristi |
| 3 | **Dataset** | URFD: 70 sequences, 3 modalities, split strategy, class distribution chart | Ram |
| 4 | **What is Novel** | Combined: pose-geometry + LLM reasoning + RAG grounding — not just a detector, an *explainer* | Shristi |
| 5 | **Methodology Overview** | 3-baseline architecture diagram: Prompting → RAG → LoRA | Harshitha |
| 6 | **Experimental Design** | Baselines table, metrics (recall priority), evaluation protocol | All |
| 7 | **Initial Results** | Zero-shot vs. CoT comparison table + key insight: CoT reduces FP by X% | Shristi |
| 8 | **RAG Demo** | Show a retrieval example — query case → top-3 retrieved similar cases → grounded LLM answer | Harshitha |
| 9 | **Challenges & Mitigations** | LLM hallucination → RAG grounding; limited data → synthetic cases; compute → Colab + quantized models | Ram |
| 10 | **Future Plans** | LoRA fine-tuning, full RAG evaluation, temporal analysis, cross-dataset (Le2i) | All |

**Key message to highlight (novelty):**
> *We are not building another fall detector. We are building an **explainable fall reasoning system** that uses LLM chain-of-thought + pose geometry + knowledge-grounded retrieval to both detect AND explain falls — something no prior CV-only approach does.*

**Presentation logistics**:
- 8 minutes minimum: ~45 seconds per slide, leave 90 seconds for Q&A buffer
- All three members present: Shristi (problem + results), Harshitha (methodology + RAG), Ram (data + challenges)
- Practice run: April 6 (Sunday), internal critique session

---

### Phase 1 Checklist

```
[ ] Environment set up and shared (All) — Day 2
[ ] URFD dataset downloaded and split (Ram) — Day 3
[ ] Pose features extracted for all 70 sequences (Ram) — Day 5
[ ] Zero-shot baseline results computed (Shristi) — Day 7
[ ] CoT baseline results computed (Shristi) — Day 7
[ ] Initial knowledge base built (Harshitha) — Day 8
[ ] FAISS index working, retrieval demo ready (Harshitha) — Day 9
[ ] Slide deck drafted (All) — Day 10
[ ] Internal practice run (All) — Day 12 (April 6)
[ ] Final slide polish (All) — Day 13 (April 7)
[ ] PRESENTATION — April 8
```

---
---

## PHASE 2: Full Pipeline Implementation
**Dates: April 9 – April 22 | Duration: 14 days**

### Objective
Build the complete end-to-end system: full RAG pipeline with hybrid retrieval, few-shot and self-consistency prompting, and LoRA fine-tuning. This is where the paper-quality work happens.

---

### 2.1 — Complete RAG Pipeline *(Days 1–5 | Owner: Harshitha)*

**Goal**: Go from skeleton knowledge base to a fully functioning RAG system with medical literature grounding.

**Step 1: Expand the Knowledge Base**
- Annotate all 70 URFD sequences with textual descriptions
- Scrape 20–30 PubMed/arXiv abstracts: query `"fall detection pose"`, `"elderly fall ECG stress correlation"`
- Add physiological pattern descriptions from the `Revised_GenAI_Methodology.md` reference

**Step 2: Hybrid Retrieval (Vector + BM25)**
```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, cases, faiss_index, embeddings, model):
        self.cases = cases
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.model = model
        corpus = [c["description"] + " " + c["pose_summary"] for c in cases]
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])

    def retrieve(self, query, top_k=5, alpha=0.5):
        # Dense retrieval
        q_embed = self.model.encode([query])
        _, dense_ids = self.faiss_index.search(q_embed, top_k * 2)
        
        # Sparse retrieval
        sparse_scores = self.bm25.get_scores(query.split())
        sparse_ids = np.argsort(sparse_scores)[::-1][:top_k * 2]
        
        # Reciprocal Rank Fusion
        scores = {}
        for rank, idx in enumerate(dense_ids[0]):
            scores[idx] = scores.get(idx, 0) + 1 / (rank + 60)
        for rank, idx in enumerate(sparse_ids):
            scores[idx] = scores.get(idx, 0) + alpha / (rank + 60)
        
        top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [self.cases[i] for i in top_ids]
```

**Step 3: RAG-Enhanced Prompt**
```python
def rag_classify(frame_path, pose_features, retrieved_cases):
    context = "\n".join([
        f"Case {i+1}: {c['description']} → {c['outcome']}"
        for i, c in enumerate(retrieved_cases)
    ])
    
    prompt = f"""
Retrieved similar cases:
{context}

Current case:
- Pose: torso_angle={pose_features['torso_angle']:.1f}°, hip_height={pose_features['hip_y']:.2f}
- Visual: [see attached frame]

Using the retrieved cases as reference, classify this frame: FALL or NO_FALL?
Explain which retrieved case is most similar and why.
"""
    return ollama.chat(model="llava",
                       messages=[{"role": "user", "content": prompt,
                                  "images": [encode_frame(frame_path)]}])
```

**Evaluation**: Measure Precision@K (K=3,5) on retrieval quality separately from end-task accuracy.

---

### 2.2 — Few-Shot & Self-Consistency Prompting *(Days 3–7 | Owner: Shristi)*

**Goal**: Implement all four prompting strategies from the proposal and compare them rigorously.

**Few-Shot**:
```python
FEW_SHOT_EXAMPLES = """
Example 1: [Person upright, walking normally] → NO_FALL
  Reasoning: Torso vertical (12°), steady gait, no rapid descent.

Example 2: [Person tilted 75°, arms outstretched] → FALL
  Reasoning: Torso near horizontal, arms bracing, rapid downward motion.

Example 3: [Person sitting down slowly] → NO_FALL
  Reasoning: Controlled descent, torso angle 45° but velocity low.
"""

def few_shot_classify(frame_path, pose_features):
    prompt = FEW_SHOT_EXAMPLES + f"\nCurrent: torso={pose_features['torso_angle']:.1f}° → ?"
    # ... call ollama
```

**Self-Consistency**:
```python
def self_consistency_classify(frame_path, pose_features, n_paths=5):
    perspectives = [
        "Focus on body geometry: torso angle and hip height.",
        "Focus on motion dynamics: velocity and acceleration.",
        "Focus on contextual cues: environment and arm position.",
        "Focus on pose symmetry: bilateral joint positions.",
        "Focus on temporal context: is this a transition frame?"
    ]
    votes = []
    for p in perspectives:
        result = classify_with_perspective(frame_path, pose_features, p)
        votes.append(parse_label(result))  # "FALL" or "NO_FALL"
    
    from collections import Counter
    final_label = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / n_paths
    return final_label, confidence
```

**Results table to fill in by end of Phase 2**:
| Strategy | Accuracy | Precision | Recall | F1 | FPR |
|----------|----------|-----------|--------|----|-----|
| Zero-Shot | — | — | — | — | — |
| CoT | — | — | — | — | — |
| Few-Shot | — | — | — | — | — |
| Self-Consistency | — | — | — | — | — |
| RAG-Enhanced | — | — | — | — | — |

---

### 2.3 — LoRA Fine-Tuning *(Days 5–12 | Owner: Shristi + Ram)*

**Goal**: Fine-tune a vision model using LoRA for binary fall classification. Compare against prompting-only approaches.

**Setup (Google Colab T4 GPU — free)**:
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Load base model (ViT or CLIP-based)
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.IMAGE_CLASSIFICATION,
    r=8,              # Low-rank dimension
    lora_alpha=16,    # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Inject into attention layers only
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# Expected: ~0.5% of total params — minimal cost
```

**Training**:
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_fall_detector",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="best",
    metric_for_best_model="recall",   # Priority metric
    load_best_model_at_end=True,
    fp16=True                          # Mixed precision for Colab
)
```

**Deliverable**: Trained LoRA adapter weights (~10MB), training curves, test-set metrics.

---

### Phase 2 Checklist

```
[ ] Knowledge base expanded to 70+ cases + 20 literature snippets (Harshitha) — Apr 13
[ ] Hybrid retrieval (dense + BM25) implemented (Harshitha) — Apr 14
[ ] Retrieval quality evaluated: Precision@3, Precision@5 (Harshitha) — Apr 15
[ ] Few-shot prompting implemented and evaluated (Shristi) — Apr 14
[ ] Self-consistency prompting implemented and evaluated (Shristi) — Apr 16
[ ] RAG-enhanced classification evaluated end-to-end (Harshitha + Shristi) — Apr 17
[ ] LoRA training started on Colab (Shristi + Ram) — Apr 14
[ ] LoRA training complete, test metrics recorded (Shristi + Ram) — Apr 20
[ ] All 5 baselines compared in unified results table (All) — Apr 22
```

---
---

## PHASE 3: Evaluation, Analysis & Final Report
**Dates: April 23 – May 5 | Duration: ~12 days**

### Objective
Produce research-quality evaluation, interpret results with genuine insight (not just numbers), and write the final report. This is what separates a good project from an award-winning one.

---

### 3.1 — Comprehensive Evaluation *(Days 1–5 | Owner: All)*

**Primary Metrics** (computed on identical test split for all baselines):
- **Recall** — priority metric (missing a real fall is dangerous)
- **False Positive Rate** — second priority (alarm fatigue)
- Accuracy, Precision, F1-score
- Inference latency (target: <2s per clip)

**RAG-Specific Metrics**:
- `Precision@K` for K = 3, 5
- Explanation coherence: 3-point human rating (1=poor, 2=adequate, 3=strong)

**Statistical Rigor**:
```python
# Confidence intervals via bootstrap
from sklearn.utils import resample
import numpy as np

def bootstrap_metric(y_true, y_pred, metric_fn, n_iter=1000):
    scores = []
    for _ in range(n_iter):
        indices = resample(range(len(y_true)))
        scores.append(metric_fn(
            [y_true[i] for i in indices],
            [y_pred[i] for i in indices]
        ))
    return np.mean(scores), np.percentile(scores, [2.5, 97.5])
```

**Ablation Study** (what contributes what):
| Configuration | Recall | Notes |
|--------------|--------|-------|
| Vision only (zero-shot) | — | Baseline floor |
| Vision + Pose (CoT) | — | +Geometry |
| Vision + Pose + RAG | — | +Grounding |
| Vision + Pose + RAG + LoRA | — | Full system |

---

### 3.2 — Error Analysis & Insights *(Days 4–7 | Owner: Shristi)*

**Goal**: Understand *why* the model fails, not just *that* it fails.

```python
# Tag failure modes
failure_modes = {
    "FP_sitting": 0,      # Predicted FALL when person was sitting
    "FP_bending": 0,      # Predicted FALL when person was bending
    "FN_slow_fall": 0,    # Missed a slow/gradual fall
    "FN_occluded": 0,     # Missed a fall due to occlusion
}

# For each false positive/negative, manually inspect and categorize
# This produces qualitative insight for the paper
```

**Qualitative Example to include in report**:
> "In 7 of 12 false positives, the system misclassified a person bending down to pick up an object. CoT prompting reduced this by 43% by explicitly reasoning about intent and velocity, while RAG retrieved similar bending cases as counter-examples."

---

### 3.3 — Robustness Testing *(Days 5–8 | Owner: Ram)*

**Goal**: Test stability across settings (required for top-tier evaluation).

```python
# Test across lighting conditions (simulate via brightness/contrast transforms)
# Test across camera angles (use different cameras in URFD if available)
# Test cross-dataset: apply model trained on URFD to Le2i dataset (if time permits)

import cv2
def augment_frame(frame, brightness=1.0, contrast=1.0):
    augmented = cv2.convertScaleAbs(frame, alpha=contrast, beta=(brightness-1)*127)
    return augmented

# Run evaluation on augmented frames — does recall drop significantly?
```

---

### 3.4 — Final Report *(Days 7–12 | Owner: All)*

**Structure** (research paper format, 6–8 pages):

```
1. Abstract (150 words)
2. Introduction — Problem, motivation, contributions (bullets)
3. Related Work — Prior fall detection, VLMs, RAG in healthcare
4. Dataset & Preprocessing — URFD stats, split, pose extraction
5. Methodology — 3 baselines described formally
6. Experimental Setup — Hardware, hyperparams, evaluation protocol
7. Results — Tables + figures, statistical significance
8. Discussion — What worked, what didn't, why
9. Conclusion & Future Work
10. References
```

**Key Figures to generate**:
- Figure 1: System architecture diagram (Prompting → RAG → LoRA)
- Figure 2: Baseline comparison bar chart (Recall + FPR)
- Figure 3: RAG retrieval example (query → top-3 cases → output)
- Figure 4: LoRA training curves (loss + recall vs. epoch)
- Figure 5: Confusion matrices for each baseline

---

### Phase 3 Checklist

```
[ ] All baselines evaluated on identical test split (All) — Apr 25
[ ] Bootstrap confidence intervals computed (Shristi) — Apr 26
[ ] Ablation study table complete (All) — Apr 27
[ ] Error analysis & failure mode categorization (Shristi) — Apr 28
[ ] Robustness testing (augmentation + cross-setting) (Ram) — Apr 29
[ ] All figures generated (All) — Apr 30
[ ] Final report first draft (All) — May 2
[ ] Peer review / internal critique (All) — May 3
[ ] Final report submitted (All) — May 5
```

---
---

## Novelty Statement (for Slide 4 and Abstract)

> **What makes this novel**: Prior fall detection systems are *detectors* — they output a binary label. Our system is a *fall reasoning system*. By combining (1) pose-geometry features as structured LLM input, (2) Chain-of-Thought decomposition for interpretable decisions, and (3) knowledge-grounded RAG retrieval from annotated case histories, the system not only classifies falls but **explains its reasoning in natural language** — a critical property for clinical trust and healthcare deployment. This is the first known application of RAG-grounded VLM reasoning to multimodal fall detection.

---

## Team Responsibilities Summary

| Member | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Ram** | Dataset download, split, pose extraction | Expand KB, robustness tests | Robustness eval, report §3–4 |
| **Harshitha** | Skeleton RAG + FAISS index | Full hybrid RAG pipeline, retrieval eval | RAG analysis, report §5 |
| **Shristi** | Zero-shot + CoT baselines | Few-shot, self-consistency, LoRA | Error analysis, report §6–8 |

---

## Award Criteria Alignment

| Dr. Masum's Criteria | How We Address It |
|---------------------|------------------|
| **Originality** | Pose-geometry + CoT + RAG for fall *reasoning*, not just detection |
| **Rigorous Evaluation** | 5 baselines, bootstrap CI, identical splits, Precision@K for RAG |
| **Robustness** | Augmentation tests, cross-setting evaluation |
| **Reproducibility** | All code open, `requirements.txt`, fixed random seeds, documented splits |
