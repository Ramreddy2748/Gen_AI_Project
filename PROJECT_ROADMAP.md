# DATA 266 — Multimodal Fall Detection: Revised Project Roadmap

**Project**: GenAI-Based Multimodal Fall Detection using Sliding Window Temporal Reasoning  
**Team**: Shristi Kumar · Harshitha Boinepally · Venkata Ramireddy Seelam (Ram)  
**Course**: DATA 266 — Generative AI  
**Final deadline**: Week of May 5

---

## Executive Summary

This project implements an **Explainable AI (XAI) system for fall detection** using a sliding window temporal approach with LLM-based reasoning. We demonstrate mastery of three core Gen AI pillars:

| Requirement | Implementation |
|-------------|----------------|
| **A. Prompt Engineering** | 4 strategies: Zero-Shot, Few-Shot, Chain-of-Thought (CoT), Self-Consistency |
| **B. RAG Pipeline** | Full retrieval-augmented generation with FAISS vector store and domain knowledge base |
| **C. Fine-Tuning (PEFT)** | LoRA fine-tuning for domain-specialized Explainable AI reasoning |

---

## System Architecture

### Sliding Window Temporal Fall Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW TEMPORAL FALL DETECTION                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌───────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  Video   │───▶│     Pose      │───▶│   Sliding Window    │───▶│       LLM       │
│  Input   │    │  Extraction   │    │     Generator       │    │  (GPT-4o-mini)  │
│          │    │  (MediaPipe)  │    │                     │    │                 │
│ Fall/    │    │ ENHANCED      │    │  Window 1: [F1,F2,F3]│    │   3 Frames      │
│ No-Fall  │    │ FEATURES:     │    │  Window 2: [F3,F4,F5]│    │   per call      │
│ Video    │    │ • Position    │    │  Window 3: [F5,F6,F7]│    │                 │
│          │    │ • Velocity    │    │                     │    │                 │
│          │    │ • Acceleration│    │  Stride: 2          │    │                 │
│          │    │ • Posture     │    │                     │    │                 │
│          │    │ • Fall Flags  │    │                     │    │                 │
└──────────┘    └───────────────┘    └─────────────────────┘    └────────┬────────┘
                                                                          │
                                                                          ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────────────────┐
│   Video-Level   │◀───│    Majority Vote    │◀───│      Window Predictions         │
│    Verdict:     │    │    Aggregation      │    │                                 │
│     FALL ✓      │    │                     │    │  Window 1: Fall                 │
│                 │    │                     │    │  Window 2: Fall                 │
│                 │    │                     │    │  Window 3: No-Fall              │
└─────────────────┘    └─────────────────────┘    └─────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────────────┐
                    │     Hallucination Filter      │
                    │                               │
                    │ Bending → Fall(1 window) +    │
                    │ No-Fall(2 windows) = No-Fall ✓│
                    └───────────────────────────────┘
```

---

## Enhanced Feature Engineering (Data Layer)

### Design Principle: Same Features for ALL Techniques

All GenAI techniques (Zero-Shot, Few-Shot, CoT, RAG, LoRA) receive **identical input features**.
Improvement comes from the **technique**, not from different data.

```
SAME ENRICHED FEATURES
        │
        ├──▶ Zero-Shot  ──▶ Performance A
        ├──▶ Few-Shot   ──▶ Performance B  
        ├──▶ CoT        ──▶ Performance C
        ├──▶ RAG + CoT  ──▶ Performance D  (+ Retrieved Knowledge)
        └──▶ LoRA       ──▶ Performance E  (+ Domain Specialization)

Fair comparison: Same input, different technique
```

### Comprehensive Feature Set Per Frame

| Category | Features | Purpose |
|----------|----------|---------|
| **Position** | `hip_y`, `shoulder_y`, `nose_y`, `body_center_y` | Where is the body? |
| **Velocity** | `velocity_hip_y`, `velocity_shoulder_y`, `velocity_magnitude` | How fast is it moving? |
| **Acceleration** | `acceleration_hip_y`, `acceleration_magnitude`, `jerk` | Is movement sudden? (KEY for falls!) |
| **Orientation** | `body_angle_degrees`, `torso_vertical_diff` | Is body upright or horizontal? |
| **Posture State** | `posture_state`: "upright", "transitioning", "fallen" | Classified posture |
| **Fall Flags** | `is_rapid_descent`, `is_on_ground`, `is_horizontal`, `is_high_acceleration` | Binary fall indicators |

### Why These Features Matter for Fall Detection

```
FALL ANATOMY:
─────────────────────────────────────────────────────────────────────────
Phase:       │    Pre-Fall    │     Falling     │     Post-Fall    │
─────────────────────────────────────────────────────────────────────────
hip_y:       │    ───────     │     ╱           │     ─────────    │
             │    stable      │   rapid ↑       │   high (ground)  │
─────────────────────────────────────────────────────────────────────────
velocity:    │     ~0         │   HIGH SPIKE!   │      ~0          │
─────────────────────────────────────────────────────────────────────────
acceleration:│     ~0         │   VERY HIGH!    │      ~0          │
─────────────────────────────────────────────────────────────────────────
posture:     │   "upright"    │ "transitioning" │    "fallen"      │
─────────────────────────────────────────────────────────────────────────

KEY INSIGHT: Falls have HIGH VELOCITY + HIGH ACCELERATION + NO RECOVERY
```

### Window-Level Analysis

Each 3-frame sliding window includes computed analysis:

```python
window_analysis = {
    # Motion metrics
    "window_motion_score": 0.25,
    "max_velocity_in_window": 0.15,
    "max_acceleration_in_window": 0.08,
    
    # Trajectory
    "trajectory_direction": "descending",  # "ascending", "stable"
    "is_controlled_movement": False,       # False = possible fall
    
    # Fall indicators
    "has_velocity_spike": True,
    "has_acceleration_spike": True,
    "shows_descent": True,
    "ends_on_ground": True,
    "ends_horizontal": True,
    
    # Posture transition
    "start_posture": "upright",
    "end_posture": "fallen",
    "posture_transition": "upright_to_fallen"  # KEY fall pattern!
}
```

### LLM-Friendly Descriptions

Each window includes human-readable text for LLM reasoning:

```python
llm_descriptions = {
    "window_summary": "RAPID DOWNWARD MOVEMENT detected. Person went from upright to fallen. HIGH FALL LIKELIHOOD.",
    
    "frame_descriptions": [
        "Frame 1: standing/upright, stationary, posture=upright",
        "Frame 2: mid-position, moving DOWN rapidly, posture=transitioning", 
        "Frame 3: on/near ground, stationary, posture=fallen"
    ],
    
    "motion_description": "RAPID DOWNWARD MOVEMENT detected - possible fall in progress",
    "posture_description": "CRITICAL: Person went from upright to fallen position",
    "fall_assessment": "HIGH FALL LIKELIHOOD: rapid movement, sudden acceleration, downward trajectory, ends on ground, body horizontal"
}
```

### Video-Level Context

Each window also receives video-level context:

```python
video_context = {
    "initial_state": {"hip_y": 0.25, "posture_state": "upright"},
    "final_state": {"hip_y": 0.65, "posture_state": "fallen"},
    
    "trajectory": {
        "total_hip_descent": 0.40,
        "max_velocity_observed": 0.18,
        "max_acceleration_observed": 0.12
    },
    
    "phases": {
        "started_upright": True,
        "ended_on_ground": True,
        "has_significant_transition": True,
        "shows_recovery": False  # No recovery = likely fall
    },
    
    "fall_indicators": {
        "has_rapid_descent": True,
        "has_high_acceleration": True,
        "fall_likelihood_score": 0.85
    }
}
```

### Why Sliding Window Works — Hallucination Filter

| Scenario | Window Predictions | Majority Vote | Result |
|----------|-------------------|---------------|--------|
| **Real Fall** (person stays down) | Fall → Fall → Fall | **FALL** ✓ | Continuous fall confirms detection |
| **Bending** (picks up object) | Fall → No-Fall → No-Fall | **NO_FALL** ✓ | Recovery cancels false alarm |
| **Stumble + Recovery** | No-Fall → Fall → No-Fall | **NO_FALL** ✓ | Transient event filtered |
| **Slow gradual fall** | No-Fall → Fall → Fall | **FALL** ✓ | Persistent change detected |
| **Sleeping** (already lying) | No-Fall → No-Fall → No-Fall | **NO_FALL** ✓ | No transition = no fall |

**Key Insight**: A real fall is a **persistent state change** — the person goes down and stays down. False positives (bending, stumbling) are **transient** — the person recovers. Majority vote naturally distinguishes these cases.

---

## Part A: Prompt Engineering (4 Strategies)

We implement four distinct prompting strategies, progressively building sophistication:

### Strategy 1: Zero-Shot Prompting

Direct classification without examples or reasoning structure.

```python
ZERO_SHOT_PROMPT = """
You are a fall detection system analyzing pose data from video frames.

Given the following 3-frame temporal window with pose features:
{window_data}

Determine if this represents a FALL or NO_FALL.
Output JSON: {"label": "fall" or "no_fall", "confidence": 0.0-1.0}
"""
```

**Characteristics**:
- No examples provided
- Relies entirely on LLM's pretrained knowledge
- Fast but may lack domain specificity
- Baseline for comparison

---

### Strategy 2: Few-Shot Prompting

Provides exemplar cases to guide the model's reasoning pattern.

```python
FEW_SHOT_PROMPT = """
You are analyzing pose data for fall detection. Here are example cases:

EXAMPLE 1 - FALL:
Input: hip_y changes from 0.20 to 0.65, motion_score spikes to 1.2
Output: {"label": "fall", "confidence": 0.95, "reason": "Rapid downward movement with high velocity indicates uncontrolled fall"}

EXAMPLE 2 - NO_FALL (Bending):
Input: hip_y changes from 0.25 to 0.45, motion_score is 0.3, then recovers
Output: {"label": "no_fall", "confidence": 0.88, "reason": "Controlled movement with recovery indicates intentional bending"}

EXAMPLE 3 - NO_FALL (Already lying):
Input: hip_y starts at 0.70 and stays at 0.70, motion_score is 0.05
Output: {"label": "no_fall", "confidence": 0.92, "reason": "No transition detected - person was already in lying position"}

Now analyze this window:
{window_data}

Output JSON with label, confidence, and reason.
"""
```

**Characteristics**:
- 3-5 curated examples covering edge cases
- Demonstrates expected output format
- Guides reasoning by analogy
- Better handling of ambiguous cases

---

### Strategy 3: Chain-of-Thought (CoT) Prompting

Forces structured step-by-step reasoning before conclusion.

```python
COT_PROMPT = """
You are an expert fall detection system. Analyze this 3-frame temporal window using structured reasoning.

Window Data:
{window_data}

Video Context:
{video_context}

Follow these reasoning steps:

STEP 1 - INITIAL STATE CHECK:
Was the person already lying down at video start? Check initial_state.hip_y.
If hip_y > 0.5 initially → person may already be lying (sleeping, not falling)

STEP 2 - MOTION TRAJECTORY ANALYSIS:
Track how hip_y, shoulder_y, nose_y change across the 3 frames.
Falling = consistent downward trajectory (values increasing toward 1.0)

STEP 3 - VELOCITY ASSESSMENT:
Check motion_score values. 
Sudden spike (> 0.8) suggests rapid uncontrolled movement.
Gradual change (< 0.3) suggests controlled movement.

STEP 4 - POSTURE CHANGE ANALYSIS:
Check torso_vertical_diff across frames.
Large → person upright. Small → person horizontal/fallen.
Falling = torso_vertical_diff decreasing significantly.

STEP 5 - POSE GAP CONSIDERATION:
Check frame_gap_from_previous values.
Large gaps may indicate the person was in unusual position (MediaPipe couldn't detect).
Factor uncertainty into confidence score.

STEP 6 - FINAL DECISION:
Synthesize all evidence. A fall requires:
- Person started upright (low hip_y initially)
- Rapid downward transition occurred
- Person ended in fallen position
- No recovery within the window

Output JSON:
{
  "label": "fall" or "no_fall",
  "confidence": 0.0-1.0,
  "reasoning": {
    "initial_state_check": "...",
    "motion_trajectory": "...",
    "velocity_analysis": "...",
    "posture_change": "...",
    "pose_gaps": "...",
    "decision_basis": "..."
  },
  "reason": "One-sentence summary"
}
"""
```

**Characteristics**:
- Explicit reasoning steps
- Feature-grounded analysis
- Produces structured explanations
- Foundation for Explainable AI

---

### Strategy 4: Self-Consistency Prompting

Multiple independent reasoning paths with aggregated consensus.

```python
SELF_CONSISTENCY_PROMPT = """
You are a fall detection expert. Analyze this window THREE different ways, 
then provide a final consensus decision.

Window Data:
{window_data}

ANALYSIS PATH A - Motion-Focused:
Focus primarily on motion_score and velocity changes.
{reasoning_a}

ANALYSIS PATH B - Posture-Focused:
Focus primarily on body position changes (hip_y, torso_vertical_diff).
{reasoning_b}

ANALYSIS PATH C - Temporal-Focused:
Focus on the sequence of changes across frames and video_context.
{reasoning_c}

CONSENSUS:
Based on all three analysis paths, what is your final decision?
If paths disagree, explain why and use majority reasoning.

Output JSON:
{
  "label": "fall" or "no_fall",
  "confidence": 0.0-1.0,
  "path_a_verdict": "fall/no_fall",
  "path_b_verdict": "fall/no_fall", 
  "path_c_verdict": "fall/no_fall",
  "consensus_reasoning": "...",
  "reason": "Final one-sentence explanation"
}
"""
```

**Characteristics**:
- Multiple reasoning perspectives
- Built-in self-verification
- Higher reliability through consensus
- Reduces single-path reasoning errors

---

### Prompt Engineering Comparison Matrix

| Strategy | Reasoning Depth | Explainability | Reliability | API Calls | Use Case |
|----------|----------------|----------------|-------------|-----------|----------|
| Zero-Shot | Low | Minimal | Baseline | 1 | Quick baseline |
| Few-Shot | Medium | Limited | Good | 1 | Standard classification |
| Chain-of-Thought | High | **Excellent** | Very Good | 1 | Explainable decisions |
| Self-Consistency | Very High | Excellent | **Highest** | 3 | Critical decisions |

---

## Part B: Retrieval-Augmented Generation (RAG) Pipeline

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          KNOWLEDGE BASE                  │
│                                         │
│  1. Fall Case Library                   │
│     - Labeled fall/no_fall examples     │
│     - Pose feature patterns             │
│     - Ground truth explanations         │
│                                         │
│  2. Edge Case Documentation             │
│     - Bending scenarios                 │
│     - Sleeping detection rules          │
│     - Occlusion handling                │
│     - Stumble vs fall patterns          │
│                                         │
│  3. Domain Rules                        │
│     - Clinical fall definitions         │
│     - Pose threshold guidelines         │
│     - Confidence calibration rules      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         EMBEDDING & INDEXING             │
│                                          │
│  • Embedding Model: text-embedding-3-small
│  • Vector Store: FAISS                   │
│  • Chunk Size: 512 tokens                │
│  • Overlap: 50 tokens                    │
│  • Index Type: IVF with cosine similarity│
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│            RETRIEVER                     │
│                                          │
│  Input: Current window features          │
│  Process:                                │
│    1. Embed window as query vector       │
│    2. Similarity search in FAISS         │
│    3. Return top-k (k=3) similar cases   │
│  Output: Retrieved context documents     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         PROMPT FUSION                    │
│                                          │
│  Combine:                                │
│    • Retrieved similar cases             │
│    • Current window pose data            │
│    • CoT reasoning template              │
│    • Domain rules snippet                │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         LLM REASONING                    │
│                                          │
│  • Model: GPT-4o-mini                    │
│  • Input: Augmented prompt               │
│  • Output: Grounded prediction +         │
│            explanation                   │
└──────────────────────────────────────────┘
```

### RAG Implementation Details

#### Knowledge Base Construction

```python
# Knowledge Base Categories
KNOWLEDGE_BASE = {
    "fall_exemplars": [
        {
            "id": "fall_001",
            "description": "Classic forward fall - person trips and falls forward",
            "features": {
                "initial_hip_y": 0.22,
                "final_hip_y": 0.68,
                "max_motion_score": 1.15,
                "torso_change": "vertical to horizontal"
            },
            "label": "fall",
            "explanation": "Rapid uncontrolled descent from standing position..."
        },
        # ... more fall examples
    ],
    
    "no_fall_exemplars": [
        {
            "id": "bend_001", 
            "description": "Person bending to pick up object",
            "features": {
                "initial_hip_y": 0.25,
                "mid_hip_y": 0.45,
                "final_hip_y": 0.26,  # Returns to upright
                "max_motion_score": 0.35
            },
            "label": "no_fall",
            "explanation": "Controlled bending with recovery to standing..."
        },
        # ... more no_fall examples
    ],
    
    "edge_case_rules": [
        {
            "scenario": "sleeping_detection",
            "rule": "If initial_hip_y > 0.5 AND no significant change, person was already lying",
            "action": "Classify as NO_FALL regardless of final position"
        },
        {
            "scenario": "occlusion_handling",
            "rule": "If pose_detection_rate < 0.7, reduce confidence proportionally",
            "action": "Flag for human review if confidence < 0.6"
        }
    ],
    
    "clinical_definitions": [
        {
            "term": "fall",
            "definition": "An unintentional descent to the ground or lower level",
            "key_indicators": ["loss of balance", "rapid descent", "no controlled recovery"]
        }
    ]
}
```

#### RAG-Enhanced Prompt Template

```python
RAG_COT_PROMPT = """
You are an expert fall detection system enhanced with domain knowledge.

=== RETRIEVED SIMILAR CASES ===
{retrieved_cases}

=== DOMAIN RULES ===
{domain_rules}

=== CURRENT WINDOW DATA ===
{window_data}

=== VIDEO CONTEXT ===
{video_context}

Instructions:
1. Compare the current window to the retrieved similar cases
2. Apply relevant domain rules to handle edge cases
3. Use Chain-of-Thought reasoning for your analysis
4. Cite which retrieved case(s) most influenced your decision

Follow these reasoning steps:
[Same CoT steps as Strategy 3]

Output JSON:
{
  "label": "fall" or "no_fall",
  "confidence": 0.0-1.0,
  "retrieved_case_used": "case_id that most influenced decision",
  "domain_rule_applied": "rule name if applicable",
  "reasoning": {
    "initial_state_check": "...",
    "motion_trajectory": "...",
    "velocity_analysis": "...",
    "posture_change": "...",
    "similarity_to_retrieved": "How this compares to retrieved cases",
    "decision_basis": "..."
  },
  "reason": "One-sentence grounded explanation"
}
"""
```

### RAG Benefits for Fall Detection

| Aspect | Without RAG | With RAG |
|--------|-------------|----------|
| Edge case handling | May hallucinate | Grounded in documented cases |
| Consistency | Variable reasoning | Pattern-matched to exemplars |
| Explainability | Generic explanations | Case-referenced explanations |
| Sleeping vs falling | May confuse | Rule-guided distinction |
| Confidence calibration | Overconfident | Calibrated to similar cases |

---

## Part C: Fine-Tuning — LoRA for Explainable AI

### Why LoRA Fine-Tuning?

We use **LoRA (Low-Rank Adaptation)** to create a **domain-specialized Explainable AI model** that:

1. **Generates structured explanations** — not just labels
2. **Embeds domain expertise** — fall detection reasoning patterns
3. **Runs locally** — no API dependency, zero inference cost
4. **Maintains explainability** — every prediction includes clinical-grade reasoning

### LoRA Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LoRA FINE-TUNING PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                    TRAINING DATA GENERATION
                    ════════════════════════
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
│  Window JSONs    │───▶│   GPT-4o-mini    │───▶│  Structured CoT Output   │
│  (239 windows)   │    │   with CoT       │    │  (label + reasoning)     │
└──────────────────┘    │   Prompting      │    └────────────┬─────────────┘
                        └──────────────────┘                 │
                                                             ▼
                                               ┌──────────────────────────┐
                                               │   Manual Review &        │
                                               │   Correction             │
                                               │   (quality assurance)    │
                                               └────────────┬─────────────┘
                                                            │
                    LoRA FINE-TUNING                        ▼
                    ════════════════        ┌──────────────────────────────┐
┌──────────────────┐                        │  Training Data Pairs         │
│  Base Model      │                        │  ┌────────────────────────┐  │
│  (Phi-3-mini or  │───────────────────────▶│  │ instruction: "Analyze  │  │
│   Mistral-7B)    │                        │  │   this window..."      │  │
└──────────────────┘                        │  │ input: <window JSON>   │  │
        │                                   │  │ output: <structured    │  │
        ▼                                   │  │   reasoning + label>   │  │
┌──────────────────┐                        │  └────────────────────────┘  │
│  LoRA Adapters   │                        └──────────────────────────────┘
│  (r=16, α=32)    │                                       │
│                  │◀──────────────────────────────────────┘
│  Target modules: │
│  q_proj, v_proj, │
│  k_proj, o_proj  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPLAINABLE AI INFERENCE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
│  New Window      │───▶│  LoRA-Tuned      │───▶│  Explainable Output      │
│  (test data)     │    │  Local LLM       │    │                          │
└──────────────────┘    └──────────────────┘    │  {                       │
                                                │    "label": "fall",      │
                                                │    "confidence": 0.94,   │
                                                │    "reasoning": {        │
                                                │      "initial_state":    │
                                                │        "Upright...",     │
                                                │      "motion": "Rapid    │
                                                │        descent...",      │
                                                │      ...                 │
                                                │    },                    │
                                                │    "reason": "..."       │
                                                │  }                       │
                                                └──────────────────────────┘
```

### LoRA Training Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch

# Base Model Selection
model_name = "microsoft/phi-3-mini-4k-instruct"  # 3.8B params
# Alternative: "mistralai/Mistral-7B-Instruct-v0.2"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,                          # Rank of update matrices
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.05,             # Dropout for regularization
    target_modules=[               # Attention layers to adapt
        "q_proj", 
        "v_proj", 
        "k_proj", 
        "o_proj"
    ],
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

# Apply LoRA
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# Output: trainable params: ~4M (~0.1% of 3.8B)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./models/lora_fall_reasoner",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)
```

### Training Data Format

Each training example teaches the model to produce explainable reasoning:

```json
{
  "instruction": "You are a fall detection expert. Analyze this 3-frame temporal window and provide a detailed explanation of your reasoning.",
  
  "input": {
    "sequence": {
      "frame_1": {"hip_y": 0.198, "shoulder_y": 0.447, "motion_score": 0.0},
      "frame_2": {"hip_y": 0.412, "shoulder_y": 0.589, "motion_score": 0.87},
      "frame_3": {"hip_y": 0.616, "shoulder_y": 0.775, "motion_score": 0.32}
    },
    "video_context": {
      "initial_state": {"hip_y": 0.198},
      "final_state": {"hip_y": 0.616},
      "has_significant_transition": true
    }
  },
  
  "output": {
    "label": "fall",
    "confidence": 0.94,
    "reasoning": {
      "initial_state_check": "Person was upright at video start (hip_y=0.198, well below midline)",
      "motion_trajectory": "Clear downward trajectory: hip_y increased from 0.198 → 0.412 → 0.616",
      "velocity_analysis": "Motion score spiked to 0.87 in frame 2, indicating rapid uncontrolled movement",
      "posture_change": "Body transitioned from vertical to near-horizontal position",
      "pose_gaps": "No significant gaps - continuous pose detection",
      "decision_basis": "All indicators consistent with uncontrolled fall: upright start, rapid descent, no recovery"
    },
    "reason": "Rapid uncontrolled descent from standing position with no recovery indicates a fall"
  }
}
```

### Explainable AI Output Format

The LoRA fine-tuned model generates **clinical-grade explanations**:

```
═══════════════════════════════════════════════════════════════════
VIDEO: fall_07_cam1 — VERDICT: FALL (majority: 2/3 windows)
═══════════════════════════════════════════════════════════════════

WINDOW 1 [frames 0-90]: NO_FALL (confidence: 0.85)
────────────────────────────────────────────────────
Initial State:  Person standing upright (hip_y=0.21)
Motion:         Stable, low motion_score (0.05)
Posture:        Torso vertical (diff=0.25)
Assessment:     Normal standing activity, no fall indicators

WINDOW 2 [frames 90-180]: FALL (confidence: 0.94)
────────────────────────────────────────────────────
Initial State:  Was upright at window start
Motion:         Dramatic descent — hip_y surged 0.21 → 0.62
Velocity:       Motion score spiked to 1.05 (rapid, uncontrolled)
Posture:        Torso collapsed from 0.25 to 0.08 (horizontal)
Assessment:     Clear fall event — sudden uncontrolled descent

WINDOW 3 [frames 180-270]: FALL (confidence: 0.91)
────────────────────────────────────────────────────
Initial State:  Person in fallen position (hip_y=0.65)
Motion:         Minimal movement (motion_score=0.03)
Posture:        Remains horizontal (torso_diff=0.06)
Assessment:     Person remains on ground — no recovery detected

═══════════════════════════════════════════════════════════════════
CLINICAL NARRATIVE:
The individual was initially standing (Window 1), experienced a 
sudden uncontrolled descent characterized by rapid downward movement 
and high velocity (Window 2), and remained on the ground without 
recovery (Window 3). Pattern: CONFIRMED FALL requiring attention.
═══════════════════════════════════════════════════════════════════
```

### Why This Is Explainable AI (XAI)

| XAI Criterion | How We Achieve It |
|---------------|-------------------|
| **Transparency** | Every prediction includes step-by-step reasoning |
| **Feature Attribution** | Explanations cite specific pose features (hip_y, motion_score) |
| **Temporal Causality** | Reasoning tracks progression across frames |
| **Human-Readable** | Clinical narrative format for healthcare workers |
| **Auditable** | Complete evidence chain from raw data to verdict |
| **Uncertainty Quantification** | Confidence scores calibrated to evidence quality |

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    DATA LAYER                    REASONING LAYER                 OUTPUT LAYER
    ══════════                    ═══════════════                 ════════════

┌──────────────┐            ┌─────────────────────┐
│ Video Input  │            │   PROMPT STRATEGIES │
│ (URFD,       │            │                     │
│  GMNCSA24)   │            │  ┌───────────────┐  │
└──────┬───────┘            │  │  Zero-Shot    │  │
       │                    │  └───────────────┘  │
       ▼                    │  ┌───────────────┐  │
┌──────────────┐            │  │  Few-Shot     │  │
│ MediaPipe    │            │  └───────────────┘  │
│ Pose Extract │            │  ┌───────────────┐  │        ┌──────────────────┐
└──────┬───────┘            │  │  CoT          │  │        │  PREDICTIONS     │
       │                    │  └───────────────┘  │        │                  │
       ▼                    │  ┌───────────────┐  │        │  Window-level    │
┌──────────────┐            │  │Self-Consistency│ │        │  predictions     │
│ Sliding      │───────────▶│  └───────────────┘  │───────▶│        │         │
│ Windows      │            └─────────────────────┘        │        ▼         │
│ (size=3,     │                      │                    │  Majority Vote   │
│  stride=2)   │                      │                    │        │         │
└──────────────┘                      │                    │        ▼         │
                                      │                    │  Video-level     │
                            ┌─────────▼─────────┐          │  verdict         │
                            │      RAG          │          └────────┬─────────┘
                            │                   │                   │
                            │  Knowledge Base   │                   │
                            │  + FAISS Index    │                   ▼
                            │  + Retriever      │          ┌──────────────────┐
                            └─────────┬─────────┘          │  EXPLAINABILITY  │
                                      │                    │                  │
                                      │                    │  • Reasoning     │
                            ┌─────────▼─────────┐          │    traces        │
                            │   LoRA FINE-TUNED │          │  • Feature       │
                            │   LOCAL MODEL     │          │    evidence      │
                            │                   │          │  • Clinical      │
                            │  Phi-3 / Mistral  │─────────▶│    narrative     │
                            │  + LoRA adapters  │          │  • Confidence    │
                            │                   │          │    scores        │
                            │  Domain Expert    │          └──────────────────┘
                            │  Explainable AI   │
                            └───────────────────┘
```

---

## Evaluation Framework

### Metrics

| Metric | Priority | Description |
|--------|----------|-------------|
| **Recall** | Critical | Missing a real fall is dangerous |
| **FPR (False Positive Rate)** | High | Alarm fatigue from false positives |
| **F1 Score** | Medium | Balance between precision and recall |
| **Accuracy** | Standard | Overall correctness |

### Dual-Level Evaluation

- **Window-level**: Accuracy of individual 3-frame window predictions
- **Video-level**: Accuracy of majority-vote aggregated final verdict

### Gen AI Progression Comparison Table

| Strategy | Gen AI Technique | Window Acc | Video Acc | Recall | F1 | Explainable? |
|----------|-----------------|------------|-----------|--------|----|--------------| 
| Zero-Shot | Basic Prompting | — | — | — | — | Minimal |
| Few-Shot | Example-based Prompting | — | — | — | — | Limited |
| CoT | Structured Prompting | — | — | — | — | ✓ Good |
| Self-Consistency | Multi-path Prompting | — | — | — | — | ✓ Good |
| RAG + CoT | Retrieval + Prompting | — | — | — | — | ✓ Grounded |
| **LoRA Fine-Tuned** | **PEFT Fine-Tuning** | — | — | — | — | **✓ Expert-level** |

---

## Project Timeline

| Phase | Tasks | Status |
|-------|-------|--------|
| **Phase 1** | Data preparation, pose extraction, initial baselines | ✅ Complete |
| **Phase 2** | Sliding window pipeline, 4 prompting strategies | ✅ Complete |
| **Phase 3** | RAG implementation, LoRA fine-tuning, evaluation, report | 🔄 In Progress |

---

## Data Pipeline (pipeline/)

The data preparation pipeline is implemented in `pipeline/` with modular, reusable scripts:

### Pipeline Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `config.py` | Configuration settings | Feature thresholds, paths |
| `step1_create_video_manifest.py` | Scan videos, assign labels, create splits | `manifests/video_manifest.csv` |
| `step2_extract_frames.py` | Extract frames from videos (OpenCV) | `frames/{split}/{label}/{video_id}/` |
| `step3_extract_pose_features.py` | Extract enhanced pose features (MediaPipe) | `pose_features/{split}/{label}/{video_id}/` |
| `step4_create_sliding_windows.py` | Generate 3-frame windows with analysis | `sliding_windows/{split}/{label}/{video_id}/` |
| `run_pipeline.py` | Orchestrate all steps | Complete data preparation |

### Running the Pipeline

```bash
# Install dependencies
cd pipeline
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py --all

# Run specific steps
python run_pipeline.py --step 1 2 3 4
python run_pipeline.py --from-step 2

# Process specific split
python run_pipeline.py --all --split train
```

### Pipeline Data Flow

```
STEP 1: Video Manifest
──────────────────────
Videos (URFD + Voxel + GMNCSA24)
        │
        ▼
Scan directories → Assign labels → Create train/val/test splits
        │
        ▼
manifests/video_manifest.csv (prevents data leakage)


STEP 2: Frame Extraction
────────────────────────
Videos → OpenCV → Sample every N frames → Save JPG
        │
        ▼
frames/{split}/{label}/{video_id}/frame_000000.jpg


STEP 3: Pose Feature Extraction (ENHANCED)
──────────────────────────────────────────
Frames → MediaPipe → Extract landmarks → Compute derivatives
        │
        ├── Position: hip_y, shoulder_y, nose_y
        ├── Velocity: velocity_hip_y, velocity_magnitude
        ├── Acceleration: acceleration_hip_y, jerk
        ├── Posture: posture_state, body_angle_degrees
        └── Fall flags: is_rapid_descent, is_on_ground
        │
        ▼
pose_features/{split}/{label}/{video_id}/pose_features.json


STEP 4: Sliding Windows
───────────────────────
Pose features → 3-frame windows (stride=2) → Window analysis
        │
        ├── Sequence: frame_1, frame_2, frame_3 (full features)
        ├── Window analysis: trajectory, fall indicators
        ├── Video context: initial/final state, recovery detection
        └── LLM descriptions: human-readable text
        │
        ▼
sliding_windows/{split}/{label}/{video_id}/{window_id}.json
```

---

## GenAI Script Reference

| Script | Purpose | Gen AI Component |
|--------|---------|------------------|
| `zero_shot_windowed.py` | Zero-shot baseline | Prompt Strategy 1 |
| `few_shot_windowed.py` | Few-shot baseline | Prompt Strategy 2 |
| `cot_windowed.py` | Chain-of-Thought baseline | Prompt Strategy 3 |
| `self_consistency_windowed.py` | Self-consistency baseline | Prompt Strategy 4 |
| `rag_pipeline.py` | RAG-enhanced inference | RAG Pipeline |
| `lora_prepare_data.py` | Generate LoRA training data | Fine-Tuning |
| `lora_train.py` | LoRA fine-tune local LLM | Fine-Tuning (PEFT) |
| `lora_inference.py` | Run LoRA model on windows | Explainable AI |
| `evaluate_all.py` | Compare all strategies | Evaluation |

---

## Team Responsibilities

| Member | Responsibilities |
|--------|-----------------|
| **Ram** | Sliding window pipeline, LoRA training, robustness testing |
| **Harshitha** | RAG pipeline, knowledge base, retrieval evaluation |
| **Shristi** | Prompting strategies, LoRA data prep, explainability analysis |

---

## Novelty Statement

> **What makes this novel**: We present the first **explainable, temporally self-correcting, domain-specialized Gen AI system for fall detection**. Our innovations include: (1) **Sliding window temporal consensus** that naturally filters LLM hallucinations; (2) **Four prompting strategies** demonstrating progressive sophistication in fall reasoning; (3) **Full RAG pipeline** grounding predictions in documented fall cases and domain rules; and (4) **LoRA fine-tuning** creating a local, zero-cost Explainable AI model that generates clinical-grade reasoning for every prediction. Unlike black-box deep learning approaches, our system provides transparent, auditable explanations that healthcare workers can understand and trust.

---

## Requirements Checklist

```
[✓] DATA PIPELINE — Enhanced feature extraction:
    [✓] Video manifest with labels and splits (data leakage prevention)
    [✓] Frame extraction with OpenCV
    [✓] Pose extraction with MediaPipe
    [✓] Enhanced features: position, velocity, acceleration, posture state
    [✓] Sliding window generation with window-level analysis
    [✓] LLM-friendly descriptions included

[ ] A. Prompt Engineering — 4 strategies to implement:
    [ ] Zero-Shot Prompting
    [ ] Few-Shot Prompting  
    [ ] Chain-of-Thought (CoT) Prompting
    [ ] Self-Consistency Prompting

[ ] B. RAG Pipeline — To implement:
    [ ] Knowledge base construction (fall cases, edge cases, rules)
    [ ] FAISS vector indexing
    [ ] Retrieval integration
    [ ] Prompt fusion with CoT

[ ] C. Fine-Tuning (PEFT) — LoRA to implement:
    [ ] Training data generation from CoT outputs
    [ ] LoRA configuration (Phi-3/Mistral base)
    [ ] Explainable AI output format
    [ ] Domain-specialized reasoning
```

---

## What's Next

1. **Run Data Pipeline** → Generate sliding windows with enhanced features
2. **Implement Prompting Strategies** → Zero-Shot, Few-Shot, CoT, Self-Consistency
3. **Build RAG Pipeline** → Knowledge base with fall cases and rules
4. **LoRA Fine-Tuning** → Domain-specialized Explainable AI model
5. **Evaluation** → Compare all techniques on same data
