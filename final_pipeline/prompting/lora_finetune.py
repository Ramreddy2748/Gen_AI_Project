"""
LoRA/SFT Fine-Tuning for Fall Detection
=======================================

Fine-tunes GPT-4o-mini using OpenAI's supervised fine-tuning API.
The project refers to this branch as LoRA fine-tuning; OpenAI exposes it as
supervised fine-tuning, while the training goal is the same: adapt the base LLM
to this fall/no-fall pose-window classification task.

Steps:
1. Generate training data in JSONL format
2. Upload to OpenAI
3. Create fine-tuning job
4. Evaluate fine-tuned model

Usage:
    python lora_finetune.py --prepare    # Generate training data
    python lora_finetune.py --upload     # Upload to OpenAI
    python lora_finetune.py --train      # Start fine-tuning
    python lora_finetune.py --evaluate   # Evaluate model
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from zero_shot import (
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "lora_finetune"
TRAINING_DIR = Path(__file__).parent.parent / "lora_training"

BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Base model for fine-tuning

SYSTEM_MESSAGE = """You are a medical fall detection assistant analyzing human pose data from video frames.

Your task is to classify each 3-frame sequence as a fall or normal activity.

DEFINITIVE FALL (any 1 alone is sufficient):
- posture = "fallen" in any frame
- on_ground flag = True in any frame
- horizontal flag = True (body angle < 35 degrees) in any frame

PROBABLE FALL (classify FALL only if 2 or more apply together):
- rapid_descent flag = True
- velocity > 0.08 (fast downward movement)
- body angle < 50 degrees (significantly tilted)
- descending trajectory across all 3 frames
- hip_y > 0.60 (body lowering toward ground)

NORMAL ACTIVITY (NO_FALL when none of the above apply):
- Upright posture throughout (angle > 60 degrees, posture = "upright")
- Slow controlled movement (velocity < 0.05)
- No fall flags triggered in any frame
- Stable or ascending trajectory

NOTE: Bending, sitting, or crouching may show low hip_y or tilt — require UNCONTROLLED descent with multiple indicators before classifying as FALL.

Weigh velocity, posture, and trajectory collectively before deciding.
Respond with ONLY: "FALL" or "NO_FALL" followed by a brief explanation."""

FALL_RESPONSES = [
    "FALL - The sequence shows rapid downward movement with velocity spike. The person's body is transitioning toward a horizontal position near ground level.",
    "FALL - Fall indicators are present: the person is descending rapidly with loss of upright posture. Ground-level position or horizontal body angle detected.",
    "FALL - Body angle and position indicate a fall event. Significant downward velocity and postural destabilization observed across the sequence.",
    "FALL - The person appears to be losing balance and descending toward the ground with rapid, uncontrolled movement.",
    "FALL - The sequence shows the person transitioning from standing toward a fallen position. Velocity spike or ground contact detected.",
    "FALL - Body orientation is becoming horizontal with rapid descent velocity. One or more fall indicators triggered in this sequence.",
    "FALL - Uncontrolled downward movement is detected. Body position and trajectory are consistent with a fall event.",
    "FALL - The person shows a fall trajectory: rapid downward velocity, body angle deviation, or ground-level positioning detected.",
    "FALL - Posture state indicates a transitioning or fallen position. Combined with downward trajectory, this sequence is classified as a fall.",
    "FALL - The fall likelihood score and motion features indicate a high-risk fall event. Body is descending and losing stability.",
    "FALL - Any single fall indicator present in this sequence is sufficient. Rapid descent, ground contact, or horizontal position is detected.",
    "FALL - The motion pattern across these three frames is consistent with an uncontrolled fall. When in doubt, this is classified as FALL for safety.",
]

NO_FALL_RESPONSES = [
    "NO_FALL - The person maintains stable upright posture throughout all three frames. Movement is slow and controlled with no fall indicators present.",
    "NO_FALL - Body remains upright with minimal velocity. No rapid descent, ground contact, or horizontal position detected across any frame.",
    "NO_FALL - Controlled movement pattern with stable posture maintained. All fall criteria are absent and fall likelihood score is low.",
    "NO_FALL - The person is standing or moving normally. No velocity spikes, ground contact, or horizontal orientation detected in any frame.",
    "NO_FALL - Stable body position throughout the sequence. Movement is within normal parameters and no fall indicators were triggered.",
    "NO_FALL - No fall detected. Body angle, position, and velocity are all within normal ranges. Posture remains consistently upright.",
]


def interpret_features_compact(frame: Dict) -> str:
    """Create compact feature description."""
    features = frame.get("features", {})
    enhanced = frame.get("enhanced", {})
    
    if not frame.get("pose_detected", False):
        return "person not visible"
    
    hip_y = features.get("hip_y", 0)
    body_angle = features.get("body_angle_degrees", 90)
    velocity = enhanced.get("velocity_hip_y", 0)
    posture = enhanced.get("posture_state", "unknown")
    on_ground = enhanced.get("is_on_ground", False)
    rapid = enhanced.get("is_rapid_descent", False)
    horizontal = enhanced.get("is_horizontal", False)
    
    parts = []
    
    # Position
    if hip_y > 0.65:
        parts.append("body very low / near ground")
    elif hip_y > 0.50:
        parts.append("body mid-height")
    else:
        parts.append("body standing height")

    # Orientation
    if body_angle < 35:
        parts.append("horizontal/lying")
    elif body_angle < 60:
        parts.append("significantly tilted")
    else:
        parts.append("upright")

    # Movement
    if velocity and abs(velocity) > 0.08:
        parts.append("moving rapidly downward" if velocity > 0 else "moving rapidly upward")
    elif velocity and abs(velocity) > 0.03:
        parts.append("moderate downward movement" if velocity > 0 else "moderate upward movement")
    else:
        parts.append("stable/minimal movement")
    
    # Fall alerts
    alerts = []
    if rapid:
        alerts.append("rapid descent detected")
    if on_ground:
        alerts.append("person on ground")
    if horizontal:
        alerts.append("body horizontal")

    if alerts:
        parts.append("alerts: " + ", ".join(alerts))
    
    return "; ".join(parts)


def window_to_training_text(window_data: Dict) -> str:
    """Convert window to training text."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})

    text = "Analyze this 3-frame sequence for fall detection:\n\n"

    for i, key in enumerate(["frame_1", "frame_2", "frame_3"], 1):
        frame = sequence.get(key, {})
        posture = frame.get("enhanced", {}).get("posture_state", "unknown")
        text += f"Frame {i}: {interpret_features_compact(frame)} | posture={posture}\n"

    trajectory  = analysis.get("trajectory_direction", "unknown")
    max_vel     = analysis.get("max_velocity", 0) or 0
    spike       = analysis.get("has_velocity_spike", False)
    fall_score  = analysis.get("fall_likelihood_score", None)

    score_str = f"{fall_score:.2f}" if fall_score is not None else "unknown"
    text += f"\nOverall: trajectory={trajectory}, max_velocity={max_vel:.3f}, velocity_spike={spike}, fall_likelihood_score={score_str}"

    # Explicit high-risk flag — gives the model a direct signal
    if fall_score is not None and fall_score >= 0.5:
        text += "\nNote: fall_likelihood_score >= 0.5 — high probability of fall event."
    elif fall_score is not None and fall_score >= 0.3:
        text += "\nNote: fall_likelihood_score >= 0.3 — moderate fall risk, classify carefully."

    return text


def prepare_training_data(balance_classes: bool = True):
    """Generate training data in OpenAI fine-tuning format."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    examples_by_label = defaultdict(list)
    
    labels = ["fall", "no_fall"]
    for label in labels:
        label_dir = WINDOWS_DIR / "train" / label
        if not label_dir.exists():
            continue
        
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            for window_file in video_dir.glob("*.json"):
                with open(window_file) as f:
                    data = json.load(f)
                
                user_message = window_to_training_text(data)
                
                # Use varied responses to prevent template memorization
                if label == "fall":
                    assistant_message = random.choice(FALL_RESPONSES)
                else:
                    assistant_message = random.choice(NO_FALL_RESPONSES)
                
                examples_by_label[label].append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message}
                    ]
                })

    missing_labels = [label for label in labels if not examples_by_label[label]]
    if missing_labels:
        raise RuntimeError(
            f"Missing training windows for {missing_labels} in {WINDOWS_DIR / 'train'}"
        )

    random.seed(42)
    fall_items = examples_by_label["fall"]
    nofall_items = examples_by_label["no_fall"]

    if balance_classes:
        # Target 75% FALL / 25% NO_FALL ratio
        # More aggressive FALL bias → model predicts FALL more liberally → higher recall
        nofall_count = len(nofall_items)
        fall_count = int(nofall_count * (75 / 25))  # 3x no_fall → 75/25 split
        sampled_fall = random.sample(fall_items, min(fall_count, len(fall_items)))
        training_examples = sampled_fall + nofall_items
    else:
        training_examples = fall_items + nofall_items
    
    # Shuffle and split
    random.shuffle(training_examples)
    
    # Use 90% for training, 10% for validation
    split_idx = int(len(training_examples) * 0.9)
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    # Save as JSONL
    train_path = TRAINING_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    val_path = TRAINING_DIR / "validation.jsonl"
    with open(val_path, "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Training data prepared:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    print(f"  Train file: {train_path}")
    print(f"  Val file: {val_path}")
    
    # Count labels
    train_counts = Counter(
        "no_fall" if ex["messages"][2]["content"].startswith("NO_FALL") else "fall"
        for ex in train_data
    )
    val_counts = Counter(
        "no_fall" if ex["messages"][2]["content"].startswith("NO_FALL") else "fall"
        for ex in val_data
    )
    print(f"  Balanced classes: {balance_classes}")
    print(f"  Train labels: {dict(train_counts)}")
    print(f"  Val labels: {dict(val_counts)}")
    
    return train_path, val_path


def upload_training_files(client: OpenAI):
    """Upload training files to OpenAI."""
    train_path = TRAINING_DIR / "train.jsonl"
    val_path = TRAINING_DIR / "validation.jsonl"
    
    if not train_path.exists():
        print("Training data not found. Run with --prepare first.")
        return None, None
    
    print("Uploading training file...")
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  Train file ID: {train_file.id}")
    
    print("Uploading validation file...")
    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  Val file ID: {val_file.id}")
    
    # Save file IDs
    with open(TRAINING_DIR / "file_ids.json", "w") as f:
        json.dump({
            "train_file_id": train_file.id,
            "val_file_id": val_file.id
        }, f, indent=2)
    
    return train_file.id, val_file.id


def require_client(client):
    """Fail early when an OpenAI API key is required."""
    if OpenAI is None:
        raise RuntimeError("Install the OpenAI SDK first: pip install openai")
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is required for upload, train, status, and evaluate.")


def start_fine_tuning(
    client: OpenAI,
    train_file_id: str = None,
    val_file_id: str = None,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
):
    """Start fine-tuning job."""
    if not train_file_id:
        ids_path = TRAINING_DIR / "file_ids.json"
        if ids_path.exists():
            with open(ids_path) as f:
                ids = json.load(f)
            train_file_id = ids["train_file_id"]
            val_file_id = ids.get("val_file_id")
        else:
            print("File IDs not found. Run with --upload first.")
            return None
    
    print(f"Starting fine-tuning job...")
    print(f"  Base model: {base_model}")
    print(f"  Training file: {train_file_id}")
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=base_model,
        hyperparameters={
            "n_epochs": epochs,
        },
        suffix="fall-detector"
    )
    
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    
    # Save job info
    with open(TRAINING_DIR / "job_info.json", "w") as f:
        json.dump({
            "job_id": job.id,
            "status": job.status,
            "base_model": base_model,
            "epochs": epochs,
        }, f, indent=2)
    
    return job.id


def check_job_status(client: OpenAI, job_id: str = None):
    """Check fine-tuning job status."""
    if not job_id:
        job_path = TRAINING_DIR / "job_info.json"
        if job_path.exists():
            with open(job_path) as f:
                job_id = json.load(f)["job_id"]
        else:
            print("Job info not found.")
            return None
    
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"Job Status:")
    print(f"  ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Model: {job.fine_tuned_model or 'Not ready'}")
    
    if job.fine_tuned_model:
        # Save model ID
        with open(TRAINING_DIR / "model_info.json", "w") as f:
            json.dump({
                "model_id": job.fine_tuned_model,
                "job_id": job.id,
            }, f, indent=2)
        print(f"\nFine-tuned model ready: {job.fine_tuned_model}")
    
    return job


def evaluate_finetuned_model(client: OpenAI, model_id: str = None, 
                             split: str = "test", sample_size: int = None):
    """Evaluate fine-tuned model."""
    if not model_id:
        model_path = TRAINING_DIR / "model_info.json"
        if model_path.exists():
            with open(model_path) as f:
                model_id = json.load(f)["model_id"]
        else:
            print("Model info not found. Check job status first.")
            return None
    
    print(f"Evaluating model: {model_id}")
    
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    
    window_ids = list(metadata.keys())
    if sample_size:
        random.seed(42)
        window_ids = random.sample(window_ids, min(sample_size, len(window_ids)))
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[metadata[wid]["video_id"]].append(wid)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    processed = 0
    total = len(window_ids)
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(Path(metadata[wid]["window_path"])) as f:
                data = json.load(f)
            
            user_message = window_to_training_text(data)
            
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                result = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error: {e}")
                result = "ERROR"
            
            # Parse
            result_upper = result.upper()
            if "NO_FALL" in result_upper or "NO FALL" in result_upper:
                pred = "no_fall"
            elif "FALL" in result_upper:
                pred = "fall"
            else:
                pred = "unknown"
            
            window_preds[wid] = pred
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
                "raw_response": result,
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{total}...")
            
            time.sleep(0.1)
        
        # 30% threshold: balances precision and recall for fine-tuned model
        fall_count = sum(1 for p in window_preds.values() if p == "fall")
        total_windows = len(window_preds)
        video_pred = "fall" if (total_windows > 0 and fall_count / total_windows >= 0.30) else "no_fall"
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
        })
    
    # Save
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Metrics
    print("\n" + "="*60)
    print("LORA FINE-TUNED MODEL RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({
            "window_level": w_metrics, 
            "video_level": v_metrics,
            "model": model_id
        }, f, indent=2)
    
    return w_metrics, v_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["prepare", "upload", "train", "status", "evaluate"],
        help="Alternative command style used in the project README",
    )
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--upload", action="store_true", help="Upload files to OpenAI")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--no-balance", action="store_true", help="Use all windows without class balancing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (OpenAI is not None and api_key) else None

    mode = args.mode
    if args.prepare:
        mode = "prepare"
    elif args.upload:
        mode = "upload"
    elif args.train:
        mode = "train"
    elif args.status:
        mode = "status"
    elif args.evaluate:
        mode = "evaluate"

    if mode == "prepare":
        prepare_training_data(balance_classes=not args.no_balance)
    elif mode == "upload":
        require_client(client)
        upload_training_files(client)
    elif mode == "train":
        require_client(client)
        start_fine_tuning(client, base_model=args.base_model, epochs=args.epochs)
    elif mode == "status":
        require_client(client)
        check_job_status(client)
    elif mode == "evaluate":
        require_client(client)
        evaluate_finetuned_model(client, split=args.split, sample_size=args.sample)
    else:
        print("Specify --mode prepare|upload|train|status|evaluate or use --prepare/--upload/--train/--status/--evaluate")


if __name__ == "__main__":
    main()
