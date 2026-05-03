"""
LoRA Fine-Tuning for Fall Detection
====================================

Fine-tunes GPT-4o-mini using OpenAI's fine-tuning API.
This should achieve 85%+ accuracy by training directly on our data.

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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from zero_shot import (
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "lora_finetune"
TRAINING_DIR = Path(__file__).parent.parent / "lora_training"

BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Base model for fine-tuning


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
    if hip_y > 0.7:
        parts.append("body very low")
    elif hip_y > 0.55:
        parts.append("body mid-height")
    else:
        parts.append("body standing height")
    
    # Orientation  
    if body_angle < 25:
        parts.append("horizontal/lying")
    elif body_angle < 50:
        parts.append("significantly tilted")
    else:
        parts.append("upright")
    
    # Movement
    if velocity and abs(velocity) > 0.12:
        parts.append("moving rapidly downward" if velocity > 0 else "moving rapidly upward")
    elif velocity and abs(velocity) > 0.05:
        parts.append("moderate movement")
    else:
        parts.append("stable/minimal movement")
    
    # Critical alerts
    alerts = []
    if rapid:
        alerts.append("RAPID DESCENT DETECTED")
    if on_ground:
        alerts.append("PERSON ON GROUND")
    if horizontal:
        alerts.append("BODY HORIZONTAL")
    
    if alerts:
        parts.append("ALERTS: " + ", ".join(alerts))
    
    return "; ".join(parts)


def window_to_training_text(window_data: Dict) -> str:
    """Convert window to training text."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    text = "Analyze this 3-frame sequence for fall detection:\n\n"
    
    for i, key in enumerate(["frame_1", "frame_2", "frame_3"], 1):
        frame = sequence.get(key, {})
        text += f"Frame {i}: {interpret_features_compact(frame)}\n"
    
    trajectory = analysis.get("trajectory_direction", "unknown")
    max_vel = analysis.get("max_velocity", 0) or 0
    spike = analysis.get("has_velocity_spike", False)
    
    text += f"\nOverall: trajectory={trajectory}, max_velocity={max_vel:.3f}, velocity_spike={spike}"
    
    return text


def prepare_training_data():
    """Generate training data in OpenAI fine-tuning format."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    system_message = """You are a fall detection AI analyzing pose sequences from video frames.

Your task: Determine if the person is FALLING or NOT FALLING.

FALL indicators:
- Rapid downward movement with velocity spike
- Body transitioning from upright to horizontal
- Person ending up on ground level
- ALERTS indicating rapid descent, ground contact, or horizontal position

NO_FALL indicators:
- Slow, controlled movements
- Body remains upright or only slightly tilted
- No velocity spikes
- No critical alerts

Respond with ONLY: "FALL" or "NO_FALL" followed by a brief explanation."""

    training_examples = []
    
    for label in ["fall", "no_fall"]:
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
                
                # Create appropriate response based on label
                if label == "fall":
                    assistant_message = "FALL - The sequence shows rapid downward movement with the person transitioning to a horizontal position near ground level, indicating an uncontrolled fall."
                else:
                    assistant_message = "NO_FALL - The sequence shows controlled movement with the person maintaining stability. No indicators of uncontrolled falling detected."
                
                training_examples.append({
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message}
                    ]
                })
    
    # Shuffle and split
    random.seed(42)
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
    train_falls = sum(1 for ex in train_data if "FALL -" in ex["messages"][2]["content"] and "NO_FALL" not in ex["messages"][2]["content"])
    train_nofalls = len(train_data) - train_falls
    print(f"  Train falls: {train_falls}, no-falls: {train_nofalls}")
    
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


def start_fine_tuning(client: OpenAI, train_file_id: str = None, val_file_id: str = None):
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
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training file: {train_file_id}")
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=BASE_MODEL,
        hyperparameters={
            "n_epochs": 3,
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
            "base_model": BASE_MODEL,
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
        
        video_pred = aggregate_video_predictions(window_preds, "majority")
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
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--upload", action="store_true", help="Upload files to OpenAI")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None
    
    if args.prepare:
        prepare_training_data()
    elif args.upload:
        upload_training_files(client)
    elif args.train:
        start_fine_tuning(client)
    elif args.status:
        check_job_status(client)
    elif args.evaluate:
        evaluate_finetuned_model(client, split=args.split, sample_size=args.sample)
    else:
        print("Specify --prepare, --upload, --train, --status, or --evaluate")


if __name__ == "__main__":
    main()
