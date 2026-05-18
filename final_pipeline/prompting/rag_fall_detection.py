"""
RAG (Retrieval-Augmented Generation) Fall Detection
=====================================================

Uses semantic similarity to retrieve relevant examples from training data
and provides them as context to improve LLM predictions.

Flow:
1. Build knowledge base from TRAIN set (1,305 windows)
2. For each query, retrieve top-K similar examples
3. Use retrieved examples to guide prediction
4. Evaluate on VAL+TEST set (577 windows)
"""

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

from zero_shot import (
    load_metadata, aggregate_video_predictions,
    compute_metrics, print_metrics, DATA_DIR, WINDOWS_DIR
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rag"
INDEX_DIR = Path(__file__).parent.parent / "rag_index"

MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 4  # Retrieve top 4 similar examples (2 fall, 2 no-fall ideally)


# =============================================================================
# FEATURE EXTRACTION & TEXT REPRESENTATION
# =============================================================================

def extract_features(window_data: Dict) -> Dict:
    """Extract key features from window."""
    sequence = window_data.get("sequence", {})
    analysis = window_data.get("window_analysis", {})
    
    frames = []
    for key in ["frame_1", "frame_2", "frame_3"]:
        frame = sequence.get(key, {})
        if frame.get("pose_detected", False):
            f = frame.get("features", {})
            e = frame.get("enhanced", {})
            frames.append({
                "hip_y": f.get("hip_y", 0),
                "angle": f.get("body_angle_degrees", 90),
                "vel": e.get("velocity_hip_y", 0),
                "posture": e.get("posture_state", "unknown"),
                "ground": e.get("is_on_ground", False),
                "rapid": e.get("is_rapid_descent", False),
                "horiz": e.get("is_horizontal", False),
            })
        else:
            frames.append(None)
    
    return {
        "frames": frames,
        "trajectory": analysis.get("trajectory_direction", "unknown"),
        "max_vel": analysis.get("max_velocity", 0) or 0,
        "spike": analysis.get("has_velocity_spike", False),
    }


def features_to_text(features: Dict) -> str:
    """Convert features to text for embedding."""
    text = ""
    for i, f in enumerate(features["frames"], 1):
        if f:
            text += f"Frame{i}: hip={f['hip_y']:.2f} angle={f['angle']:.0f} "
            text += f"vel={f['vel']:.3f} posture={f['posture']} "
            if f['ground']: text += "ON_GROUND "
            if f['rapid']: text += "RAPID_DESCENT "
            if f['horiz']: text += "HORIZONTAL "
    
    text += f"trajectory={features['trajectory']} "
    text += f"max_vel={features['max_vel']:.3f} "
    if features['spike']: text += "VELOCITY_SPIKE"
    
    return text.strip()


def features_to_prompt(features: Dict) -> str:
    """Convert features to detailed prompt text."""
    text = "Pose Data:\n"
    for i, f in enumerate(features["frames"], 1):
        if f:
            text += f"Frame {i}: hip_y={f['hip_y']:.2f}, angle={f['angle']:.0f}°, "
            text += f"velocity={f['vel']:.3f}, posture={f['posture']}"
            flags = []
            if f['ground']: flags.append("ON_GROUND")
            if f['rapid']: flags.append("RAPID_DESCENT")
            if f['horiz']: flags.append("HORIZONTAL")
            if flags:
                text += f" [{', '.join(flags)}]"
            text += "\n"
        else:
            text += f"Frame {i}: pose not detected\n"
    
    text += f"Trajectory: {features['trajectory']}, "
    text += f"Max velocity: {features['max_vel']:.3f}, "
    text += f"Velocity spike: {features['spike']}"
    
    return text


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

def get_embedding(client: OpenAI, text: str) -> np.ndarray:
    """Get embedding for text."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def build_knowledge_base(client: OpenAI, force_rebuild: bool = False) -> Dict:
    """Build knowledge base from training data."""
    kb_path = INDEX_DIR / "knowledge_base.pkl"
    
    if kb_path.exists() and not force_rebuild:
        print("Loading existing knowledge base...")
        with open(kb_path, "rb") as f:
            return pickle.load(f)
    
    print("Building knowledge base from training data...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    kb = {
        "texts": [],
        "embeddings": [],
        "labels": [],
        "features": [],
    }
    
    train_metadata = load_metadata("train")
    print(f"Training windows: {len(train_metadata)}")
    
    processed = 0
    for wid, meta in train_metadata.items():
        window_path = Path(meta["window_path"])
        
        with open(window_path) as f:
            data = json.load(f)
        
        features = extract_features(data)
        text = features_to_text(features)
        
        embedding = get_embedding(client, text)
        if embedding is not None:
            kb["texts"].append(text)
            kb["embeddings"].append(embedding)
            kb["labels"].append(meta["label"])
            kb["features"].append(features)
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(train_metadata)}...")
        
        time.sleep(0.02)  # Rate limiting
    
    kb["embeddings"] = np.array(kb["embeddings"])
    
    # Save
    with open(kb_path, "wb") as f:
        pickle.dump(kb, f)
    
    fall_count = sum(1 for l in kb["labels"] if l == "fall")
    print(f"Knowledge base built: {len(kb['labels'])} examples")
    print(f"  Fall: {fall_count}, No-Fall: {len(kb['labels']) - fall_count}")
    
    return kb


def retrieve_similar(client: OpenAI, query_features: Dict, kb: Dict, 
                     top_k: int = TOP_K) -> List[Dict]:
    """Retrieve most similar examples from knowledge base."""
    query_text = features_to_text(query_features)
    query_embedding = get_embedding(client, query_text)
    
    if query_embedding is None:
        return []
    
    # Cosine similarity
    embeddings = kb["embeddings"]
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(embeddings_norm, query_norm)
    
    # Get top indices
    top_indices = np.argsort(similarities)[-top_k*2:][::-1]  # Get more, then balance
    
    # Balance retrieved examples (aim for equal fall/no-fall)
    results = []
    fall_count = 0
    nofall_count = 0
    max_per_class = (top_k + 1) // 2
    
    for idx in top_indices:
        label = kb["labels"][idx]
        
        if label == "fall" and fall_count < max_per_class:
            results.append({
                "features": kb["features"][idx],
                "label": "FALL",
                "similarity": float(similarities[idx]),
            })
            fall_count += 1
        elif label == "no_fall" and nofall_count < max_per_class:
            results.append({
                "features": kb["features"][idx],
                "label": "NO_FALL",
                "similarity": float(similarities[idx]),
            })
            nofall_count += 1
        
        if len(results) >= top_k:
            break
    
    return results


# =============================================================================
# RAG PROMPTING
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an expert fall detection system using retrieval-augmented generation.

I will provide:
1. SIMILAR CASES from our database with their known classifications
2. A NEW CASE to classify

Use the similar cases to understand patterns, then classify the new case.

DEFINITIVE FALL (any 1 alone is sufficient):
- posture = "fallen", on_ground = True, or horizontal body (angle < 35 degrees)

PROBABLE FALL (need 2+ together):
- rapid_descent + velocity spike + descending trajectory + body angle < 50 degrees

NO_FALL (controlled movements — look for these):
- Bending or sitting: slow descent, velocity stays low, body returns upright
- Crouching: controlled, posture stays "transitioning" not "fallen"
- Upright posture with stable or ascending trajectory

Compare the new case against retrieved examples carefully. If it closely resembles a NO_FALL case, prefer NO_FALL. Only classify as FALL when multiple strong indicators are present or a definitive indicator is triggered.

Respond with ONLY: FALL or NO_FALL"""


def create_rag_prompt(similar: List[Dict], query_features: Dict) -> str:
    """Create RAG prompt with retrieved examples."""
    prompt = "=== SIMILAR CASES FROM DATABASE ===\n\n"
    
    for i, ex in enumerate(similar, 1):
        prompt += f"Case {i} (similarity: {ex['similarity']:.2f}):\n"
        prompt += features_to_prompt(ex["features"])
        prompt += f"\nClassification: {ex['label']}\n\n"
    
    prompt += "=" * 40 + "\n"
    prompt += "=== NEW CASE TO CLASSIFY ===\n\n"
    prompt += features_to_prompt(query_features)
    prompt += "\n\nClassification:"
    
    return prompt


def call_api(client: OpenAI, system: str, user: str) -> str:
    """Call API with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


def parse_response(response: str) -> str:
    """Parse response."""
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


# =============================================================================
# EVALUATION
# =============================================================================

def run_rag_evaluation(splits: List[str] = ["val", "test"], 
                       sample_size: int = None,
                       rebuild_kb: bool = False):
    """Run RAG evaluation on specified splits."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Build/load knowledge base
    kb = build_knowledge_base(client, rebuild_kb)
    
    # Load evaluation data
    all_metadata = {}
    for split in splits:
        split_meta = load_metadata(split)
        all_metadata.update(split_meta)
    
    print(f"\nEvaluation data: {len(all_metadata)} windows")
    
    window_ids = list(all_metadata.keys())
    if sample_size:
        random.seed(42)
        window_ids = random.sample(window_ids, min(sample_size, len(window_ids)))
        print(f"Sampled: {len(window_ids)} windows")
    
    # Group by video
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_windows[all_metadata[wid]["video_id"]].append(wid)
    
    print(f"Videos: {len(video_windows)}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    processed = 0
    total = len(window_ids)
    
    for video_id, wids in video_windows.items():
        video_label = all_metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            window_path = Path(all_metadata[wid]["window_path"])
            
            with open(window_path) as f:
                data = json.load(f)
            
            features = extract_features(data)
            
            # Retrieve similar examples
            similar = retrieve_similar(client, features, kb, TOP_K)
            
            # Create RAG prompt
            user_prompt = create_rag_prompt(similar, features)
            
            # Get prediction
            response = call_api(client, RAG_SYSTEM_PROMPT, user_prompt)
            pred = parse_response(response)
            
            window_preds[wid] = pred
            
            window_results.append({
                "window_id": wid,
                "true_label": all_metadata[wid]["label"],
                "predicted": pred,
                "num_retrieved": len(similar),
            })
            
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{total}...")
            
            time.sleep(0.15)
        
        # Aggregate with 25% threshold (tiered prompt reduces FP at window level)
        fall_count = sum(1 for p in window_preds.values() if p == "fall")
        video_pred = "fall" if fall_count / len(window_preds) >= 0.25 else "no_fall"
        
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
            "fall_windows": fall_count,
            "total_windows": len(window_preds),
        })
    
    # Save results
    split_name = "_".join(splits)
    with open(RESULTS_DIR / f"window_results_{split_name}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    with open(RESULTS_DIR / f"video_results_{split_name}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RAG FALL DETECTION RESULTS")
    print("="*60)
    
    w_true = [r["true_label"] for r in window_results]
    w_pred = [r["predicted"] for r in window_results]
    w_metrics = compute_metrics(w_true, w_pred)
    print_metrics(w_metrics, "Window")
    
    v_true = [r["true_label"] for r in video_results]
    v_pred = [r["predicted"] for r in video_results]
    v_metrics = compute_metrics(v_true, v_pred)
    print_metrics(v_metrics, "Video")
    
    # Save metrics
    with open(RESULTS_DIR / f"metrics_{split_name}.json", "w") as f:
        json.dump({
            "window_level": w_metrics,
            "video_level": v_metrics,
            "config": {
                "model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "top_k": TOP_K,
                "splits": splits,
                "kb_size": len(kb["labels"]),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return w_metrics, v_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--rebuild-kb", action="store_true")
    args = parser.parse_args()
    
    run_rag_evaluation(args.splits, args.sample, args.rebuild_kb)


if __name__ == "__main__":
    main()
