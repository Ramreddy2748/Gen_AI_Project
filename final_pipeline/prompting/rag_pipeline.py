"""
RAG (Retrieval-Augmented Generation) Pipeline for Fall Detection
================================================================

Uses semantic similarity to retrieve the most relevant examples
for each query, providing targeted context to the LLM.

Key Components:
1. Embedding generation using sentence-transformers
2. FAISS index for fast similarity search
3. Dynamic example retrieval based on query similarity
4. GPT-4o for final classification

Target: 85%+ accuracy
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
from enhanced_few_shot import create_rich_description

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rag"
INDEX_DIR = Path(__file__).parent.parent / "rag_index"

MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.0
MAX_TOKENS = 150
TOP_K = 5  # Number of similar examples to retrieve


# =============================================================================
# EMBEDDING AND INDEXING
# =============================================================================

def get_embedding(client: OpenAI, text: str) -> np.ndarray:
    """Get embedding for text using OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def build_knowledge_base(client: OpenAI, force_rebuild: bool = False):
    """Build or load the knowledge base with embeddings."""
    index_file = INDEX_DIR / "knowledge_base.pkl"
    
    if index_file.exists() and not force_rebuild:
        print("Loading existing knowledge base...")
        with open(index_file, "rb") as f:
            return pickle.load(f)
    
    print("Building knowledge base from training data...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    knowledge_base = {
        "descriptions": [],
        "labels": [],
        "embeddings": [],
        "window_ids": [],
    }
    
    # Process training data
    for label in ["fall", "no_fall"]:
        label_dir = WINDOWS_DIR / "train" / label
        if not label_dir.exists():
            continue
        
        count = 0
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            for window_file in video_dir.glob("*.json"):
                with open(window_file) as f:
                    data = json.load(f)
                
                description = create_rich_description(data)
                
                # Get embedding
                embedding = get_embedding(client, description)
                
                knowledge_base["descriptions"].append(description)
                knowledge_base["labels"].append(label)
                knowledge_base["embeddings"].append(embedding)
                knowledge_base["window_ids"].append(window_file.stem)
                
                count += 1
                if count % 50 == 0:
                    print(f"  Processed {count} {label} windows...")
                
                time.sleep(0.05)  # Rate limiting
    
    # Convert to numpy array for efficient search
    knowledge_base["embeddings"] = np.array(knowledge_base["embeddings"])
    
    # Save
    with open(index_file, "wb") as f:
        pickle.dump(knowledge_base, f)
    
    print(f"Knowledge base built with {len(knowledge_base['labels'])} examples")
    print(f"  Fall: {knowledge_base['labels'].count('fall')}")
    print(f"  No-Fall: {knowledge_base['labels'].count('no_fall')}")
    
    return knowledge_base


def retrieve_similar(client: OpenAI, query_description: str, 
                     knowledge_base: Dict, top_k: int = TOP_K) -> List[Dict]:
    """Retrieve most similar examples from knowledge base."""
    # Get query embedding
    query_embedding = get_embedding(client, query_description)
    
    # Compute cosine similarities
    embeddings = knowledge_base["embeddings"]
    
    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Cosine similarity
    similarities = np.dot(embeddings_norm, query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return similar examples with balanced labels
    results = []
    fall_count = 0
    nofall_count = 0
    max_per_class = (top_k + 1) // 2
    
    for idx in top_indices:
        label = knowledge_base["labels"][idx]
        
        # Balance the retrieved examples
        if label == "fall" and fall_count < max_per_class:
            results.append({
                "description": knowledge_base["descriptions"][idx],
                "label": label.upper().replace("_", " "),
                "similarity": float(similarities[idx]),
            })
            fall_count += 1
        elif label == "no_fall" and nofall_count < max_per_class:
            results.append({
                "description": knowledge_base["descriptions"][idx],
                "label": label.upper().replace("_", " "),
                "similarity": float(similarities[idx]),
            })
            nofall_count += 1
        
        if len(results) >= top_k:
            break
    
    # If not enough balanced, add remaining
    if len(results) < top_k:
        for idx in top_indices:
            if idx not in [knowledge_base["window_ids"].index(r.get("window_id", "")) 
                          for r in results if "window_id" in r]:
                results.append({
                    "description": knowledge_base["descriptions"][idx],
                    "label": knowledge_base["labels"][idx].upper().replace("_", " "),
                    "similarity": float(similarities[idx]),
                })
            if len(results) >= top_k:
                break
    
    return results


# =============================================================================
# RAG PROMPTING
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an expert fall detection AI. You will analyze a sequence of human movement data.

I will provide you with SIMILAR CASES from our database that are most relevant to the current sequence.
Use these similar cases to inform your decision, but focus on the specific characteristics of the NEW sequence.

## FALL INDICATORS:
- Rapid downward movement (velocity spike)
- Body transitioning from upright to horizontal
- Person ending up on/near ground level
- Sudden uncontrolled movement

## NORMAL ACTIVITY INDICATORS:
- Slow, controlled movements
- Body maintains upright orientation
- Gradual position changes
- No velocity spikes

Based on the similar cases and the new sequence, respond with ONLY: "FALL" or "NO_FALL"
"""


def create_rag_prompt(similar_examples: List[Dict], query_description: str) -> str:
    """Create RAG prompt with retrieved examples."""
    prompt = "## SIMILAR CASES FROM DATABASE:\n\n"
    
    for i, ex in enumerate(similar_examples, 1):
        prompt += f"### Similar Case {i} (similarity: {ex['similarity']:.2f}):\n"
        prompt += ex["description"]
        prompt += f"\n**Classification: {ex['label']}**\n"
        prompt += "-" * 40 + "\n\n"
    
    prompt += "=" * 50 + "\n"
    prompt += "## NEW SEQUENCE TO CLASSIFY:\n\n"
    prompt += query_description
    prompt += "\n\n**Your classification (FALL or NO_FALL):**"
    
    return prompt


def call_openai(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"


def parse_response(response: str) -> str:
    """Parse response."""
    response = response.upper().strip()
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_rag_evaluation(split: str = "test", sample_size: int = None,
                       top_k: int = TOP_K, rebuild_index: bool = False):
    """Run RAG evaluation."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Using model: {MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Top-K retrieval: {top_k}")
    
    # Build/load knowledge base
    knowledge_base = build_knowledge_base(client, rebuild_index)
    
    print(f"\nLoading {split} split...")
    metadata = load_metadata(split)
    print(f"Total windows: {len(metadata)}")
    
    window_ids = list(metadata.keys())
    if sample_size and sample_size < len(window_ids):
        random.seed(42)
        window_ids = random.sample(window_ids, sample_size)
        print(f"Sampled {sample_size} windows")
    
    video_windows = defaultdict(list)
    for wid in window_ids:
        video_id = metadata[wid]["video_id"]
        video_windows[video_id].append(wid)
    
    print(f"Videos: {len(video_windows)}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    window_results = []
    video_results = []
    
    total_windows = len(window_ids)
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            window_path = Path(metadata[wid]["window_path"])
            
            with open(window_path) as f:
                window_data = json.load(f)
            
            # Create description
            description = create_rich_description(window_data)
            
            # Retrieve similar examples
            similar = retrieve_similar(client, description, knowledge_base, top_k)
            
            # Create RAG prompt
            user_prompt = create_rag_prompt(similar, description)
            
            # Get prediction
            response = call_openai(client, RAG_SYSTEM_PROMPT, user_prompt)
            prediction = parse_response(response)
            
            window_preds[wid] = prediction
            
            window_results.append({
                "window_id": wid,
                "video_id": video_id,
                "true_label": metadata[wid]["label"],
                "predicted": prediction,
                "raw_response": response,
                "num_similar_retrieved": len(similar),
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_windows}...")
            
            time.sleep(0.1)
        
        video_pred = aggregate_video_predictions(window_preds, "majority")
        
        video_results.append({
            "video_id": video_id,
            "true_label": video_label,
            "predicted": video_pred,
            "window_count": len(wids),
            "fall_windows": sum(1 for p in window_preds.values() if p == "fall"),
        })
    
    # Save results
    with open(RESULTS_DIR / f"window_results_{split}.json", "w") as f:
        json.dump(window_results, f, indent=2)
    
    with open(RESULTS_DIR / f"video_results_{split}.json", "w") as f:
        json.dump(video_results, f, indent=2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RAG PIPELINE RESULTS")
    print("="*60)
    
    window_true = [r["true_label"] for r in window_results]
    window_pred = [r["predicted"] for r in window_results]
    window_metrics = compute_metrics(window_true, window_pred)
    print_metrics(window_metrics, "Window")
    
    video_true = [r["true_label"] for r in video_results]
    video_pred = [r["predicted"] for r in video_results]
    video_metrics = compute_metrics(video_true, video_pred)
    print_metrics(video_metrics, "Video")
    
    # Save metrics
    with open(RESULTS_DIR / f"metrics_{split}.json", "w") as f:
        json.dump({
            "window_level": window_metrics,
            "video_level": video_metrics,
            "config": {
                "model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "split": split,
                "top_k": top_k,
                "technique": "rag_with_semantic_retrieval"
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return window_metrics, video_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    
    run_rag_evaluation(args.split, args.sample, args.top_k, args.rebuild_index)


if __name__ == "__main__":
    main()
