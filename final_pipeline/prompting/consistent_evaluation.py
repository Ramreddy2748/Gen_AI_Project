"""
Consistent Evaluation - All Techniques on Val+Test
===================================================

Runs all prompting strategies on the same evaluation set (Val+Test)
for fair comparison.

Evaluation Set: 577 windows, 115 videos
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from zero_shot import (
    load_metadata, sanitize_window_for_llm, create_zero_shot_prompt,
    aggregate_video_predictions, compute_metrics, print_metrics
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "consistent"


# =============================================================================
# PROMPTS
# =============================================================================

ZERO_SHOT_SYSTEM = """You are a fall detection system analyzing human pose data.
Respond with ONLY: FALL or NO_FALL"""

FEW_SHOT_SYSTEM = """You are a fall detection system. Based on the examples and the new data, classify as FALL or NO_FALL.
Respond with ONLY: FALL or NO_FALL"""

COT_SYSTEM = """You are a fall detection expert. Analyze step by step:
1. Position changes (hip_y movement)
2. Body orientation (angle)
3. Velocity patterns
4. Fall indicators

Then respond with your decision.
End your response with: DECISION: FALL or DECISION: NO_FALL"""

SAFETY_SYSTEM = """You are a SAFETY-CRITICAL fall detection system for elderly monitoring.

CRITICAL: Missing a fall (false negative) is DANGEROUS. When in doubt, classify as FALL.

FALL indicators (any ONE is significant):
- Rapid descent (velocity > 0.05)
- Horizontal body (angle < 45°)
- Person on ground (hip_y > 0.7)
- "fallen" posture state
- Velocity spike

NO_FALL requires ALL of:
- Upright posture throughout
- Stable/slow movement
- No fall flags triggered

Respond with ONLY: FALL or NO_FALL"""


def load_few_shot_examples(metadata: Dict, num_examples: int = 6) -> List[Dict]:
    """Load balanced examples from training data."""
    import random
    random.seed(42)
    
    train_meta = load_metadata("train")
    fall_ids = [w for w in train_meta if train_meta[w]["label"] == "fall"]
    nofall_ids = [w for w in train_meta if train_meta[w]["label"] == "no_fall"]
    
    examples = []
    for wid in random.sample(fall_ids, num_examples // 2):
        with open(train_meta[wid]["window_path"]) as f:
            data = json.load(f)
        examples.append({"data": sanitize_window_for_llm(data), "label": "FALL"})
    
    for wid in random.sample(nofall_ids, num_examples // 2):
        with open(train_meta[wid]["window_path"]) as f:
            data = json.load(f)
        examples.append({"data": sanitize_window_for_llm(data), "label": "NO_FALL"})
    
    random.shuffle(examples)
    return examples


def call_api(client: OpenAI, model: str, system: str, user: str, 
             temperature: float = 0.0, max_tokens: int = 50) -> str:
    """Call API with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


def parse_response(response: str) -> str:
    """Parse response to fall/no_fall."""
    response = response.upper()
    if "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


def parse_cot_response(response: str) -> str:
    """Parse CoT response."""
    response = response.upper()
    if "DECISION: NO_FALL" in response or "DECISION: NO FALL" in response:
        return "no_fall"
    elif "DECISION: FALL" in response or "DECISION:" in response and "FALL" in response:
        return "fall"
    elif "NO_FALL" in response or "NO FALL" in response:
        return "no_fall"
    elif "FALL" in response:
        return "fall"
    return "unknown"


def create_few_shot_prompt(examples: List[Dict], query_data: Dict) -> str:
    """Create few-shot prompt."""
    prompt = "Here are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"{json.dumps(ex['data'], indent=2)[:500]}...\n"
        prompt += f"Classification: {ex['label']}\n\n"
    
    prompt += "Now classify this:\n"
    prompt += json.dumps(query_data, indent=2)[:800]
    prompt += "\n\nClassification:"
    
    return prompt


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_technique(client: OpenAI, technique: str, metadata: Dict, 
                       model: str = "gpt-4o-mini") -> Dict:
    """Evaluate a single technique."""
    
    # Prepare based on technique
    if technique == "few_shot":
        examples = load_few_shot_examples(metadata)
    else:
        examples = None
    
    results = []
    video_windows = defaultdict(list)
    
    # Group by video
    for wid, meta in metadata.items():
        video_windows[meta["video_id"]].append(wid)
    
    total = len(metadata)
    processed = 0
    
    for video_id, wids in video_windows.items():
        video_label = metadata[wids[0]]["label"]
        window_preds = {}
        
        for wid in sorted(wids):
            with open(metadata[wid]["window_path"]) as f:
                data = json.load(f)
            
            sanitized = sanitize_window_for_llm(data)
            
            # Create prompt based on technique
            if technique == "zero_shot":
                user_prompt = create_zero_shot_prompt(sanitized)
                response = call_api(client, model, ZERO_SHOT_SYSTEM, user_prompt)
                pred = parse_response(response)
                
            elif technique == "few_shot":
                user_prompt = create_few_shot_prompt(examples, sanitized)
                response = call_api(client, model, FEW_SHOT_SYSTEM, user_prompt)
                pred = parse_response(response)
                
            elif technique == "cot":
                user_prompt = create_zero_shot_prompt(sanitized)
                response = call_api(client, model, COT_SYSTEM, user_prompt, max_tokens=300)
                pred = parse_cot_response(response)
                
            elif technique == "self_consistency":
                votes = []
                for _ in range(5):
                    user_prompt = create_zero_shot_prompt(sanitized)
                    response = call_api(client, model, ZERO_SHOT_SYSTEM, user_prompt, temperature=0.7)
                    votes.append(parse_response(response))
                pred = "fall" if votes.count("fall") > votes.count("no_fall") else "no_fall"
                
            elif technique == "safety_first":
                user_prompt = create_zero_shot_prompt(sanitized)
                response = call_api(client, "gpt-4o", SAFETY_SYSTEM, user_prompt)
                pred = parse_response(response)
            
            else:
                pred = "unknown"
            
            window_preds[wid] = pred
            results.append({
                "window_id": wid,
                "true_label": metadata[wid]["label"],
                "predicted": pred,
            })
            
            processed += 1
            if processed % 50 == 0:
                print(f"  {technique}: {processed}/{total}...")
            
            time.sleep(0.15)
    
    return results


def aggregate_results(results: List[Dict], metadata: Dict, threshold: float = 0.25) -> List[Dict]:
    """Aggregate window results to video level."""
    video_windows = defaultdict(list)
    
    for r in results:
        wid = r["window_id"]
        video_id = metadata[wid]["video_id"]
        video_windows[video_id].append(r)
    
    video_results = []
    for video_id, window_results in video_windows.items():
        true_label = window_results[0]["true_label"]
        fall_count = sum(1 for r in window_results if r["predicted"] == "fall")
        total = len(window_results)
        
        # Use threshold aggregation
        video_pred = "fall" if fall_count / total >= threshold else "no_fall"
        
        video_results.append({
            "video_id": video_id,
            "true_label": true_label,
            "predicted": video_pred,
            "fall_windows": fall_count,
            "total_windows": total,
        })
    
    return video_results


def run_consistent_evaluation():
    """Run all techniques on Val+Test."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load Val+Test data
    all_metadata = {}
    for split in ["val", "test"]:
        split_meta = load_metadata(split)
        all_metadata.update(split_meta)
    
    print(f"Evaluation data: {len(all_metadata)} windows")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    techniques = ["zero_shot", "few_shot", "cot", "self_consistency", "safety_first"]
    all_results = {}
    
    for technique in techniques:
        print(f"\nEvaluating: {technique}")
        
        results = evaluate_technique(client, technique, all_metadata)
        
        # Window metrics
        w_true = [r["true_label"] for r in results]
        w_pred = [r["predicted"] for r in results]
        w_metrics = compute_metrics(w_true, w_pred)
        
        # Video metrics (25% threshold)
        video_results = aggregate_results(results, all_metadata, threshold=0.25)
        v_true = [r["true_label"] for r in video_results]
        v_pred = [r["predicted"] for r in video_results]
        v_metrics = compute_metrics(v_true, v_pred)
        
        all_results[technique] = {
            "window_results": results,
            "video_results": video_results,
            "window_metrics": w_metrics,
            "video_metrics": v_metrics,
        }
        
        print(f"  Window Acc: {w_metrics['overall']['accuracy']:.1%}, Fall Recall: {w_metrics['fall']['recall']:.1%}")
        print(f"  Video Acc: {v_metrics['overall']['accuracy']:.1%}, Fall Recall: {v_metrics['fall']['recall']:.1%}")
    
    # Save all results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        # Convert to serializable format
        serializable = {}
        for tech, data in all_results.items():
            serializable[tech] = {
                "window_metrics": data["window_metrics"],
                "video_metrics": data["video_metrics"],
            }
        json.dump(serializable, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("CONSISTENT EVALUATION SUMMARY (Val+Test: 577 windows, 115 videos)")
    print("="*80)
    
    print(f"\n{'Technique':<20} {'W.Acc':>8} {'W.Recall':>10} {'V.Acc':>8} {'V.Recall':>10}")
    print("-"*60)
    
    for tech in techniques:
        wm = all_results[tech]["window_metrics"]
        vm = all_results[tech]["video_metrics"]
        print(f"{tech:<20} {wm['overall']['accuracy']:>7.1%} {wm['fall']['recall']:>9.1%} "
              f"{vm['overall']['accuracy']:>7.1%} {vm['fall']['recall']:>9.1%}")
    
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    run_consistent_evaluation()
