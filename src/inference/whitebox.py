"""
White-box inference branch with hierarchical attribute verification and evidence-based reasoning.
"""

import os
import re
import math
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from config.settings import (
    CSV_DIR, ATTR_TOP_K, VISIBILITY_THRESHOLD,
    LAYER_WEIGHTS, CONSISTENCY_BONUS
)
from models.loader import load_vlm
from utils.tree import load_tree, get_valid_children


# ============================================================================
# Internal helpers for VLM interaction
# ============================================================================

def _get_dimension_confidence(dimension: str, pil_image: Image.Image, model, processor) -> float:
    """
    Evaluate the visual observability/identifiability of a given attribute dimension.
    Returns a continuous confidence score in [0, 1].
    """
    prompt = f"""You are an expert biological taxonomist analyzing an animal image.
Analyze the specific visual attribute: "{dimension}".

Rate how clearly this specific visual attribute is observable or distinguishable in the image on a scale from 0.0 to 1.0:
- 1.0: Extremely clear, prominent, and fully visible.
- 0.7 - 0.9: Clearly visible, but may require careful observation.
- 0.4 - 0.6: Moderately visible, partially obscured, blurry, or ambiguous.
- 0.1 - 0.3: Barely visible, extremely faint, or mostly hidden.
- 0.0: Completely invisible, absent, or impossible to determine from this angle.

Output strictly in JSON format:
{{"confidence": float_value_between_0_and_1}}"""

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[pil_image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    try:
        json_match = re.search(r'\{.*\}', response, re.IGNORECASE | re.DOTALL)
        result = json.loads(json_match.group(0)) if json_match else json.loads(response)
        conf = float(result.get("confidence", 0.0))
        return max(0.0, min(1.0, conf))
    except Exception:
        return 0.0


def _interrogate_dimension_soft(
    dimension: str,
    options: List[str],
    pil_image: Image.Image,
    model,
    processor,
    top_k: int = ATTR_TOP_K
) -> Dict[str, Any]:
    """
    Query VLM for the most likely attribute values among the given options.
    Returns a dict with keys: 'observable', 'visual_description', 'evidence', 'distribution'.
    """
    options_str = "\n".join([f"- {opt}" for opt in options])

    # Build a distribution placeholder for top_k options
    # We fill with dummy values to guide JSON formatting
    dist_examples = []
    for i in range(top_k):
        prob = 0.5 if i == 0 else 0.3 if i == 1 else 0.1 if i == 2 else round(0.1 / max(1, top_k - 2), 2)
        dist_examples.append(f'        "<TOP{i+1}_EXACT>": {prob:.2f}')
    dist_json = ",\n".join(dist_examples)

    prompt = f"""You are an expert biological taxonomist analyzing an animal image.
Focus strictly on this specific visual attribute: "{dimension}".

Candidate values:
{options_str}

Tasks:
1. Is this specific attribute visually observable in the image?
2. Step 1 (Visual Analysis): Completely ignore the candidate list. Look at the image and describe ONLY the body part related to "{dimension}".
3. Step 2 (Matching): Compare your observation with the "Candidate values" list. Select up to {top_k} options that best match your observation (Top-1 to Top-{top_k}). You MUST copy the exact strings from the Candidate values. Assign a probability score (0.0 to 1.0) to each selected option so they sum to 1.0. If none fit well, choose "None".

Output strictly in the following JSON format:
{{
    "observable": true,
    "visual_description": "<actual observation>",
    "evidence": "<brief reasoning>",
    "distribution": {{
{dist_json}
    }}
}}"""

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[pil_image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            temperature=0.1,
            top_p=0.9,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    try:
        json_match = re.search(r'\{.*\}', response, re.IGNORECASE | re.DOTALL)
        result = json.loads(json_match.group(0)) if json_match else json.loads(response)
        # Clean up distribution keys: remove the dummy placeholders
        dist = result.get("distribution", {})
        cleaned_dist = {}
        for k, v in dist.items():
            # Skip placeholder keys like "<TOP1_EXACT>"
            if k.startswith("<TOP") or k.startswith("TOP"):
                continue
            try:
                cleaned_dist[k.strip()] = float(v)
            except (ValueError, TypeError):
                continue
        result["distribution"] = cleaned_dist
        return result
    except Exception:
        return {"observable": False, "distribution": {}}


# ============================================================================
# Single-layer inference
# ============================================================================

def infer_layer(
    pil_image: Image.Image,
    layer_name: str,
    model,
    processor,
    attr_top_k: int = ATTR_TOP_K,
    print_output: bool = False
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Perform attribute verification for one taxonomic level.
    Returns:
        scores: {class_name: accumulated_score}
        evidence_log: list of evidence dicts with keys:
            Layer, Dimension, Selected, Confidence, Option_Prob, Final_Weight
    """
    # Determine CSV path: L1 uses a special file, others use standardized format
    if layer_name == "L1":
        csv_path = os.path.join(CSV_DIR, "L1_filled_new.csv")
    else:
        csv_path = os.path.join(CSV_DIR, f"{layer_name}_standardized_local.csv")

    if not os.path.exists(csv_path):
        if print_output:
            print(f"Warning: CSV for {layer_name} not found at {csv_path}")
        return {}, []

    df = pd.read_csv(csv_path, index_col=0)
    target_classes = [c.split('(')[0].strip() for c in df.columns]
    N = len(target_classes)
    scores = {cls: 0.0 for cls in target_classes}
    evidence_log = []

    if print_output:
        print(f"\n[Layer {layer_name}] Dimensions: {len(df.index)}, Classes: {N}")

    for dimension in df.index:
        # Clean attribute strings
        raw_attrs = [str(df.loc[dimension, col]) for col in df.columns]
        clean_attrs = [
            re.sub(r'[（\(].*?[）\)]', '', a).replace('_', ' ').strip()
            for a in raw_attrs
        ]
        unique_options = list(dict.fromkeys(
            a for a in clean_attrs
            if a.lower() not in ('nan', '') and a != ''
        ))

        if len(unique_options) <= 1:
            continue

        # Visibility pre-check (skip for L1, apply for L2-L6)
        visibility = 1.0
        if layer_name != "L1":
            visibility = _get_dimension_confidence(dimension, pil_image, model, processor)
            if visibility <= VISIBILITY_THRESHOLD:
                if print_output:
                    print(f"  [Skip] {dimension[:30]}... visibility={visibility:.2f} <= {VISIBILITY_THRESHOLD}")
                continue

        # Query VLM for top-K candidates
        vlm_result = _interrogate_dimension_soft(
            dimension, unique_options, pil_image, model, processor, top_k=attr_top_k
        )
        if not vlm_result.get("observable", False):
            continue

        distribution = vlm_result.get("distribution", {})
        sorted_options = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        matched_any = False

        if print_output and sorted_options:
            top_display = " | ".join([f"{opt} ({prob:.2f})" for opt, prob in sorted_options[:attr_top_k]])
            print(f"  [VLM] {dimension[:25]}... Top-{attr_top_k}: {top_display}")

        # Process each candidate attribute value
        import string
        punct_table = str.maketrans('', '', string.punctuation)

        for opt, prob in sorted_options:
            if opt.lower() == "none" or prob <= 0.05:
                continue

            # Normalize for fuzzy matching (remove punctuation, lowercase)
            clean_opt = opt.translate(punct_table).lower().strip()
            matching_classes = []
            for idx, cls in enumerate(target_classes):
                clean_cls_attr = clean_attrs[idx].translate(punct_table).lower().strip()
                if clean_cls_attr == clean_opt:
                    matching_classes.append(cls)

            k = len(matching_classes)
            if k > 0:
                matched_any = True
                # Inverse Category Frequency (ICF) weighting
                raw_weight = math.log(N / k) + 1.0
                final_weight = raw_weight * visibility * prob

                for cls in matching_classes:
                    scores[cls] += final_weight

                evidence_log.append({
                    "Layer": layer_name,
                    "Dimension": dimension,
                    "Selected": opt,
                    "Confidence": visibility,
                    "Option_Prob": prob,
                    "Final_Weight": final_weight,
                    "Matching_Classes": matching_classes
                })

                if print_output:
                    print(f"    -> {opt} (k={k}, weight={final_weight:.2f})")

        if not matched_any and print_output:
            print(f"  [No match] {dimension[:30]}...")

    return scores, evidence_log


# ============================================================================
# Flexible Top-K path segmentation (for HC, POR, etc.)
# ============================================================================

def evaluate_flexible_topk_path(
    layer_topk_results: Dict[str, Any],
    tree_dict: Dict,
    k: int = 3
) -> Tuple[List[str], int, float, List[Dict]]:
    """
    Enhanced version: supports segment connectivity across broken intermediate layers.
    Returns:
        main_path: the longest connected path chosen from the best segment
        max_depth: depth of the best segment (1~6)
        best_score: accumulated score of the best segment
        segments: list of all disconnected segments with their start_level and score
    """
    taxa_levels = ["L1", "L2", "L3", "L4", "L5", "L6"]

    candidates = {}
    for level in taxa_levels:
        if level in layer_topk_results and "all_sorted" in layer_topk_results[level]:
            candidates[level] = layer_topk_results[level]["all_sorted"][:k]
        else:
            candidates[level] = []

    def get_valid_children(path, tree):
        node = tree.get("Animalia", tree)
        for taxon in path:
            if isinstance(node, dict) and taxon in node:
                node = node[taxon]
            else:
                return []
        if isinstance(node, dict):
            return list(node.keys())
        elif isinstance(node, list):
            return node
        return []

    segments = []
    current_segment = []
    current_score = 0.0
    current_start_level_idx = 0

    for depth_idx, level in enumerate(taxa_levels):
        level_cands = candidates[level]
        if not level_cands:
            continue

        valid_children = get_valid_children(current_segment, tree_dict)
        extended = False
        best_next = None
        best_next_score = -1.0

        for name, score in level_cands:
            if not current_segment or name in valid_children:
                if score > best_next_score:
                    best_next = name
                    best_next_score = score
                extended = True

        if extended and best_next is not None:
            current_segment.append(best_next)
            current_score += best_next_score
        else:
            # Break: save current segment
            if current_segment:
                segments.append({
                    "path": current_segment[:],
                    "score": current_score,
                    "start_level": taxa_levels[current_start_level_idx],
                    "depth": len(current_segment)
                })
            # Start new segment
            current_segment = [best_next] if best_next else []
            current_score = best_next_score if best_next else 0.0
            current_start_level_idx = depth_idx

    # Final segment
    if current_segment:
        segments.append({
            "path": current_segment[:],
            "score": current_score,
            "start_level": taxa_levels[current_start_level_idx],
            "depth": len(current_segment)
        })

    if not segments:
        return [], 0, 0.0, []

    # Choose best segment (depth first, then score)
    best_segment = max(segments, key=lambda s: (s["depth"], s["score"]))

    # Build main path by merging connected segments from root
    main_path = []
    for seg in segments:
        if not main_path or seg["path"][0] in get_valid_children(main_path, tree_dict):
            main_path.extend(seg["path"])
        else:
            break

    return main_path, best_segment["depth"], best_segment["score"], segments


# ============================================================================
# Global DAG path search with backward pruning
# ============================================================================

def global_path_search(
    layer_scores: Dict[str, Dict[str, float]],
    tree_dict: Dict,
    top_m_backward: int = 10
) -> List[Tuple[List[str], float, Dict[str, Any]]]:
    """
    Optimized global path search with top-m backward pruning.
    Returns a list of (path, total_score, extra_info) sorted by total_score descending.
    Extra_info contains 'confidence' (consistency fraction) and 'layer_details' (weighted scores).
    """
    weights = LAYER_WEIGHTS
    bonus = CONSISTENCY_BONUS
    root = tree_dict.get("Animalia", tree_dict)

    # Step 1: Enumerate all species with their leaf-level (L6) score
    all_species_with_scores = []

    def collect_species(node, current_path, depth):
        if depth == 5:
            if isinstance(node, list):
                for sp in node:
                    score = layer_scores["L6"].get(sp, 0.0) * weights[5]
                    all_species_with_scores.append((current_path + [sp], score))
            return
        if isinstance(node, dict):
            for taxon, child in node.items():
                collect_species(child, current_path + [taxon], depth + 1)

    collect_species(root, [], 0)

    # Step 2: Sort by leaf score and keep top_m_backward
    all_species_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_species = all_species_with_scores[:top_m_backward]

    # Step 3: For each kept species, compute full path score with consistency bonus
    scored_paths = []
    for path, base_score in top_species:
        consistency_count = 0
        total_bonus = 0.0
        layer_details = []

        for i, name in enumerate(path):
            lv = f"L{i+1}"
            score = layer_scores[lv].get(name, 0.0)
            weighted = score * weights[i]
            layer_details.append(weighted)
            if score > 0:
                consistency_count += 1
                total_bonus += bonus * consistency_count

        final_score = base_score + total_bonus
        confidence = consistency_count / 6.0
        scored_paths.append((path, final_score, {
            "confidence": confidence,
            "layer_details": layer_details
        }))

    # Step 4: Sort descending by final_score, then confidence
    scored_paths.sort(key=lambda x: (x[1], x[2]["confidence"]), reverse=True)
    return scored_paths


# ============================================================================
# Main entry: full white-box inference on a single image
# ============================================================================

def infer_single_image(
    image_path: str,
    print_output: bool = False,
    attr_top_k: int = ATTR_TOP_K
) -> Dict[str, Any]:
    """
    Run full white-box inference on a single image.
    Returns a rich dictionary with:
        - image_path
        - final_top1_path (list of L1~L6)
        - final_top1_score
        - final_top1_extra (confidence, layer_details)
        - ranked_paths (all scored paths)
        - layer_scores (per-layer scores)
        - layer_evidence_log (list of evidence dicts)
        - our_confidence (margin * integrity)
        - topk_segments (k1, k3, k5 segment paths for metrics)
    """
    model, processor = load_vlm()
    pil_image = Image.open(image_path).convert('RGB')

    if print_output:
        print("\n" + "=" * 80)
        print(f"White-box inference for: {image_path}")
        print("=" * 80)

    # ------------------------------------------------------------------------
    # Step 1: Layer-wise attribute verification
    # ------------------------------------------------------------------------
    layer_scores = {}
    layer_evidence_log = []
    layer_topk_results = {}

    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        if print_output:
            print(f"\n--- Processing layer {level} ---")

        scores, evidence = infer_layer(
            pil_image, level, model, processor,
            attr_top_k=attr_top_k,
            print_output=print_output
        )
        layer_scores[level] = scores
        layer_evidence_log.extend(evidence)

        # Build top-k results for flexible path segmentation
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        layer_topk_results[level] = {
            "all_sorted": sorted_items,
            "top1": sorted_items[0][0] if sorted_items else None,
            "top3": [x[0] for x in sorted_items[:3]] if sorted_items else [],
            "top5": [x[0] for x in sorted_items[:5]] if sorted_items else []
        }

        if print_output and sorted_items:
            print(f"\n[{level}] Top-10 scores:")
            for rank, (cls, sc) in enumerate(sorted_items[:10], 1):
                print(f"  {rank:2d}. {cls:<35} | {sc:.2f}")

    # ------------------------------------------------------------------------
    # Step 2: Global path search (DAG with backward pruning)
    # ------------------------------------------------------------------------
    tree_dict = load_tree()
    ranked_paths = global_path_search(layer_scores, tree_dict, top_m_backward=10)

    if not ranked_paths:
        return {
            "image_path": image_path,
            "final_top1_path": [],
            "final_top1_score": 0.0,
            "final_top1_extra": None,
            "ranked_paths": [],
            "layer_scores": layer_scores,
            "layer_evidence_log": layer_evidence_log,
            "our_confidence": 0.0,
            "topk_segments": {"k1_segments": [], "k3_segments": [], "k5_segments": []}
        }

    top1_path, top1_score, top1_extra = ranked_paths[0]

    # Compute our_confidence: margin * structural integrity
    if len(ranked_paths) >= 2:
        margin = (top1_score - ranked_paths[1][1]) / (top1_score + 1e-8)
    else:
        margin = 1.0
    integrity = top1_extra["confidence"]
    our_confidence = margin * integrity

    # ------------------------------------------------------------------------
    # Step 3: Flexible Top-K segmentation (for metrics)
    # ------------------------------------------------------------------------
    _, _, _, seg_k1 = evaluate_flexible_topk_path(layer_topk_results, tree_dict, k=1)
    _, _, _, seg_k3 = evaluate_flexible_topk_path(layer_topk_results, tree_dict, k=3)
    _, _, _, seg_k5 = evaluate_flexible_topk_path(layer_topk_results, tree_dict, k=5)

    # ------------------------------------------------------------------------
    # Step 4: Print results
    # ------------------------------------------------------------------------
    if print_output:
        print("\n" + "-" * 80)
        print("FINAL RESULTS")
        print("-" * 80)
        print(f"Top-1 Path: {' -> '.join(top1_path)}")
        print(f"Top-1 Score: {top1_score:.2f}")
        print(f"Confidence (margin * integrity): {our_confidence:.4f}")

        print("\nTop-5 Paths:")
        for i, (path, score, extra) in enumerate(ranked_paths[:5], 1):
            conf = extra["confidence"] * 100
            print(f"  #{i}: {score:.2f} | {' -> '.join(path)} | integrity={conf:.1f}%")

        print("\nFlexible Top-K Segment Connectivity:")
        print(f"  K=1 segments: {len(seg_k1)}")
        print(f"  K=3 segments: {len(seg_k3)}")
        print(f"  K=5 segments: {len(seg_k5)}")

        print("\nEvidence Log (top 10 by weight):")
        sorted_ev = sorted(layer_evidence_log, key=lambda x: x.get("Final_Weight", 0), reverse=True)
        for i, ev in enumerate(sorted_ev[:10], 1):
            print(f"  {i:2d}. [{ev['Layer']}] {ev['Dimension'][:30]}... -> {ev['Selected']} (w={ev['Final_Weight']:.2f})")

        print("=" * 80 + "\n")

    # ------------------------------------------------------------------------
    # Step 5: Return structured result
    # ------------------------------------------------------------------------
    return {
        "image_path": image_path,
        "final_top1_path": top1_path,
        "final_top1_score": top1_score,
        "final_top1_extra": top1_extra,
        "ranked_paths": ranked_paths,
        "layer_scores": layer_scores,
        "layer_evidence_log": layer_evidence_log,
        "our_confidence": our_confidence,
        "topk_segments": {
            "k1_segments": [s["path"] for s in seg_k1],
            "k3_segments": [s["path"] for s in seg_k3],
            "k5_segments": [s["path"] for s in seg_k5]
        }
    }