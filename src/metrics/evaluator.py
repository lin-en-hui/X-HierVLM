"""
Comprehensive hierarchical evaluation metrics.

Implements all metrics defined in the paper:
- Acc_L1 ~ Acc_L6: Per-level top-1 accuracy (Eq. 9)
- EPR: Error Propagation Rate (Eq. 10)
- HCA: Hierarchical Consistency Accuracy (Eq. 11)
- POR: Point-Overlap Ratio (Eq. 12)
- S-POR: Strict Point-Overlap Ratio (Eq. 13)
- TOR: Top-Overlap Ratio (Eq. 14)
- ECC: Evidence Chain Completeness (Eq. 15)
- EPC: Evidence-Prediction Consistency (Eq. 16)
"""

import os
import copy
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np


def calculate_all_metrics(
    results: List[Dict[str, Any]],
    epc_perturb_ratio: float = 0.3,
    epc_num_trials: int = 1
) -> Dict[str, float]:
    """
    Compute all evaluation metrics from a list of inference results.

    Args:
        results: List of result dicts. Each must contain:
            - final_top1_path: list[str] of length 6 (L1~L6)
            - ground_truth: dict with keys 'L1'~'L6' (or will be extracted)
            - layer_evidence_log: list of evidence dicts with 'Matching_Classes' and 'Final_Weight'
            - topk_segments: dict with 'k1_segments' (for HCA)
            - ranked_paths: list of (path, score, extra) for margin-based EPC fallback
        epc_perturb_ratio: Fraction of weakest evidence to remove for EPC (default 0.3).
        epc_num_trials: Number of perturbation trials for EPC (default 1).

    Returns:
        dict: Metric names (e.g., "Acc_L1", "HCA", "EPR", "POR", "S-POR", "TOR", "ECC", "EPC")
              with values as percentages (0~100).
    """
    N = len(results)
    if N == 0:
        return {}

    levels = ["L1", "L2", "L3", "L4", "L5", "L6"]

    # --------------------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------------------
    acc = {f"L{i}": 0 for i in range(1, 7)}
    hca_count = 0
    epr_num = 0
    epr_den = 0
    por_sum = 0.0
    spor_sum = 0.0
    tor_sum = 0.0
    ecc_sum = 0.0
    epc_correct = 0

    # --------------------------------------------------------------------
    # Loop over each sample
    # --------------------------------------------------------------------
    for res in results:
        # ---------- Extract data ----------
        pred_path = res.get("final_top1_path", [])
        while len(pred_path) < 6:
            pred_path.append("None")

        gt = res.get("ground_truth", {})
        # If ground_truth is not provided, try to use default keys
        if not gt:
            # Try to infer from keys like 'L1', 'L2' etc. in the result itself
            for lv in levels:
                if lv in res:
                    gt[lv] = res[lv]
            # If still empty, skip this sample for metrics requiring GT
            if not gt:
                continue

        # Get per-layer correctness flags
        correct_flags = []
        for i, lv in enumerate(levels):
            pred_val = pred_path[i].strip().lower() if i < len(pred_path) else "none"
            gt_val = gt.get(lv, "").strip().lower()
            if pred_val == gt_val:
                correct_flags.append(1)
                acc[lv] += 1
            else:
                correct_flags.append(0)

        # ---------- HCA (Eq. 11): full path correct ----------
        if all(correct_flags):
            hca_count += 1

        # ---------- EPR (Eq. 10): L6 wrong AND (L1 or L2 wrong) ----------
        if not correct_flags[5]:  # L6 wrong
            epr_den += 1
            if not correct_flags[0] or not correct_flags[1]:  # L1 or L2 wrong
                epr_num += 1

        # ---------- POR (Eq. 12): average of correct nodes ----------
        por_sum += sum(correct_flags) / 6.0

        # ---------- S-POR (Eq. 13): longest consecutive correct from root ----------
        # Strictly: max over a,b of (b-a+1) * product_{j=a}^b I(correct_j)
        # Since we want root-to-leaf sequence, we compute the longest prefix of corrects
        longest_prefix = 0
        for flag in correct_flags:
            if flag:
                longest_prefix += 1
            else:
                break
        # The formula allows any subarray, but "starting from root" implies prefix.
        # We use the prefix length divided by 6.
        spor_sum += longest_prefix / 6.0

        # ---------- TOR (Eq. 14): adjacent level consistency ----------
        tor_count = 0
        for i in range(5):
            if correct_flags[i] and correct_flags[i+1]:
                tor_count += 1
        tor_sum += tor_count / 5.0

        # ---------- ECC (Eq. 15): evidence coverage ----------
        evidence_log = res.get("layer_evidence_log", [])
        covered_layers = set()
        for ev in evidence_log:
            if ev.get("Final_Weight", 0.0) > 0:
                layer = ev.get("Layer", "")
                covered_layers.add(layer)
        ecc_sum += len(covered_layers) / 6.0

        # ---------- EPC (Eq. 16): evidence perturbation stability ----------
        epc_flag = compute_epc_for_sample(
            res,
            perturb_ratio=epc_perturb_ratio,
            num_trials=epc_num_trials
        )
        epc_correct += epc_flag

    # --------------------------------------------------------------------
    # Compile final metrics (as percentages)
    # --------------------------------------------------------------------
    metrics = {}
    for lv in levels:
        metrics[f"Acc_{lv}"] = (acc[lv] / N) * 100.0

    metrics["HCA"] = (hca_count / N) * 100.0

    # EPR: if denominator is 0, set to 0 (no errors to propagate)
    metrics["EPR"] = (epr_num / epr_den * 100.0) if epr_den > 0 else 0.0

    metrics["POR"] = (por_sum / N) * 100.0
    metrics["S-POR"] = (spor_sum / N) * 100.0
    metrics["TOR"] = (tor_sum / N) * 100.0
    metrics["ECC"] = (ecc_sum / N) * 100.0
    metrics["EPC"] = (epc_correct / N) * 100.0

    return metrics


# ============================================================================
# EPC Helper (exact implementation of Eq. 16)
# ============================================================================

def compute_epc_for_sample(
    result: Dict[str, Any],
    perturb_ratio: float = 0.3,
    num_trials: int = 1
) -> int:
    """
    Compute Evidence-Prediction Consistency (EPC) for a single sample.

    Removes the weakest `perturb_ratio` fraction of evidence (by Final_Weight)
    and checks if the top-1 species prediction remains unchanged.

    Returns:
        1 if consistent (species unchanged), 0 otherwise.
    """
    # Get original top-1 species from the final path
    pred_path = result.get("final_top1_path", [])
    if not pred_path:
        return 1  # No path, assume consistent (or skip)
    original_species = pred_path[-1].strip().lower()

    evidence_log = result.get("layer_evidence_log", [])
    if len(evidence_log) < 2:
        return 1  # Too little evidence to perturb meaningfully

    # Determine how many pieces to remove
    num_to_remove = max(1, int(len(evidence_log) * perturb_ratio))

    # Sort evidence by Final_Weight (ascending: weakest first)
    sorted_ev = sorted(evidence_log, key=lambda x: x.get("Final_Weight", 0.0))

    # We'll do multiple trials (if num_trials > 1, we take majority vote)
    consistent_count = 0

    for trial in range(num_trials):
        # For deterministic reproducibility, we remove the weakest `num_to_remove` items.
        # If num_trials > 1, we can optionally shuffle the pool of candidates,
        # but for simplicity we remove the globally weakest ones (most conservative).
        # To simulate randomness, we can sample from the bottom half, but deterministic
        # removal of the absolute weakest is a strong test and perfectly valid.
        kept_ev = sorted_ev[num_to_remove:]  # keep the stronger ones

        # Recompute class scores from the remaining evidence
        new_scores = defaultdict(float)
        for ev in kept_ev:
            weight = ev.get("Final_Weight", 0.0)
            matching_classes = ev.get("Matching_Classes", [])
            for cls in matching_classes:
                new_scores[cls] += weight

        if not new_scores:
            # No evidence left: cannot determine, assume consistent (or skip)
            consistent_count += 1
            continue

        # Get top-1 species from perturbed evidence
        perturbed_top1 = max(new_scores.items(), key=lambda x: x[1])[0].strip().lower()

        # Check if the species name matches the original
        # Note: species names may have underscores or spaces; normalize
        perturbed_species = perturbed_top1.replace('_', ' ').strip()
        original_species_clean = original_species.replace('_', ' ').strip()

        if perturbed_species == original_species_clean:
            consistent_count += 1

    # Return 1 if majority of trials are consistent (for num_trials=1, this is just the flag)
    return 1 if consistent_count >= (num_trials // 2 + 1) else 0


# ============================================================================
# Utility: print metrics in table format (for paper tables)
# ============================================================================

def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print metrics in a table format suitable for paper reproduction."""
    print("\n" + "=" * 90)
    print(f"📊 {title}")
    print("=" * 90)

    # Hierarchical accuracy row
    acc_row = "| Acc_L1 | Acc_L2 | Acc_L3 | Acc_L4 | Acc_L5 | Acc_L6 |"
    val_row = f"| {metrics.get('Acc_L1', 0):.2f}  | {metrics.get('Acc_L2', 0):.2f}  | {metrics.get('Acc_L3', 0):.2f}  | {metrics.get('Acc_L4', 0):.2f}  | {metrics.get('Acc_L5', 0):.2f}  | {metrics.get('Acc_L6', 0):.2f}  |"
    print(acc_row)
    print(val_row)

    # Consistency metrics
    print("\n| HCA (%) | EPR (%) | POR (%) | S-POR (%) | TOR (%) | ECC (%) | EPC (%) |")
    print(f"| {metrics.get('HCA', 0):.2f}   | {metrics.get('EPR', 0):.2f}   | {metrics.get('POR', 0):.2f}   | {metrics.get('S-POR', 0):.2f}     | {metrics.get('TOR', 0):.2f}   | {metrics.get('ECC', 0):.2f}   | {metrics.get('EPC', 0):.2f}   |")

    print("=" * 90 + "\n")


def save_metrics_to_csv(metrics: Dict[str, float], output_path: str):
    """Save metrics to a CSV file."""
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    print(f"✅ Metrics saved to: {output_path}")


# ============================================================================
# Compatibility wrapper for the original print_metrics style
# ============================================================================

def legacy_print_metrics(metrics: Dict[str, Any]):
    """
    Backward-compatible print for the old-style metrics dict.
    Use print_metrics_table instead for paper-format output.
    """
    print_metrics_table(metrics)
