"""
I/O utilities: image path resolution, CSV loading, result saving.
"""

import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from PIL import Image


def resolve_image_path(
    img_path: str,
    fallback_roots: Optional[List[str]] = None
) -> str:
    """
    Try multiple possible root prefixes if the original path doesn't exist.
    Useful for handling dataset path differences between machines.
    """
    if os.path.exists(img_path):
        return img_path

    if fallback_roots is None:
        fallback_roots = [
            "../../dataset_ani/Paper_Dataset_Splits_Massive/1_Dataset_Fit",
            "../../dataset_ani/Paper_Dataset_Splits/1_Dataset_Fit",
            "../dataset_ani/Paper_Dataset_Splits/1_Dataset_Fit",
        ]

    # Try replacing the root
    for root in fallback_roots:
        if "Paper_Dataset_Splits" in img_path:
            parts = img_path.split("Paper_Dataset_Splits")
            if len(parts) > 1:
                alt_path = os.path.join(root, parts[-1].lstrip("/\\"))
                if os.path.exists(alt_path):
                    return alt_path
        # Also try relative path from current dir
        alt_path = os.path.join(root, os.path.basename(img_path))
        if os.path.exists(alt_path):
            return alt_path

    # If still not found, return original (caller will handle error)
    return img_path


def load_ground_truth_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV with ground truth labels."""
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required = ["image_path", "L1", "L2", "L3", "L4", "L5", "L6"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    return df


def load_image(path: str) -> Image.Image:
    """Load image as PIL Image."""
    return Image.open(path).convert('RGB')


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_dir: str
) -> None:
    """Save detailed predictions and metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Detailed predictions
    records = []
    for res in results:
        record = {
            "image_path": res.get("image_path", ""),
            "final_species": res.get("final_decision", res.get("final_top1_path", [""])[-1] if res.get("final_top1_path") else ""),
            "decision_source": res.get("decision_source", "Whitebox"),
            "final_path": " -> ".join(res.get("final_top1_path", [])),
        }
        # Add ground truth if present
        gt = res.get("ground_truth", {})
        for i in range(1, 7):
            record[f"gt_L{i}"] = gt.get(f"L{i}", "")
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)