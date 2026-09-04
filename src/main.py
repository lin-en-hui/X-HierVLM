#!/usr/bin/env python3
"""
X-HierVLM: Interpretable Hierarchical Image Classification

Command-line interface for white-box, black-box, and dual-track inference.

Usage examples:
    # Single image (white-box only)
    python main.py --mode single --image path/to/image.jpg

    # Single image (dual-track with arbitration)
    python main.py --mode single --image path/to/image.jpg --dual

    # Batch evaluation from CSV (white-box only)
    python main.py --mode csv --csv val_ground_truth.csv --output ./results/

    # Batch evaluation from CSV (dual-track)
    python main.py --mode csv --csv val_ground_truth.csv --dual --output ./results/

    # Directory mode (all images in a folder)
    python main.py --mode dir --dir path/to/folder/ --dual

    # With custom attribute top-k
    python main.py --mode single --image test.jpg --attr_top_k 5
"""

import os
import sys
import argparse
import json
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ATTR_TOP_K, RELIABILITY_CSV
from inference.whitebox import infer_single_image
from inference.blackbox import BlackBoxPredictor, predict_bioclip
from arbitration.arbiter import DualTrackArbiter
from metrics.evaluator import calculate_all_metrics, print_metrics_table, save_metrics_to_csv
from utils.tree import load_tree, get_species_list_from_tree, find_path_for_species
from utils.io import load_ground_truth_csv, save_results, resolve_image_path


# ============================================================================
# Helper functions
# ============================================================================

def extract_ground_truth_from_path(image_path: str) -> Dict[str, str]:
    """
    Extract hierarchical labels from folder name.
    Assumes format: ID_Phylum_Class_Order_Family_Genus_species
    Example: 00349_Animalia_Arthropoda_Insecta_Coleoptera_Erotylidae_Cypherotylus_californicus
    """
    folder_name = os.path.basename(os.path.dirname(image_path))
    parts = folder_name.split('_')

    # Try different possible formats
    if len(parts) >= 7:
        return {
            "L1": parts[2],
            "L2": parts[3],
            "L3": parts[4],
            "L4": parts[5],
            "L5": parts[6],
            "L6": " ".join(parts[6:]) if len(parts) > 7 else parts[6]
        }
    elif len(parts) >= 6:
        # Try to infer
        return {
            "L1": parts[1] if len(parts) > 1 else "Unknown",
            "L2": parts[2] if len(parts) > 2 else "Unknown",
            "L3": parts[3] if len(parts) > 3 else "Unknown",
            "L4": parts[4] if len(parts) > 4 else "Unknown",
            "L5": parts[5] if len(parts) > 5 else "Unknown",
            "L6": " ".join(parts[5:]) if len(parts) > 6 else parts[5]
        }
    return {f"L{i}": "Unknown" for i in range(1, 7)}


def run_whitebox_single(
    image_path: str,
    attr_top_k: int = ATTR_TOP_K,
    print_output: bool = True
) -> Dict[str, Any]:
    """Run white-box inference on a single image."""
    return infer_single_image(image_path, print_output=print_output, attr_top_k=attr_top_k)


def run_dual_single(
    image_path: str,
    species_list: List[str],
    attr_top_k: int = ATTR_TOP_K,
    print_output: bool = True
) -> Dict[str, Any]:
    """Run dual-track inference on a single image."""
    arbiter = DualTrackArbiter(reliability_csv=RELIABILITY_CSV)
    return arbiter.arbitrate(
        image_path,
        species_list=species_list,
        attr_top_k=attr_top_k,
        print_output=print_output
    )


def run_whitebox_csv(
    csv_path: str,
    output_dir: str,
    attr_top_k: int = ATTR_TOP_K,
    max_samples: Optional[int] = None,
    print_progress: bool = True
) -> Dict[str, Any]:
    """Run white-box inference on all images in a CSV file."""
    df = load_ground_truth_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    results = []
    total = len(df)

    if print_progress:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=total, desc="White-box inference")
    else:
        iterator = df.iterrows()

    for idx, row in iterator:
        img_path = resolve_image_path(str(row['image_path']))
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            res = infer_single_image(img_path, print_output=False, attr_top_k=attr_top_k)
            # Add ground truth
            res["ground_truth"] = {f"L{i}": str(row[f"L{i}"]) for i in range(1, 7)}
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Compute metrics
    metrics = calculate_all_metrics(results)
    print_metrics_table(metrics, title="White-box Results")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, metrics, output_dir)
    save_metrics_to_csv(metrics, os.path.join(output_dir, "metrics.csv"))

    return {"results": results, "metrics": metrics}


def run_dual_csv(
    csv_path: str,
    output_dir: str,
    attr_top_k: int = ATTR_TOP_K,
    max_samples: Optional[int] = None,
    print_progress: bool = True
) -> Dict[str, Any]:
    """Run dual-track inference on all images in a CSV file."""
    df = load_ground_truth_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    species_list = get_species_list_from_tree()
    arbiter = DualTrackArbiter(reliability_csv=RELIABILITY_CSV)

    results = []
    total = len(df)

    if print_progress:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=total, desc="Dual-track inference")
    else:
        iterator = df.iterrows()

    for idx, row in iterator:
        img_path = resolve_image_path(str(row['image_path']))
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            res = arbiter.arbitrate(
                img_path,
                species_list=species_list,
                attr_top_k=attr_top_k,
                print_output=False
            )
            # Add ground truth
            res["ground_truth"] = {f"L{i}": str(row[f"L{i}"]) for i in range(1, 7)}
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Compute metrics
    metrics = calculate_all_metrics(results)
    print_metrics_table(metrics, title="Dual-Track Results")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, metrics, output_dir)
    save_metrics_to_csv(metrics, os.path.join(output_dir, "metrics.csv"))

    return {"results": results, "metrics": metrics}


def run_whitebox_directory(
    dir_path: str,
    output_dir: str,
    attr_top_k: int = ATTR_TOP_K,
    max_images: Optional[int] = None,
    print_progress: bool = True
) -> Dict[str, Any]:
    """Run white-box inference on all images in a directory."""
    # Find all images
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = []
    for ext in img_exts:
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))

    images = sorted(set(images))
    if max_images:
        images = images[:max_images]

    if not images:
        print(f"No images found in {dir_path}")
        return {"results": [], "metrics": {}}

    print(f"Found {len(images)} images in {dir_path}")

    results = []
    if print_progress:
        from tqdm import tqdm
        iterator = tqdm(images, desc="White-box inference")
    else:
        iterator = images

    for img_path in iterator:
        try:
            res = infer_single_image(img_path, print_output=False, attr_top_k=attr_top_k)
            # Extract ground truth from folder name
            gt = extract_ground_truth_from_path(img_path)
            res["ground_truth"] = gt
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    metrics = calculate_all_metrics(results)
    print_metrics_table(metrics, title="White-box Results")

    os.makedirs(output_dir, exist_ok=True)
    save_results(results, metrics, output_dir)
    save_metrics_to_csv(metrics, os.path.join(output_dir, "metrics.csv"))

    return {"results": results, "metrics": metrics}


def run_dual_directory(
    dir_path: str,
    output_dir: str,
    attr_top_k: int = ATTR_TOP_K,
    max_images: Optional[int] = None,
    print_progress: bool = True
) -> Dict[str, Any]:
    """Run dual-track inference on all images in a directory."""
    # Find all images
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = []
    for ext in img_exts:
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        images.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))

    images = sorted(set(images))
    if max_images:
        images = images[:max_images]

    if not images:
        print(f"No images found in {dir_path}")
        return {"results": [], "metrics": {}}

    print(f"Found {len(images)} images in {dir_path}")

    species_list = get_species_list_from_tree()
    arbiter = DualTrackArbiter(reliability_csv=RELIABILITY_CSV)

    results = []
    if print_progress:
        from tqdm import tqdm
        iterator = tqdm(images, desc="Dual-track inference")
    else:
        iterator = images

    for img_path in iterator:
        try:
            res = arbiter.arbitrate(
                img_path,
                species_list=species_list,
                attr_top_k=attr_top_k,
                print_output=False
            )
            # Extract ground truth from folder name
            gt = extract_ground_truth_from_path(img_path)
            res["ground_truth"] = gt
            results.append(res)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    metrics = calculate_all_metrics(results)
    print_metrics_table(metrics, title="Dual-Track Results")

    os.makedirs(output_dir, exist_ok=True)
    save_results(results, metrics, output_dir)
    save_metrics_to_csv(metrics, os.path.join(output_dir, "metrics.csv"))

    return {"results": results, "metrics": metrics}


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="X-HierVLM: Interpretable Hierarchical Image Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # White-box single image
  python main.py --mode single --image test.jpg

  # Dual-track single image
  python main.py --mode single --image test.jpg --dual

  # Batch CSV (white-box)
  python main.py --mode csv --csv val.csv --output ./results/

  # Batch CSV (dual-track)
  python main.py --mode csv --csv val.csv --dual --output ./results/

  # Directory (dual-track)
  python main.py --mode dir --dir /path/to/images/ --dual

  # With custom top-k and thresholds
  python main.py --mode single --image test.jpg --attr_top_k 3
"""
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        choices=["single", "csv", "dir"],
        required=True,
        help="Inference mode: single image, CSV batch, or directory"
    )

    parser.add_argument(
        "--image",
        help="Path to image file (required for single mode)"
    )

    parser.add_argument(
        "--csv",
        help="Path to CSV file with ground truth (required for csv mode)"
    )

    parser.add_argument(
        "--dir",
        help="Directory containing images (required for dir mode)"
    )

    # Optional arguments
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Enable dual-track arbitration (otherwise white-box only)"
    )

    parser.add_argument(
        "--attr_top_k",
        type=int,
        default=ATTR_TOP_K,
        help=f"Number of top attribute candidates per dimension (default: {ATTR_TOP_K})"
    )

    parser.add_argument(
        "--output",
        default="./results/",
        help="Output directory for results (default: ./results/)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for csv/dir mode)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and detailed output"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "single" and not args.image:
        parser.error("--image is required for single mode")
    if args.mode == "csv" and not args.csv:
        parser.error("--csv is required for csv mode")
    if args.mode == "dir" and not args.dir:
        parser.error("--dir is required for dir mode")

    # Resolve image path for single mode
    if args.mode == "single":
        args.image = resolve_image_path(args.image)
        if not os.path.exists(args.image):
            parser.error(f"Image not found: {args.image}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print banner
    print("=" * 80)
    print("X-HierVLM: Interpretable Hierarchical Image Classification")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Dual-track: {args.dual}")
    print(f"Attribute Top-K: {args.attr_top_k}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # ========================================================================
    # Run selected mode
    # ========================================================================

    if args.mode == "single":
        # --------------------------------------------------------------------
        # Single image mode
        # --------------------------------------------------------------------
        if args.dual:
            species_list = get_species_list_from_tree()
            result = run_dual_single(
                args.image,
                species_list,
                attr_top_k=args.attr_top_k,
                print_output=not args.quiet
            )
            # Save result
            with open(os.path.join(args.output, "single_result.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n✅ Result saved to: {os.path.join(args.output, 'single_result.json')}")

            # Print summary
            print("\n" + "-" * 40)
            print("SUMMARY")
            print("-" * 40)
            print(f"Final Decision: {result['final_decision']}")
            print(f"Source: {result['decision_source']}")
            print(f"Path: {' -> '.join(result.get('final_top1_path', []))}")
            print("-" * 40)

        else:
            result = run_whitebox_single(
                args.image,
                attr_top_k=args.attr_top_k,
                print_output=not args.quiet
            )
            # Save result
            with open(os.path.join(args.output, "single_result.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n✅ Result saved to: {os.path.join(args.output, 'single_result.json')}")

            # Print summary
            print("\n" + "-" * 40)
            print("SUMMARY")
            print("-" * 40)
            path = result.get('final_top1_path', [])
            print(f"Path: {' -> '.join(path) if path else 'None'}")
            print(f"Score: {result.get('final_top1_score', 0):.2f}")
            print(f"Confidence: {result.get('our_confidence', 0):.4f}")
            print("-" * 40)

    elif args.mode == "csv":
        # --------------------------------------------------------------------
        # CSV batch mode
        # --------------------------------------------------------------------
        if args.dual:
            run_dual_csv(
                args.csv,
                args.output,
                attr_top_k=args.attr_top_k,
                max_samples=args.max_samples,
                print_progress=not args.quiet
            )
        else:
            run_whitebox_csv(
                args.csv,
                args.output,
                attr_top_k=args.attr_top_k,
                max_samples=args.max_samples,
                print_progress=not args.quiet
            )

    elif args.mode == "dir":
        # --------------------------------------------------------------------
        # Directory mode
        # --------------------------------------------------------------------
        if args.dual:
            run_dual_directory(
                args.dir,
                args.output,
                attr_top_k=args.attr_top_k,
                max_images=args.max_samples,
                print_progress=not args.quiet
            )
        else:
            run_whitebox_directory(
                args.dir,
                args.output,
                attr_top_k=args.attr_top_k,
                max_images=args.max_samples,
                print_progress=not args.quiet
            )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()