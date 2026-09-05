"""
Dual-track cross-validation arbitration mechanism.

Implements the baseline calibration arbitration strategy based on:
- Historical precision reliability (from BioCLIP sampling)
- Adaptive correct margin (per-species confidence threshold)
- White-box evidence confidence (margin * structural integrity)

Decision branches:
    A1: Strong Consensus (bc_precision >= threshold)
    A2: Weak Consensus (bc_precision < threshold)
    B1: BioCLIP Wins (precision >= threshold AND bc_conf >= per-species margin)
    B3s: White-box Strong (precision < threshold, wb_confidence >= threshold)
    B3w: Both Weak (precision < threshold, wb_confidence < threshold)
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import pandas as pd
from PIL import Image

from config.settings import (
    RELIABILITY_CSV,
    BC_PRECISION_THRESH,
    WB_CONFIDENCE_THRESH
)
from inference.whitebox import infer_single_image
from inference.blackbox import BlackBoxPredictor
from utils.tree import load_tree, find_path_for_species, get_species_list_from_tree


class DualTrackArbiter:
    """
    Dual-track arbitrator fusing white-box evidence and black-box domain prior.
    """

    def __init__(
        self,
        reliability_csv: str = RELIABILITY_CSV,
        bc_precision_thresh: float = BC_PRECISION_THRESH,
        wb_confidence_thresh: float = WB_CONFIDENCE_THRESH
    ):
        """
        Args:
            reliability_csv: Path to BioCLIP empirical reliability CSV.
                Expected columns: class_name, precision_reliability, avg_correct_margin
            bc_precision_thresh: Minimum historical precision to trust BioCLIP.
            wb_confidence_thresh: Minimum white-box confidence for "strong" evidence.
        """
        self.bc_precision_thresh = bc_precision_thresh
        self.wb_confidence_thresh = wb_confidence_thresh

        # Load reliability data
        self.reliability_df = None
        self.precision_map = {}
        self.margin_map = {}

        if os.path.exists(reliability_csv):
            self.reliability_df = pd.read_csv(reliability_csv)
            # Build species name -> (precision, margin)
            # Assume class_name format is "Kingdom_Phylum_Class_Order_Family_Genus_species"
            # We extract the last two parts as "Genus species"
            def extract_species(name: str) -> str:
                parts = str(name).split('_')
                if len(parts) >= 2:
                    return " ".join(parts[-2:])
                return str(name)

            self.reliability_df['species'] = self.reliability_df['class_name'].apply(extract_species)
            self.precision_map = dict(zip(
                self.reliability_df['species'],
                self.reliability_df['precision_reliability']
            ))
            self.margin_map = dict(zip(
                self.reliability_df['species'],
                self.reliability_df['avg_correct_margin']
            ))
            print(f"[Arbiter] Loaded reliability for {len(self.precision_map)} species.")
        else:
            print(f"[Arbiter] Warning: Reliability CSV not found at {reliability_csv}. Using fallback defaults.")

        # Cache for black-box predictor
        self._bb_predictor = None
        self._bb_species_list = None

    def _get_bb_predictor(self, species_list: List[str]) -> BlackBoxPredictor:
        """Lazy-load BioCLIP predictor with cached text features."""
        if self._bb_predictor is None or self._bb_species_list != species_list:
            self._bb_predictor = BlackBoxPredictor(species_list)
            self._bb_species_list = species_list
        return self._bb_predictor

    def _get_historical_reliability(self, species: str) -> Tuple[float, float]:
        """
        Get historical precision and average correct margin for a species.
        Returns: (precision, margin). Defaults: (0.50, 0.85) if not found.
        """
        precision = self.precision_map.get(species, 0.50)
        margin = self.margin_map.get(species, 0.85)
        return precision, margin

    def arbitrate(
        self,
        image_path: str,
        species_list: Optional[List[str]] = None,
        attr_top_k: int = 5,
        print_output: bool = False
    ) -> Dict[str, Any]:
        """
        Run dual-track arbitration on a single image.

        Returns a comprehensive result dict with:
            - image_path: str
            - final_decision: str (species name)
            - decision_source: str (A1/A2/B1/B3s/B3w)
            - is_consensus: bool
            - bioclip: {species, confidence, precision, margin}
            - whitebox: {species, score, confidence, path}
            - final_top1_path: list[str] (full L1~L6 path for metrics)
            - layer_evidence_log: list[dict] (for ECC/EPC)
            - topk_segments: dict (for HCA/POR)
            - ground_truth: dict (if provided externally, merged later)
        """
        if species_list is None:
            species_list = get_species_list_from_tree()

        # --------------------------------------------------------------------
        # Step 1: White-box inference (full evidence chain)
        # --------------------------------------------------------------------
        if print_output:
            print(f"\n[Arbiter] White-box inference for: {os.path.basename(image_path)}")

        wb_result = infer_single_image(
            image_path,
            print_output=print_output,
            attr_top_k=attr_top_k
        )

        wb_path = wb_result.get("final_top1_path", [])
        wb_species = wb_path[-1] if wb_path else "Unknown"
        wb_score = wb_result.get("final_top1_score", 0.0)
        wb_confidence = wb_result.get("our_confidence", 0.0)
        wb_evidence = wb_result.get("layer_evidence_log", [])

        # --------------------------------------------------------------------
        # Step 2: Black-box inference (BioCLIP)
        # --------------------------------------------------------------------
        bb_predictor = self._get_bb_predictor(species_list)
        bc_species, bc_conf = bb_predictor.predict(image_path)
        bc_precision, bc_margin = self._get_historical_reliability(bc_species)

        if print_output:
            print(f"[Arbiter] BioCLIP: {bc_species} (conf={bc_conf:.4f}, precision={bc_precision:.3f}, margin={bc_margin:.3f})")
            print(f"[Arbiter] White-box: {wb_species} (score={wb_score:.2f}, confidence={wb_confidence:.4f})")

        # --------------------------------------------------------------------
        # Step 3: Arbitration decision tree
        # --------------------------------------------------------------------
        is_consensus = (wb_species.strip().lower() == bc_species.strip().lower()) if wb_species else False

        if is_consensus:
            final_species = wb_species if wb_species else bc_species
            if bc_precision >= self.bc_precision_thresh:
                decision_source = "A1_StrongConsensus"
            else:
                decision_source = "A2_WeakConsensus"
        else:
            # Use adaptive threshold: only adopt BioCLIP if confidence exceeds its typical margin
            # AND historical precision is sufficient
            if bc_precision >= self.bc_precision_thresh and bc_conf >= bc_margin:
                final_species = bc_species
                decision_source = "B1_BioCLIP"
            else:
                final_species = wb_species if wb_species else bc_species
                if wb_confidence >= self.wb_confidence_thresh:
                    decision_source = "B3s_WhiteboxStrong"
                else:
                    decision_source = "B3w_BothWeak"

        # --------------------------------------------------------------------
        # Step 4: Post-process: ensure full path exists for metrics
        # --------------------------------------------------------------------
        tree = load_tree()

        if decision_source == "B1_BioCLIP":
            # BioCLIP wins: find the correct taxonomic path in the tree
            full_path = find_path_for_species(final_species, tree)
            if full_path is not None and len(full_path) >= 6:
                wb_result["final_top1_path"] = full_path[:6]
                # Also update segments for HCA (assume fully connected if BioCLIP wins)
                wb_result["topk_segments"] = {
                    "k1_segments": [full_path[:6]],
                    "k3_segments": [full_path[:6]],
                    "k5_segments": [full_path[:6]]
                }
            else:
                # Fallback: construct from whitebox path or placeholder
                if wb_path and len(wb_path) >= 6:
                    wb_result["final_top1_path"] = wb_path[:6]
                else:
                    # Minimal fallback: use unknown placeholders
                    wb_result["final_top1_path"] = ["Unknown"] * 5 + [final_species]

        else:
            # White-box (or consensus) wins: use the white-box path directly
            if not wb_result.get("final_top1_path") or len(wb_result["final_top1_path"]) < 6:
                # Ensure we have a valid path
                if wb_path and len(wb_path) >= 6:
                    wb_result["final_top1_path"] = wb_path[:6]
                else:
                    # Try to find the path from the final species
                    full_path = find_path_for_species(final_species, tree)
                    if full_path and len(full_path) >= 6:
                        wb_result["final_top1_path"] = full_path[:6]
                    else:
                        wb_result["final_top1_path"] = ["Unknown"] * 5 + [final_species]

        # --------------------------------------------------------------------
        # Step 5: Build final output
        # --------------------------------------------------------------------
        result = {
            "image_path": image_path,
            "final_decision": final_species,
            "decision_source": decision_source,
            "is_consensus": is_consensus,
            "bioclip": {
                "species": bc_species,
                "confidence": bc_conf,
                "precision": bc_precision,
                "margin": bc_margin
            },
            "whitebox": {
                "species": wb_species,
                "score": wb_score,
                "confidence": wb_confidence,
                "path": wb_path
            },
            # These are used directly by metrics/evaluator
            "final_top1_path": wb_result.get("final_top1_path", []),
            "layer_evidence_log": wb_result.get("layer_evidence_log", []),
            "topk_segments": wb_result.get("topk_segments", {"k1_segments": [], "k3_segments": [], "k5_segments": []}),
            "ranked_paths": wb_result.get("ranked_paths", []),
            "layer_scores": wb_result.get("layer_scores", {}),
            "our_confidence": wb_result.get("our_confidence", 0.0),
            # Keep the raw whitebox result for debugging
            "_wb_raw": wb_result
        }

        if print_output:
            print(f"\n[Arbiter] Final Decision: {final_species} ({decision_source})")
            print(f"[Arbiter] Path: {' -> '.join(result['final_top1_path'])}")

        return result

    def arbitrate_batch(
        self,
        image_paths: List[str],
        species_list: Optional[List[str]] = None,
        attr_top_k: int = 5,
        print_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run arbitration on a list of images.
        """
        if species_list is None:
            species_list = get_species_list_from_tree()

        results = []
        iterator = image_paths
        if print_progress:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Arbitrating")

        for img_path in iterator:
            res = self.arbitrate(
                img_path,
                species_list=species_list,
                attr_top_k=attr_top_k,
                print_output=False
            )
            results.append(res)

        return results
