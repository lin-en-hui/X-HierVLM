"""
Black-box inference branch using BioCLIP 2.5.
Provides domain-specific prior predictions with confidence scores.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional

from config.settings import DEVICE
from models.loader import load_bioclip


class BlackBoxPredictor:
    """
    BioCLIP-based black-box predictor with precomputed text features.

    Usage:
        predictor = BlackBoxPredictor(species_list)
        species, confidence = predictor.predict("path/to/image.jpg")
    """

    def __init__(self, species_list: List[str], model_path: Optional[str] = None):
        """
        Args:
            species_list: List of all species names (L6 level) for classification.
            model_path: Optional custom path for BioCLIP weights (default uses hub).
        """
        self.species_list = species_list
        self.model_path = model_path

        # Load model, preprocess function, and precomputed text features
        print(f"[BlackBox] Loading BioCLIP for {len(species_list)} species...")
        self.model, self.preprocess, self.text_features = load_bioclip(species_list)
        self.device = self.text_features.device
        print(f"[BlackBox] BioCLIP loaded. Text features shape: {self.text_features.shape}")

    @torch.no_grad()
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict species for a single image.

        Returns:
            (predicted_species, confidence_score)
            Confidence is the softmax probability of the top-1 class.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Encode image and normalize
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)

            # Compute similarity (logits) and softmax probabilities
            similarity = 100.0 * image_features @ self.text_features.T
            probs = similarity.softmax(dim=-1)

        top_prob, top_idx = probs[0].max(dim=0)
        pred_species = self.species_list[top_idx.item()]
        confidence = top_prob.item()

        return pred_species, confidence

    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """
        Predict species for a batch of images.

        Returns:
            List of (predicted_species, confidence_score) for each image.
        """
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    @torch.no_grad()
    def predict_with_topk(self, image_path: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top-k species for a single image.

        Returns:
            List of (species, confidence) for top-k predictions.
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            similarity = 100.0 * image_features @ self.text_features.T
            probs = similarity.softmax(dim=-1)

        top_probs, top_indices = probs[0].topk(min(k, len(self.species_list)))
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append((self.species_list[idx.item()], prob.item()))
        return results

    def get_text_features(self) -> torch.Tensor:
        """Return the precomputed text feature matrix (for debugging or analysis)."""
        return self.text_features

    def get_species_list(self) -> List[str]:
        """Return the species list used for classification."""
        return self.species_list


# ========== Convenience function (compatible with the old API) ==========

def predict_bioclip(image_path: str, species_list: List[str]) -> Tuple[str, float]:
    """
    Standalone function for single-image prediction.
    Creates a temporary predictor instance (cached internally if needed).

    For repeated calls, instantiate BlackBoxPredictor once and reuse.
    """
    # Simple global cache to avoid reloading for the same species list
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None or _CACHED_PREDICTOR.get_species_list() != species_list:
        _CACHED_PREDICTOR = BlackBoxPredictor(species_list)
    return _CACHED_PREDICTOR.predict(image_path)


# Global cache for the convenience function
_CACHED_PREDICTOR = None


# ========== Optional: Module-level fast path for batch evaluation ==========

def batch_predict_bioclip(image_paths: List[str], species_list: List[str]) -> List[Tuple[str, float]]:
    """
    Batch prediction using a shared predictor instance.
    """
    predictor = BlackBoxPredictor(species_list)
    return predictor.predict_batch(image_paths)