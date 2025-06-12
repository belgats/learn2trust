import torch
import numpy as np
from typing import Tuple, Optional

class ProbabilisticDefense:
    """Probabilistic defense mechanism using VAE reconstruction error"""
    
    def __init__(self, vae, threshold_type="chebyshev", alpha=0.05):
        """
        Args:
            vae: Trained VAE model
            threshold_type: Type of bound to use ('chebyshev' or 'cantelli')
            alpha: Desired upper bound on false rejection rate
        """
        self.vae = vae
        self.threshold_type = threshold_type
        self.alpha = alpha
        self.benign_scores = []
        
    def compute_threshold(self, mean: float, std: float) -> float:
        """Compute acceptance threshold based on distribution statistics"""
        if self.threshold_type == "chebyshev":
            # Chebyshev's inequality: P(|X - μ| ≥ kσ) ≤ 1/k²
            k = np.sqrt(1 / self.alpha)
            return mean + k * std
        else:  # Cantelli's inequality
            # P(X - μ ≥ kσ) ≤ 1/(1 + k²)
            k = np.sqrt(1/self.alpha - 1)
            return mean + k * std
            
    def update_statistics(self, benign_updates: torch.Tensor):
        """Update benign distribution statistics using new updates"""
        with torch.no_grad():
            scores = self.vae.reconstruction_error(benign_updates)
            self.benign_scores.extend(scores.cpu().numpy())
            
    def check_update(self, 
                    update: torch.Tensor, 
                    confidence_threshold: Optional[float] = None
                    ) -> Tuple[bool, float]:
        """
        Check if an update is likely benign
        
        Args:
            update: Client update to check
            confidence_threshold: Optional override for acceptance threshold
            
        Returns:
            (is_benign, confidence_score)
        """
        with torch.no_grad():
            score = self.vae.reconstruction_error(update).item()
            
        if confidence_threshold is None:
            mean = np.mean(self.benign_scores)
            std = np.std(self.benign_scores)
            confidence_threshold = self.compute_threshold(mean, std)
            
        return score <= confidence_threshold, score
