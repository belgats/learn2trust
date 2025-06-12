import torch
from typing import Dict, List, Optional
from ..models.vae import UpdateVAE
from ..defense.probabilistic_defense import ProbabilisticDefense

class FederatedServer:
    """Central server for federated learning with VAE-based defense"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 input_dim: int,
                 defense_config: Optional[Dict] = None):
        """
        Args:
            model: Global model to be trained
            input_dim: Dimension of model updates
            defense_config: Configuration for defense mechanism
        """
        self.model = model
        self.defense_config = defense_config or {}
        
        # Initialize VAE for modeling update distribution
        self.vae = UpdateVAE(input_dim=input_dim)
        self.defense = ProbabilisticDefense(
            self.vae,
            threshold_type=self.defense_config.get('threshold_type', 'chebyshev'),
            alpha=self.defense_config.get('alpha', 0.05)
        )
        
    def aggregate_updates(self, 
                         client_updates: List[torch.Tensor],
                         round_num: int) -> torch.Tensor:
        """
        Aggregate client updates with defense mechanism
        
        Args:
            client_updates: List of parameter updates from clients
            round_num: Current training round
            
        Returns:
            Aggregated parameter update
        """
        accepted_updates = []
        
        for update in client_updates:
            is_benign, score = self.defense.check_update(update)
            if is_benign:
                accepted_updates.append(update)
                
        if len(accepted_updates) == 0:
            return torch.zeros_like(client_updates[0])
            
        # Simple averaging of accepted updates
        return torch.mean(torch.stack(accepted_updates), dim=0)
        
    def update_model(self, aggregated_update: torch.Tensor):
        """Apply aggregated update to global model"""
        with torch.no_grad():
            for param, update in zip(self.model.parameters(), aggregated_update):
                param.add_(update)
                
    def train_defense(self, benign_updates: torch.Tensor):
        """Train VAE on known benign updates"""
        # Train VAE here
        pass
        
    def update_defense_statistics(self, benign_updates: torch.Tensor):
        """Update defense mechanism with new benign updates"""
        self.defense.update_statistics(benign_updates)
