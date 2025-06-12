import torch
from typing import Dict, Optional

class FederatedClient:
    """Client for federated learning system"""
    
    def __init__(self, 
                 client_id: int,
                 model: torch.nn.Module,
                 is_malicious: bool = False,
                 attack_config: Optional[Dict] = None):
        """
        Args:
            client_id: Unique identifier for the client
            model: Local copy of the model
            is_malicious: Whether this client can perform attacks
            attack_config: Configuration for attack behavior
        """
        self.client_id = client_id
        self.model = model
        self.is_malicious = is_malicious
        self.attack_config = attack_config or {}
        
    def compute_update(self, 
                      data: torch.Tensor, 
                      labels: torch.Tensor,
                      global_params: Dict) -> torch.Tensor:
        """
        Compute parameter update based on local training
        
        Args:
            data: Training data
            labels: Training labels
            global_params: Current global model parameters
            
        Returns:
            Parameter update
        """
        # Store original parameters
        original_params = [p.clone() for p in self.model.parameters()]
        
        # Train model locally
        self.train_local(data, labels)
        
        # Compute parameter updates
        updates = []
        for new, old in zip(self.model.parameters(), original_params):
            updates.append(new.data - old.data)
            
        # If malicious, modify the update according to attack strategy
        if self.is_malicious and self.should_attack():
            updates = self.craft_malicious_update(updates)
            
        return updates
        
    def train_local(self, data: torch.Tensor, labels: torch.Tensor):
        """Perform local training"""
        # Local training implementation here
        pass
        
    def should_attack(self) -> bool:
        """Determine if client should perform attack in current round"""
        if not self.is_malicious:
            return False
            
        # Implement partial maliciousness logic based on attack_config
        attack_prob = self.attack_config.get('attack_probability', 1.0)
        return torch.rand(1).item() < attack_prob
        
    def craft_malicious_update(self, updates: torch.Tensor) -> torch.Tensor:
        """Create malicious update according to attack strategy"""
        # Implement attack logic here
        return updates
