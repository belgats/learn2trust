import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from PIL import Image
import torchvision.transforms as transforms

class FederatedDataset:
    """Base class for federated datasets with non-IID distribution"""
    
    def __init__(self, 
                 num_clients: int,
                 alpha: float = 0.5,  # Dirichlet concentration parameter
                 seed: Optional[int] = None):
        """
        Args:
            num_clients: Number of clients in federated setting
            alpha: Dirichlet concentration parameter (smaller alpha = more non-IID)
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.alpha = alpha
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.client_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
    def create_non_iid_partition(self, 
                               data: torch.Tensor,
                               labels: torch.Tensor) -> List[List[int]]:
        """
        Partition data among clients using Dirichlet distribution
        for non-IID setting
        """
        num_classes = len(torch.unique(labels))
        client_partitions = [[] for _ in range(self.num_clients)]
        
        # Group indices by class
        class_indices = [torch.where(labels == class_idx)[0] 
                        for class_idx in range(num_classes)]
        
        # Sample Dirichlet distribution for each class
        for indices in class_indices:
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            # Distribute indices according to proportions
            for client_idx, prop in enumerate(proportions):
                num_samples = int(prop * len(indices))
                client_partitions[client_idx].extend(
                    indices[len(indices)-num_samples:].tolist()
                )
                indices = indices[:len(indices)-num_samples]
                
        return client_partitions
        
    def get_client_data(self, client_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data for specific client"""
        return self.client_data[client_idx]
        
    def get_stats(self) -> Dict:
        """Get distribution statistics for all clients"""
        stats = {}
        for i in range(self.num_clients):
            _, labels = self.client_data[i]
            unique, counts = torch.unique(labels, return_counts=True)
            stats[f'client_{i}'] = {
                'samples_per_class': dict(zip(unique.tolist(), 
                                            counts.tolist())),
                'total_samples': len(labels)
            }
        return stats
