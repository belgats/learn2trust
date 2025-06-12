import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Tuple, Dict
from .datasets import FederatedDataset

class FederatedTimeSeriesDataset(FederatedDataset):
    """Federated dataset for time series tasks"""
    
    def __init__(self,
                 sequences: torch.Tensor,  # Shape: (N, T, F)
                 labels: torch.Tensor,     # Shape: (N,) or (N, T)
                 num_clients: int,
                 sequence_length: Optional[int] = None,
                 stride: int = 1,
                 alpha: float = 0.5,
                 seed: Optional[int] = None):
        """
        Args:
            sequences: Input time series data (N samples, T timesteps, F features)
            labels: Labels for each sequence
            num_clients: Number of clients
            sequence_length: Length of sequences (for windowing)
            stride: Stride for sequence windowing
            alpha: Dirichlet concentration parameter
            seed: Random seed
        """
        super().__init__(num_clients, alpha, seed)
        
        self.sequence_length = sequence_length
        self.stride = stride
        
        if sequence_length is not None:
            # Create overlapping windows
            sequences, labels = self._create_windows(sequences, labels)
            
        # Create non-IID partition
        partitions = self.create_non_iid_partition(sequences, labels)
        
        # Distribute data to clients
        for indices in partitions:
            client_sequences = sequences[indices]
            client_labels = labels[indices]
            self.client_data.append((client_sequences, client_labels))
            
    def _create_windows(self, 
                       sequences: torch.Tensor,
                       labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create overlapping windows from sequences"""
        N, T, F = sequences.shape
        L = self.sequence_length
        
        # Calculate number of windows
        num_windows = ((T - L) // self.stride) + 1
        
        # Create windows
        windows = torch.zeros((N * num_windows, L, F))
        window_labels = torch.zeros((N * num_windows,) + labels.shape[1:])
        
        idx = 0
        for i in range(N):
            for j in range(0, T - L + 1, self.stride):
                windows[idx] = sequences[i, j:j+L]
                window_labels[idx] = labels[i]
                idx += 1
                
        return windows, window_labels
        
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get input shape for the dataset"""
        seq_data = self.client_data[0][0]
        return seq_data.shape[1:]  # (T, F) or (L, F) if windowed
