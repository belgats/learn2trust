import os
import torch
import numpy as np
from typing import Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from .datasets import FederatedDataset
from .image_datasets import FederatedImageDataset
from .text_datasets import FederatedTextDataset
from .timeseries_datasets import FederatedTimeSeriesDataset

def prepare_federated_dataset(
    dataset_type: str,
    num_clients: int,
    alpha: float = 0.5,
    seed: Optional[int] = None,
    **kwargs
) -> FederatedDataset:
    """
    Prepare federated dataset based on type
    
    Args:
        dataset_type: Type of dataset ('image', 'text', 'timeseries')
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        Initialized federated dataset
    """
    if dataset_type == 'image':
        return FederatedImageDataset(
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            **kwargs
        )
    elif dataset_type == 'text':
        return FederatedTextDataset(
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            **kwargs
        )
    elif dataset_type == 'timeseries':
        return FederatedTimeSeriesDataset(
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def visualize_data_distribution(
    dataset: FederatedDataset,
    save_path: Optional[str] = None
):
    """
    Visualize the data distribution across clients
    
    Args:
        dataset: Federated dataset
        save_path: Path to save visualization
    """
    stats = dataset.get_stats()
    num_clients = len(stats)
    num_classes = len(stats['client_0']['samples_per_class'])
    
    # Create matrix of class distributions
    dist_matrix = np.zeros((num_clients, num_classes))
    for i in range(num_clients):
        dist = stats[f'client_{i}']['samples_per_class']
        for class_idx, count in dist.items():
            dist_matrix[i, class_idx] = count
            
    # Normalize by row
    dist_matrix = dist_matrix / dist_matrix.sum(axis=1, keepdims=True)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, 
                cmap='YlOrRd',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Client {i}' for i in range(num_clients)])
    plt.title('Data Distribution Across Clients')
    plt.xlabel('Classes')
    plt.ylabel('Clients')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def validate_non_iid(dataset: FederatedDataset, threshold: float = 0.1) -> bool:
    """
    Validate that the dataset is sufficiently non-IID
    
    Args:
        dataset: Federated dataset
        threshold: Maximum allowed Jensen-Shannon divergence 
                  between client distributions
        
    Returns:
        True if dataset is sufficiently non-IID
    """
    stats = dataset.get_stats()
    num_clients = len(stats)
    
    # Calculate distribution for each client
    distributions = []
    for i in range(num_clients):
        dist = stats[f'client_{i}']['samples_per_class']
        total = stats[f'client_{i}']['total_samples']
        dist_array = np.array([count/total for count in dist.values()])
        distributions.append(dist_array)
        
    # Calculate pairwise Jensen-Shannon divergence
    max_js_div = 0
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            js_div = jensen_shannon_divergence(distributions[i], 
                                            distributions[j])
            max_js_div = max(max_js_div, js_div)
            
    return max_js_div >= threshold

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Jensen-Shannon divergence between two distributions"""
    m = 0.5 * (p + q)
    return 0.5 * (
        np.sum(p * np.log(p/m + 1e-10)) + 
        np.sum(q * np.log(q/m + 1e-10))
    )
