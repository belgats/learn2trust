import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision import transforms
import numpy as np
from typing import Optional, List, Tuple
from .datasets import FederatedDataset

class FederatedImageDataset(FederatedDataset):
    """Federated dataset for image classification tasks"""
    
    def __init__(self, 
                 root: str,
                 dataset_name: str,
                 num_clients: int,
                 alpha: float = 0.5,
                 transform: Optional[transforms.Compose] = None,
                 seed: Optional[int] = None):
        """
        Args:
            root: Root directory for data
            dataset_name: Name of dataset ('cifar10', 'mnist', 'fashion_mnist')
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
            transform: Data transformations
            seed: Random seed
        """
        super().__init__(num_clients, alpha, seed)
        
        self.dataset_name = dataset_name.lower()
        if transform is None:
            if self.dataset_name == 'cifar10':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
                ])
            else:  # MNIST/Fashion-MNIST
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
        # Load dataset
        if self.dataset_name == 'cifar10':
            dataset = CIFAR10(root, train=True, download=True, transform=transform)
        elif self.dataset_name == 'mnist':
            dataset = MNIST(root, train=True, download=True, transform=transform)
        elif self.dataset_name == 'fashion_mnist':
            dataset = FashionMNIST(root, train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Convert to tensors
        data = torch.stack([transform(img) for img, _ in dataset])
        labels = torch.tensor([label for _, label in dataset])
        
        # Create non-IID partition
        partitions = self.create_non_iid_partition(data, labels)
        
        # Distribute data to clients
        for indices in partitions:
            client_data = data[indices]
            client_labels = labels[indices]
            self.client_data.append((client_data, client_labels))
            
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get input shape for the dataset"""
        return self.client_data[0][0].shape[1:]  # (C, H, W)
