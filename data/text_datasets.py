import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer
from .datasets import FederatedDataset

class FederatedTextDataset(FederatedDataset):
    """Federated dataset for text classification/sequence tasks"""
    
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 num_clients: int,
                 model_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 alpha: float = 0.5,
                 seed: Optional[int] = None):
        """
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            num_clients: Number of clients
            model_name: Pretrained model name for tokenizer
            max_length: Maximum sequence length
            alpha: Dirichlet concentration parameter
            seed: Random seed
        """
        super().__init__(num_clients, alpha, seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = torch.tensor(labels)
        
        # Create non-IID partition
        partitions = self.create_non_iid_partition(input_ids, labels)
        
        # Distribute data to clients
        for indices in partitions:
            client_input_ids = input_ids[indices]
            client_attention_mask = attention_mask[indices]
            client_labels = labels[indices]
            self.client_data.append({
                'input_ids': client_input_ids,
                'attention_mask': client_attention_mask,
                'labels': client_labels
            })
            
    def get_client_data(self, client_idx: int) -> Dict[str, torch.Tensor]:
        """Get data for specific client"""
        return self.client_data[client_idx]
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size
