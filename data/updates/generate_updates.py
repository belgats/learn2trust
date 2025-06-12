import torch
import numpy as np
import os
from pathlib import Path

def generate_and_save_updates(total_params, num_benign=1000, num_malicious=50, save_dir='data/updates'):
    """Generate and save synthetic weight updates for testing"""
    # Create directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate benign updates
    benign_updates = torch.randn(num_benign, total_params) * 0.01
    
    # Generate malicious updates
    mal_indices = np.random.choice(num_benign, num_malicious, replace=False)
    malicious_updates = benign_updates.clone()
    malicious_updates[mal_indices] *= 10.0
    
    # Save updates
    torch.save(benign_updates, save_dir / 'benign_updates.pt')
    torch.save(malicious_updates, save_dir / 'malicious_updates.pt')
    torch.save(mal_indices, save_dir / 'malicious_indices.pt')
    
    return benign_updates, malicious_updates, mal_indices

if __name__ == '__main__':
    # Example usage
    total_params = 100000  # Example size
    generate_and_save_updates(total_params)