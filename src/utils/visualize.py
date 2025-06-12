import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_update_reconstruction(vae, update, save_dir='figures', device='cpu'):
    """Visualize original update vs VAE reconstruction"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure update is on the same device as VAE
    update = update.to( device)
    if update.dim() == 1:
        update = update.unsqueeze(0)
    
    # Get reconstruction
    with torch.no_grad():
        recon_update, mu, logvar = vae(update)
        
    # Convert to numpy for plotting
    original = update.cpu().numpy().flatten()
    reconstructed = recon_update.cpu().numpy().flatten()
    error = np.abs(original - reconstructed)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VAE Update Analysis', fontsize=16)
    
    # Plot original update
    sns.histplot(original, bins=50, ax=axes[0,0])
    axes[0,0].set_title('Original Update Distribution')
    axes[0,0].set_xlabel('Parameter Value')
    
    # Plot reconstructed update
    sns.histplot(reconstructed, bins=50, ax=axes[0,1])
    axes[0,1].set_title('Reconstructed Update Distribution')
    axes[0,1].set_xlabel('Parameter Value')
    
    # Plot reconstruction error
    sns.histplot(error, bins=50, ax=axes[1,0])
    axes[1,0].set_title('Reconstruction Error Distribution')
    axes[1,0].set_xlabel('Absolute Error')
    
    # Scatter plot: Original vs Reconstructed
    axes[1,1].scatter(original, reconstructed, alpha=0.1, s=1)
    axes[1,1].set_title('Original vs Reconstructed Values')
    axes[1,1].set_xlabel('Original Values')
    axes[1,1].set_ylabel('Reconstructed Values')
    
    # Add diagonal line for perfect reconstruction
    lims = [
        min(axes[1,1].get_xlim()[0], axes[1,1].get_ylim()[0]),
        max(axes[1,1].get_xlim()[1], axes[1,1].get_ylim()[1]),
    ]
    axes[1,1].plot(lims, lims, 'r--', alpha=0.5, label='Perfect Reconstruction')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'update_reconstruction.png')
    plt.close()

    # Print statistics
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
    print(f"Original update norm: {torch.norm(update).item():.6f}")
    print(f"Reconstructed update norm: {torch.norm(recon_update).item():.6f}")