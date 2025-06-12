import unittest
import torch
import torch.nn as nn
import numpy as np
from src.models.vae import VAE
from src.defense.probabilistic_defense import ProbabilisticDefense
from src.utils.visualize import visualize_update_reconstruction 

from pathlib import Path

class TestModel(nn.Module):
    """A tiny transformer for testing weight updates"""
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 2)  # Binary classification
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

class TestVAEDefense(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TestModel().to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        
        # Generate benign updates
        updates = []
        for _ in range(1000):
            update = torch.randn(self.total_params) * 0.01
            updates.append(update)
        self.benign_updates = torch.stack(updates)
        
        # Generate malicious updates
        num_malicious = 50
        self.mal_indices = np.random.choice(len(self.benign_updates), num_malicious, replace=False)
        self.malicious_updates = self.benign_updates.clone()
        self.malicious_updates[self.mal_indices] *= 10.0
        
        # Initialize VAE and defense
        self.vae = VAE(input_dim=self.total_params, hidden_dim=128, latent_dim=32).to(self.device)
        self.defense = ProbabilisticDefense(self.vae, threshold_type="chebyshev", alpha=0.05)

    def test_vae_training(self):
        """Test VAE training convergence"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        # Get initial loss
        loss, _ = self.vae.elbo_loss(self.benign_updates.to(self.device))
        initial_loss = loss.item()
        
        # Train for 50 epochs
        for _ in range(50):
            optimizer.zero_grad()
            loss, _ = self.vae.elbo_loss(self.benign_updates.to(self.device))
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss, "VAE training did not converge")

    def test_reconstruction_error(self):
        """Test reconstruction error computation"""
        errors = self.vae.reconstruction_error(self.benign_updates.to(self.device))
        self.assertEqual(errors.shape[0], len(self.benign_updates))
        self.assertTrue((errors >= 0).all(), "Reconstruction errors should be non-negative")

    def test_defense_mechanism(self):
        """Test defense mechanism detection capabilities"""
        # Train VAE
        optimizer = torch.optim.Adam(self.defense.vae.parameters(), lr=1e-3)
        for _ in range(50):
            optimizer.zero_grad()
            loss, _ = self.defense.vae.elbo_loss(self.benign_updates.to(self.device))
            loss.backward()
            optimizer.step()

        # Update defense statistics
        self.defense.update_statistics(self.benign_updates.to(self.device))
        
        # Test detection
        mal_updates = self.malicious_updates.to(self.device)
        mal_updates[list(self.mal_indices)] *= 100.0
        
        total_benign = total_malicious = 0
        correct_benign = correct_malicious = 0
        
        for i, update in enumerate(mal_updates):
            update = update.unsqueeze(0) if update.dim() == 1 else update
            is_benign, _ = self.defense.check_update(update)
            
            if i in self.mal_indices:
                total_malicious += 1
                if not is_benign:
                    correct_malicious += 1
            else:
                total_benign += 1
                if is_benign:
                    correct_benign += 1
        
        benign_accuracy = correct_benign / total_benign if total_benign > 0 else 0
        malicious_accuracy = correct_malicious / total_malicious if total_malicious > 0 else 0
        
        self.assertGreater(benign_accuracy, 0.8, f"Poor benign detection rate: {benign_accuracy:.2f}")
        self.assertGreater(malicious_accuracy, 0.7, f"Poor malicious detection rate: {malicious_accuracy:.2f}")
    def test_visualization(self):
        """Test visualization of update reconstruction"""
        # Train VAE first
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        for _ in range(50):
            optimizer.zero_grad()
            loss, _ = self.vae.elbo_loss(self.benign_updates.to(self.device))
            loss.backward()
            optimizer.step()
            

        example_update = self.benign_updates[0]
        visualize_update_reconstruction(self.vae, example_update,  'tests/figures/update_reconstruction.png', self.device)
        
        # Assert figure was created
        self.assertTrue(Path('tests/figures/update_reconstruction.png').exists())
if __name__ == '__main__':
    unittest.main()