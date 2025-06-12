import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder for modeling the distribution of client weight updates (Δw)
    with affine decoder transformation f(z) = Az + b
    """
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        """
        Args:
            input_dim: Dimension of weight updates (p in Δw ∈ Rᵖ)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space (z)
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder network (Δw → μ, Σ)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output μ and log(σ²) for latent distribution N(μ, Σ)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Affine decoder transformation f(z) = Az + b
        self.decoder_matrix = nn.Parameter(torch.randn(input_dim, latent_dim))  # A
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))  # b
        
    def encode(self, delta_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode weight updates into latent distribution parameters
        
        Args:
            delta_w: Weight updates Δw ∈ Rᵖ
            
        Returns:
            μ, log(σ²) of latent distribution
        """
        h = self.encoder(delta_w)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample z ~ N(μ, Σ) using reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector using affine transformation f(z) = Az + b
        
        Args:
            z: Latent vector z ∈ Rᵏ (k = latent_dim)
            
        Returns:
            Reconstructed weight update f(z) ∈ Rᵖ
        """
        return torch.matmul(self.decoder_matrix, z.unsqueeze(-1)).squeeze(-1) + self.decoder_bias
    
    def forward(self, delta_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            delta_w: Weight updates Δw ∈ Rᵖ
            
        Returns:
            (reconstructed_delta_w, μ, log(σ²))
        """
        mu, logvar = self.encode(delta_w)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def reconstruction_error(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        Compute squared L₂ reconstruction error ||Δwᵢ - Δŵᵢ||²₂
        
        Args:
            delta_w: Weight updates Δw ∈ Rᵖ
            
        Returns:
            Squared L₂ norm of the reconstruction error for each update
        """
        # Ensure input is 2D: (batch_size, input_dim)
        if delta_w.dim() == 1:
           delta_w = delta_w.unsqueeze(0)

        recon_delta_w, _, _ = self.forward(delta_w)
        return torch.norm(delta_w - recon_delta_w, dim=1).pow(2)  # squared L₂ norm
        
    def elbo_loss(self, delta_w: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss: 𝔼_q𝜙(z|Δw)[log p𝜃(Δw|z)] - KL(q𝜙(z|Δw)||p(z))
        where:
            - q𝜙(z|Δw) = N(μ, diag(σ²)) [encoder]
            - p𝜃(Δw|z) = Az + b [affine decoder]
            - p(z) = N(0, I) [prior]
        
        Args:
            delta_w: Weight updates Δw ∈ Rᵖ
            beta: Weight of the KL term (default=1.0)
            
        Returns:
            (ELBO loss, dictionary with individual loss terms)
        """
        # Get encoder distribution parameters and sample z
        mu, logvar = self.encode(delta_w)
        z = self.reparameterize(mu, logvar)
        recon_delta_w = self.decode(z)
        
        # 1. Reconstruction term: 𝔼_q𝜙(z|Δw)[log p𝜃(Δw|z)]
        # For Gaussian likelihood, this is proportional to the squared error
        recon_loss = -0.5 * torch.sum(
            (delta_w - recon_delta_w).pow(2)
        )
        
        # 2. KL divergence term: KL(q𝜙(z|Δw)||p(z))
        # For Gaussian encoder and standard normal prior, this has closed form:
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total ELBO loss (negative because we maximize ELBO)
        total_loss = -(recon_loss - beta * kl_loss)
        
        return total_loss, {
            'elbo': -total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item()
        }
