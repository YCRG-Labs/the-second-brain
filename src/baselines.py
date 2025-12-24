"""Baseline models for comparison with the main diffusion-based approach.

This module implements standard generative and predictive models:
- VAE: Variational Autoencoder for unconditional generation
- GAN: Generative Adversarial Network with Wasserstein loss
- CompositionalVAE: VAE with compositional constraints
- LSTM: Long Short-Term Memory for temporal prediction
- Transformer: Standard transformer encoder for temporal prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict, Any
import numpy as np


class BaselineVAE(nn.Module):
    """Variational Autoencoder baseline for microbiome generation.
    
    Standard VAE with Gaussian prior and posterior. Uses reparameterization
    trick for backpropagation through sampling.
    
    Args:
        num_taxa: Number of microbial taxa (input/output dimension)
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions for encoder/decoder
        metadata_dim: Dimension of metadata conditioning (0 for unconditional)
    """
    
    def __init__(
        self,
        num_taxa: int,
        latent_dim: int = 128,
        hidden_dims: list[int] = None,
        metadata_dim: int = 0
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        self.metadata_dim = metadata_dim
        
        # Encoder
        encoder_layers = []
        input_dim = num_taxa + metadata_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim + metadata_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(hidden_dims[0], num_taxa)
    
    def encode(self, x: Tensor, metadata: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.
        
        Args:
            x: Input compositions (batch_size, num_taxa)
            metadata: Optional metadata (batch_size, metadata_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        if metadata is not None:
            x = torch.cat([x, metadata], dim=1)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor, metadata: Optional[Tensor] = None) -> Tensor:
        """Decode latent vector to composition.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            metadata: Optional metadata (batch_size, metadata_dim)
            
        Returns:
            Reconstructed composition (batch_size, num_taxa)
        """
        if metadata is not None:
            z = torch.cat([z, metadata], dim=1)
        
        h = self.decoder(z)
        logits = self.fc_out(h)
        # Use softmax to ensure output is on simplex
        return F.softmax(logits, dim=1)
    
    def forward(
        self, 
        x: Tensor, 
        metadata: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input compositions (batch_size, num_taxa)
            metadata: Optional metadata (batch_size, metadata_dim)
            
        Returns:
            recon: Reconstructed compositions (batch_size, num_taxa)
            mu: Latent distribution mean (batch_size, latent_dim)
            logvar: Latent distribution log variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x, metadata)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, metadata)
        return recon, mu, logvar
    
    def sample(
        self, 
        num_samples: int, 
        metadata: Optional[Tensor] = None,
        device: str = 'cpu'
    ) -> Tensor:
        """Generate samples from prior.
        
        Args:
            num_samples: Number of samples to generate
            metadata: Optional metadata for conditional generation
            device: Device to generate samples on
            
        Returns:
            Generated compositions (num_samples, num_taxa)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decode(z, metadata)
        return samples
    
    def loss_function(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        beta: float = 1.0
    ) -> Dict[str, Tensor]:
        """Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon: Reconstructed compositions
            x: Original compositions
            mu: Latent distribution mean
            logvar: Latent distribution log variance
            beta: Weight for KL divergence term (beta-VAE)
            
        Returns:
            Dictionary with 'loss', 'recon_loss', and 'kl_loss'
        """
        # Reconstruction loss (cross-entropy for compositional data)
        recon_loss = F.kl_div(
            torch.log(recon + 1e-10),
            x,
            reduction='batchmean'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class BaselineGAN(nn.Module):
    """Generative Adversarial Network with Wasserstein loss.
    
    Implements WGAN-GP (Wasserstein GAN with Gradient Penalty) for
    stable training and better mode coverage.
    
    Args:
        num_taxa: Number of microbial taxa
        latent_dim: Dimension of noise vector
        metadata_dim: Dimension of metadata conditioning
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        num_taxa: int,
        latent_dim: int = 128,
        metadata_dim: int = 0,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        self.metadata_dim = metadata_dim
        
        # Generator
        gen_input_dim = latent_dim + metadata_dim
        self.generator = nn.Sequential(
            nn.Linear(gen_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_taxa)
        )
        
        # Discriminator (critic)
        disc_input_dim = num_taxa + metadata_dim
        self.discriminator = nn.Sequential(
            nn.Linear(disc_input_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def generate(
        self, 
        num_samples: int, 
        metadata: Optional[Tensor] = None,
        device: str = 'cpu'
    ) -> Tensor:
        """Generate samples using generator.
        
        Args:
            num_samples: Number of samples to generate
            metadata: Optional metadata for conditional generation
            device: Device to generate samples on
            
        Returns:
            Generated compositions (num_samples, num_taxa)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        if metadata is not None:
            z = torch.cat([z, metadata], dim=1)
        
        with torch.no_grad():
            logits = self.generator(z)
            samples = F.softmax(logits, dim=1)
        
        return samples
    
    def discriminate(
        self, 
        x: Tensor, 
        metadata: Optional[Tensor] = None
    ) -> Tensor:
        """Discriminate real vs fake samples.
        
        Args:
            x: Input compositions
            metadata: Optional metadata
            
        Returns:
            Discriminator scores
        """
        if metadata is not None:
            x = torch.cat([x, metadata], dim=1)
        return self.discriminator(x)
    
    def gradient_penalty(
        self,
        real_data: Tensor,
        fake_data: Tensor,
        metadata: Optional[Tensor] = None,
        lambda_gp: float = 10.0
    ) -> Tensor:
        """Compute gradient penalty for WGAN-GP.
        
        Args:
            real_data: Real samples
            fake_data: Generated samples
            metadata: Optional metadata
            lambda_gp: Gradient penalty weight
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        disc_interpolates = self.discriminate(interpolates, metadata)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty


class CompositionalVAE(BaselineVAE):
    """VAE with compositional constraints and Dirichlet prior.
    
    Extends BaselineVAE with:
    - Dirichlet prior in latent space
    - Explicit compositional constraints
    - Aitchison distance for reconstruction loss
    
    Args:
        num_taxa: Number of microbial taxa
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
        metadata_dim: Dimension of metadata conditioning
        concentration: Dirichlet concentration parameter
    """
    
    def __init__(
        self,
        num_taxa: int,
        latent_dim: int = 128,
        hidden_dims: list[int] = None,
        metadata_dim: int = 0,
        concentration: float = 1.0
    ):
        super().__init__(num_taxa, latent_dim, hidden_dims, metadata_dim)
        self.concentration = concentration
        
        # Replace standard Gaussian parameters with Dirichlet parameters
        self.fc_alpha = nn.Linear(
            hidden_dims[-1] if hidden_dims else 256, 
            latent_dim
        )
    
    def encode(
        self, 
        x: Tensor, 
        metadata: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Encode to Dirichlet parameters.
        
        Args:
            x: Input compositions
            metadata: Optional metadata
            
        Returns:
            alpha: Dirichlet concentration parameters
            logvar: Dummy for compatibility (not used)
        """
        if metadata is not None:
            x = torch.cat([x, metadata], dim=1)
        
        h = self.encoder(x)
        alpha = F.softplus(self.fc_alpha(h)) + self.concentration
        
        # Return dummy logvar for compatibility with parent class
        logvar = torch.zeros_like(alpha)
        return alpha, logvar
    
    def reparameterize(self, alpha: Tensor, logvar: Tensor) -> Tensor:
        """Sample from Dirichlet using Gamma reparameterization.
        
        Args:
            alpha: Dirichlet concentration parameters
            logvar: Unused (for compatibility)
            
        Returns:
            Sample from Dirichlet distribution
        """
        # Sample from Gamma distributions using reparameterization
        # Gamma(alpha, 1) can be sampled using Gamma(alpha+1, 1) * U^(1/alpha)
        # For simplicity, use the mean during training
        if self.training:
            # During training, use mean of Dirichlet (alpha / alpha.sum())
            return alpha / alpha.sum(dim=1, keepdim=True)
        else:
            # During inference, sample from Dirichlet
            gamma_samples = torch._sample_dirichlet(alpha)
            return gamma_samples
    
    def loss_function(
        self,
        recon: Tensor,
        x: Tensor,
        alpha: Tensor,
        logvar: Tensor,
        beta: float = 1.0
    ) -> Dict[str, Tensor]:
        """Compute compositional VAE loss.
        
        Args:
            recon: Reconstructed compositions
            x: Original compositions
            alpha: Dirichlet concentration parameters
            logvar: Unused (for compatibility)
            beta: Weight for KL term
            
        Returns:
            Dictionary with loss components
        """
        # Aitchison distance (geometric mean-based)
        log_recon = torch.log(recon + 1e-10)
        log_x = torch.log(x + 1e-10)
        
        # Center log-ratios
        clr_recon = log_recon - log_recon.mean(dim=1, keepdim=True)
        clr_x = log_x - log_x.mean(dim=1, keepdim=True)
        
        recon_loss = F.mse_loss(clr_recon, clr_x)
        
        # KL divergence to uniform Dirichlet
        alpha0 = alpha.sum(dim=1)
        concentration0 = self.concentration * self.latent_dim
        
        # Simplified KL divergence (approximation)
        kl_loss = (
            torch.lgamma(alpha0) - torch.lgamma(torch.tensor(concentration0))
            - torch.lgamma(alpha).sum(dim=1) + self.latent_dim * torch.lgamma(torch.tensor(self.concentration))
            + ((alpha - self.concentration) * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(1)))).sum(dim=1)
        ).mean()
        
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class BaselineLSTM(nn.Module):
    """LSTM baseline for temporal microbiome prediction.
    
    Multi-layer LSTM with autoregressive prediction capability.
    
    Args:
        input_dim: Dimension of input (num_taxa + metadata_dim)
        hidden_dim: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability between layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(
        self, 
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through LSTM.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            output: Predictions (batch_size, seq_len, input_dim)
            hidden: Final hidden state
        """
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.output_layer(lstm_out)
        output = F.softmax(logits, dim=-1)
        return output, hidden
    
    def predict(
        self,
        history: Tensor,
        horizon: int,
        metadata: Optional[Tensor] = None
    ) -> Tensor:
        """Autoregressively predict future states.
        
        Args:
            history: Historical sequence (batch_size, hist_len, input_dim)
            horizon: Number of steps to predict
            metadata: Optional time-varying metadata (batch_size, horizon, metadata_dim)
            
        Returns:
            Predictions (batch_size, horizon, input_dim)
        """
        batch_size = history.size(0)
        predictions = []
        
        # Initialize with history
        _, hidden = self.forward(history)
        current = history[:, -1:, :]  # Last timestep
        
        for t in range(horizon):
            # Predict next step
            pred, hidden = self.forward(current, hidden)
            predictions.append(pred)
            
            # Use prediction as input for next step
            current = pred
            
            # Optionally incorporate metadata
            if metadata is not None:
                current = torch.cat([current, metadata[:, t:t+1, :]], dim=-1)
        
        return torch.cat(predictions, dim=1)


class BaselineTransformer(nn.Module):
    """Standard Transformer encoder for temporal prediction.
    
    Vanilla transformer with positional encoding and temporal prediction head.
    
    Args:
        input_dim: Dimension of input (num_taxa + metadata_dim)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through transformer.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Output sequence (batch_size, seq_len, input_dim)
        """
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, mask=mask)
        
        # Project to output
        logits = self.output_projection(x)
        output = F.softmax(logits, dim=-1)
        
        return output
    
    def predict(
        self,
        history: Tensor,
        horizon: int,
        metadata: Optional[Tensor] = None
    ) -> Tensor:
        """Predict future states.
        
        Args:
            history: Historical sequence (batch_size, hist_len, input_dim)
            horizon: Number of steps to predict
            metadata: Optional metadata
            
        Returns:
            Predictions (batch_size, horizon, input_dim)
        """
        batch_size = history.size(0)
        seq_len = history.size(1)
        
        # Create causal mask for autoregressive prediction
        predictions = []
        current_seq = history
        
        for t in range(horizon):
            # Predict next step
            output = self.forward(current_seq)
            next_pred = output[:, -1:, :]  # Last timestep prediction
            predictions.append(next_pred)
            
            # Append prediction to sequence
            current_seq = torch.cat([current_seq, next_pred], dim=1)
            
            # Optionally incorporate metadata
            if metadata is not None and t < metadata.size(1):
                current_seq[:, -1, :] = torch.cat([
                    next_pred.squeeze(1), 
                    metadata[:, t, :]
                ], dim=-1)
        
        return torch.cat(predictions, dim=1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
