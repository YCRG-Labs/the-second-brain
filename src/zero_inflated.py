"""Zero-Inflated Decoder for realistic microbiome generation.

This module implements a zero-inflated output distribution for generating
microbiome compositions with realistic sparsity patterns. The decoder outputs
both presence probabilities (Bernoulli) and abundance parameters (LogNormal).

The zero-inflated model addresses the common issue in microbiome data where
many taxa are absent (zero abundance) in most samples, which cannot be
captured by standard continuous distributions.

References:
    Xu, L., et al. (2015). Assessment and Selection of Competing Models for 
    Zero-Inflated Microbiome Data.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ZeroInflatedDecoder(nn.Module):
    """Decoder that outputs zero-inflated distributions.
    
    Outputs two heads:
    - Presence probability (Bernoulli): P(taxon present)
    - Abundance given presence (LogNormal): P(abundance | present)
    
    The final abundance is computed as:
        abundance = presence_mask * lognormal_sample
    
    where presence_mask ~ Bernoulli(sigmoid(presence_logits))
    and lognormal_sample ~ LogNormal(mu, sigma)
    
    Attributes:
        latent_dim: Dimension of input latent space
        num_taxa: Number of taxa to generate
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        num_taxa: int, 
        hidden_dim: int = 256
    ):
        """Initialize zero-inflated decoder.
        
        Args:
            latent_dim: Dimension of input latent space
            num_taxa: Number of taxa to generate
            hidden_dim: Hidden layer dimension (default 256)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_taxa = num_taxa
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Presence head: outputs logits for Bernoulli distribution
        # P(taxon present) = sigmoid(presence_logits)
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_taxa),
        )
        
        # Abundance head: outputs mu and logvar for LogNormal distribution
        # abundance | present ~ LogNormal(mu, exp(logvar/2))
        self.abundance_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_taxa),
        )
        
        self.abundance_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_taxa),
        )
        
        # Initialize abundance logvar to small values for stable training
        nn.init.constant_(self.abundance_logvar_head[-1].bias, -2.0)
    
    def forward(
        self, 
        z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning distribution parameters.
        
        Args:
            z: Latent representation (batch, latent_dim)
            
        Returns:
            Tuple of:
                - presence_logits: Logits for presence probability (batch, num_taxa)
                - abundance_mu: Mean of log-abundance (batch, num_taxa)
                - abundance_logvar: Log-variance of log-abundance (batch, num_taxa)
        """
        # Shared feature extraction
        features = self.shared_layers(z)
        
        # Presence head
        presence_logits = self.presence_head(features)
        
        # Abundance head
        abundance_mu = self.abundance_mu_head(features)
        abundance_logvar = self.abundance_logvar_head(features)
        
        # Clamp logvar for numerical stability
        abundance_logvar = torch.clamp(abundance_logvar, min=-10.0, max=5.0)
        
        return presence_logits, abundance_mu, abundance_logvar
    
    def sample(
        self, 
        z: Tensor, 
        temperature: float = 1.0,
        hard: bool = True
    ) -> Tensor:
        """Sample compositions using presence mask and abundance.
        
        Args:
            z: Latent representation (batch, latent_dim)
            temperature: Temperature for sampling (default 1.0)
                - Lower temperature = more deterministic
                - Higher temperature = more stochastic
            hard: If True, use hard (binary) presence mask; 
                  if False, use soft (probability) mask
            
        Returns:
            Sampled compositions (batch, num_taxa)
        """
        presence_logits, abundance_mu, abundance_logvar = self.forward(z)
        
        # Apply temperature to presence logits
        scaled_logits = presence_logits / temperature
        
        # Sample presence mask
        if hard:
            # Gumbel-softmax trick for differentiable hard sampling
            # or simple Bernoulli sampling during inference
            if self.training:
                # Use Gumbel-sigmoid for differentiable sampling
                presence_probs = torch.sigmoid(scaled_logits)
                uniform = torch.rand_like(presence_probs)
                gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
                presence_mask = torch.sigmoid(
                    (torch.log(presence_probs + 1e-10) - 
                     torch.log(1 - presence_probs + 1e-10) + 
                     gumbel_noise) / temperature
                )
                # Straight-through estimator: hard in forward, soft in backward
                presence_mask = (presence_mask > 0.5).float() - presence_mask.detach() + presence_mask
            else:
                # Hard Bernoulli sampling during inference
                presence_probs = torch.sigmoid(scaled_logits)
                presence_mask = (torch.rand_like(presence_probs) < presence_probs).float()
        else:
            # Soft presence mask (probabilities)
            presence_mask = torch.sigmoid(scaled_logits)
        
        # Sample abundance from LogNormal
        # LogNormal(mu, sigma) = exp(Normal(mu, sigma))
        std = torch.exp(0.5 * abundance_logvar)
        
        if self.training:
            # Reparameterization trick
            eps = torch.randn_like(std)
            log_abundance = abundance_mu + eps * std * temperature
        else:
            # Use mean during inference for more stable results
            # or sample with temperature
            if temperature == 1.0:
                eps = torch.randn_like(std)
                log_abundance = abundance_mu + eps * std
            else:
                # Interpolate between mean and sample based on temperature
                eps = torch.randn_like(std)
                log_abundance = abundance_mu + eps * std * temperature
        
        # Convert to abundance (exp of log-abundance)
        abundance = torch.exp(log_abundance)
        
        # Apply presence mask: absent taxa have exactly zero abundance
        composition = presence_mask * abundance
        
        # Normalize to sum to 1 (compositional constraint)
        composition = composition / (composition.sum(dim=-1, keepdim=True) + 1e-10)
        
        return composition
    
    def get_presence_probs(self, z: Tensor) -> Tensor:
        """Get presence probabilities without sampling.
        
        Args:
            z: Latent representation (batch, latent_dim)
            
        Returns:
            Presence probabilities (batch, num_taxa)
        """
        presence_logits, _, _ = self.forward(z)
        return torch.sigmoid(presence_logits)
    
    def get_expected_abundance(self, z: Tensor) -> Tensor:
        """Get expected abundance (mean of LogNormal).
        
        The mean of LogNormal(mu, sigma) is exp(mu + sigma^2/2).
        
        Args:
            z: Latent representation (batch, latent_dim)
            
        Returns:
            Expected abundance (batch, num_taxa)
        """
        presence_logits, abundance_mu, abundance_logvar = self.forward(z)
        
        # Presence probability
        presence_prob = torch.sigmoid(presence_logits)
        
        # Expected value of LogNormal: exp(mu + sigma^2/2)
        expected_log_abundance = abundance_mu + 0.5 * torch.exp(abundance_logvar)
        expected_abundance = torch.exp(expected_log_abundance)
        
        # Weight by presence probability
        expected_composition = presence_prob * expected_abundance
        
        # Normalize
        expected_composition = expected_composition / (
            expected_composition.sum(dim=-1, keepdim=True) + 1e-10
        )
        
        return expected_composition
    
    def log_prob(
        self, 
        z: Tensor, 
        target: Tensor,
        eps: float = 1e-10
    ) -> Dict[str, Tensor]:
        """Compute log probability of target compositions.
        
        For zero-inflated model:
        - P(x=0) = (1 - p) + p * P(LogNormal ≈ 0)  [approximately]
        - P(x>0) = p * LogNormal(x; mu, sigma)
        
        We simplify by treating zeros as "absent" and non-zeros as "present".
        
        Args:
            z: Latent representation (batch, latent_dim)
            target: Target compositions (batch, num_taxa)
            eps: Small constant for numerical stability
            
        Returns:
            Dict with:
                - presence_log_prob: Log probability of presence pattern
                - abundance_log_prob: Log probability of abundances (for present taxa)
                - total_log_prob: Combined log probability
        """
        presence_logits, abundance_mu, abundance_logvar = self.forward(z)
        
        # Determine which taxa are present (non-zero)
        is_present = (target > eps).float()
        
        # Presence log probability (Bernoulli)
        presence_log_prob = F.binary_cross_entropy_with_logits(
            presence_logits, is_present, reduction='none'
        )
        presence_log_prob = -presence_log_prob.sum(dim=-1)  # Sum over taxa
        
        # Abundance log probability (LogNormal for present taxa)
        # LogNormal log prob: -log(x) - log(sigma) - 0.5*log(2*pi) - (log(x)-mu)^2/(2*sigma^2)
        log_target = torch.log(target + eps)
        std = torch.exp(0.5 * abundance_logvar)
        
        # Standard normal log prob of (log(x) - mu) / sigma
        normalized = (log_target - abundance_mu) / (std + eps)
        log_prob_normal = -0.5 * (normalized ** 2 + torch.log(2 * torch.pi * std ** 2 + eps))
        
        # Subtract log(x) for LogNormal (Jacobian)
        log_prob_lognormal = log_prob_normal - log_target
        
        # Only count abundance log prob for present taxa
        abundance_log_prob = (is_present * log_prob_lognormal).sum(dim=-1)
        
        # Total log probability
        total_log_prob = presence_log_prob + abundance_log_prob
        
        return {
            'presence_log_prob': presence_log_prob,
            'abundance_log_prob': abundance_log_prob,
            'total_log_prob': total_log_prob
        }



class RealisticMicrobiomeEncoder(nn.Module):
    """Encoder for realistic microbiome generation.
    
    Encodes microbiome compositions into a latent space using
    a simple MLP architecture. Can be extended to use hyperbolic
    embeddings for phylogenetic structure preservation.
    
    Attributes:
        num_taxa: Number of input taxa
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        num_taxa: int,
        latent_dim: int = 64,
        hidden_dim: int = 256
    ):
        """Initialize encoder.
        
        Args:
            num_taxa: Number of input taxa
            latent_dim: Dimension of latent space (default 64)
            hidden_dim: Hidden layer dimension (default 256)
        """
        super().__init__()
        
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(num_taxa, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Output mean and log-variance for VAE-style encoding
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize logvar to small values
        nn.init.constant_(self.fc_logvar.bias, -2.0)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode compositions to latent space.
        
        Args:
            x: Input compositions (batch, num_taxa)
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        # Apply log transform for better numerical properties
        # Add small constant to avoid log(0)
        x_log = torch.log(x + 1e-10)
        
        # Encode
        h = self.encoder(x_log)
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


class RealisticMicrobiomeVAE(nn.Module):
    """VAE-based model for realistic microbiome generation.
    
    Combines an encoder with a ZeroInflatedDecoder to generate
    microbiome compositions with realistic sparsity patterns.
    
    This model can be used standalone or integrated with the
    diffusion model for improved generation quality.
    
    Attributes:
        num_taxa: Number of taxa
        latent_dim: Dimension of latent space
        encoder: Encoder network
        decoder: Zero-inflated decoder
    """
    
    def __init__(
        self,
        num_taxa: int,
        latent_dim: int = 64,
        hidden_dim: int = 256
    ):
        """Initialize VAE model.
        
        Args:
            num_taxa: Number of taxa
            latent_dim: Dimension of latent space (default 64)
            hidden_dim: Hidden layer dimension (default 256)
        """
        super().__init__()
        
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = RealisticMicrobiomeEncoder(
            num_taxa=num_taxa,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        # Zero-inflated decoder
        self.decoder = ZeroInflatedDecoder(
            latent_dim=latent_dim,
            num_taxa=num_taxa,
            hidden_dim=hidden_dim
        )
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode compositions to latent distribution.
        
        Args:
            x: Input compositions (batch, num_taxa)
            
        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)
    
    def decode(
        self, 
        z: Tensor, 
        temperature: float = 1.0
    ) -> Tensor:
        """Decode latent vectors to compositions.
        
        Args:
            z: Latent vectors (batch, latent_dim)
            temperature: Sampling temperature
            
        Returns:
            Generated compositions (batch, num_taxa)
        """
        return self.decoder.sample(z, temperature=temperature)
    
    def forward(
        self, 
        x: Tensor
    ) -> Dict[str, Tensor]:
        """Forward pass returning reconstruction and distribution params.
        
        Args:
            x: Input compositions (batch, num_taxa)
            
        Returns:
            Dict with:
                - reconstruction: Reconstructed compositions
                - mu: Latent mean
                - logvar: Latent log-variance
                - presence_logits: Presence logits from decoder
                - abundance_mu: Abundance mean from decoder
                - abundance_logvar: Abundance log-variance from decoder
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        presence_logits, abundance_mu, abundance_logvar = self.decoder(z)
        reconstruction = self.decoder.sample(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'presence_logits': presence_logits,
            'abundance_mu': abundance_mu,
            'abundance_logvar': abundance_logvar
        }
    
    def generate(
        self, 
        num_samples: int, 
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate new microbiome compositions.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature (default 1.0)
            device: Device to generate on (default: model's device)
            
        Returns:
            Generated compositions (num_samples, num_taxa)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior (standard normal)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode
        return self.decode(z, temperature=temperature)
    
    def reconstruction_loss(
        self, 
        x: Tensor, 
        output: Dict[str, Tensor]
    ) -> Tensor:
        """Compute reconstruction loss using log probability.
        
        Args:
            x: Original compositions
            output: Output from forward pass
            
        Returns:
            Negative log probability (reconstruction loss)
        """
        log_probs = self.decoder.log_prob(output['z'], x)
        return -log_probs['total_log_prob'].mean()
    
    def kl_loss(self, output: Dict[str, Tensor]) -> Tensor:
        """Compute KL divergence from prior.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        Args:
            output: Output from forward pass
            
        Returns:
            KL divergence
        """
        mu = output['mu']
        logvar = output['logvar']
        
        # KL divergence for Gaussian: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def loss(
        self, 
        x: Tensor, 
        beta: float = 1.0
    ) -> Dict[str, Tensor]:
        """Compute total VAE loss.
        
        Args:
            x: Input compositions
            beta: Weight for KL term (beta-VAE)
            
        Returns:
            Dict with reconstruction_loss, kl_loss, and total_loss
        """
        output = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, output)
        kl_loss = self.kl_loss(output)
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }


class ZeroInflatedDiffusionDecoder(nn.Module):
    """Decoder that integrates zero-inflation with diffusion output.
    
    This decoder takes the output from a diffusion model (in CLR space)
    and converts it to a zero-inflated composition. It learns to predict
    which taxa should be present and their abundances.
    
    Attributes:
        num_taxa: Number of taxa
        input_channels: Number of input channels from diffusion
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        num_taxa: int,
        input_channels: int = 16,
        hidden_dim: int = 256,
        image_size: int = 256
    ):
        """Initialize diffusion decoder.
        
        Args:
            num_taxa: Number of taxa
            input_channels: Number of channels from diffusion output
            hidden_dim: Hidden layer dimension
            image_size: Spatial size of diffusion output
        """
        super().__init__()
        
        self.num_taxa = num_taxa
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        
        # Flatten diffusion output and project to latent
        flat_dim = input_channels * image_size * image_size
        
        # Use adaptive pooling to handle variable sizes
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        pooled_dim = input_channels * 8 * 8
        
        # Project to latent space
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Zero-inflated decoder
        self.zi_decoder = ZeroInflatedDecoder(
            latent_dim=hidden_dim,
            num_taxa=num_taxa,
            hidden_dim=hidden_dim
        )
    
    def forward(
        self, 
        diffusion_output: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass from diffusion output to distribution params.
        
        Args:
            diffusion_output: Output from diffusion model (batch, C, H, W)
            
        Returns:
            Tuple of (presence_logits, abundance_mu, abundance_logvar)
        """
        # Pool and flatten
        pooled = self.pool(diffusion_output)
        flat = pooled.view(pooled.shape[0], -1)
        
        # Project to latent
        latent = self.projection(flat)
        
        # Get distribution parameters
        return self.zi_decoder(latent)
    
    def sample(
        self, 
        diffusion_output: Tensor, 
        temperature: float = 1.0
    ) -> Tensor:
        """Sample compositions from diffusion output.
        
        Args:
            diffusion_output: Output from diffusion model (batch, C, H, W)
            temperature: Sampling temperature
            
        Returns:
            Sampled compositions (batch, num_taxa)
        """
        # Pool and flatten
        pooled = self.pool(diffusion_output)
        flat = pooled.view(pooled.shape[0], -1)
        
        # Project to latent
        latent = self.projection(flat)
        
        # Sample from zero-inflated decoder
        return self.zi_decoder.sample(latent, temperature=temperature)
