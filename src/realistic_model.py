"""Realistic Microbiome Model for publication-quality generation.

This module implements the RealisticMicrobiomeModel that combines:
- Encoder (VAE-style or hyperbolic)
- Zero-inflated decoder for realistic sparsity
- Diversity matching losses (alpha and beta)
- Co-exclusion loss for biological constraints
- Sparsity loss for matching real data patterns

The model is designed to generate microbiome compositions that are
statistically indistinguishable from real data.

References:
    Requirements 6.1-6.5 from realistic-microbiome-generation spec
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.zero_inflated import (
    ZeroInflatedDecoder,
    RealisticMicrobiomeEncoder,
)
from src.diversity_loss import DiversityMatchingLoss
from src.coexclusion_loss import CoexclusionLoss, load_default_coexclusion_pairs
from src.sparsity_loss import SparsityLoss, RareTaxaLoss


@dataclass
class RealisticTrainingConfig:
    """Configuration for realistic microbiome model training.
    
    Attributes:
        dataset: Dataset name ('american_gut' or 'hmp')
        num_taxa: Number of taxa in the model
        min_prevalence: Minimum prevalence threshold for filtering
        embedding_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Epochs to wait before early stopping
        lambda_kl: Weight for KL divergence loss
        lambda_sparse: Weight for sparsity loss
        lambda_alpha: Weight for alpha diversity loss
        lambda_beta: Weight for beta diversity loss
        lambda_coex: Weight for co-exclusion loss
        validation_frequency: Epochs between validation
        target_ks_pvalue: Target KS test p-value
        target_coex_compliance: Target co-exclusion compliance
    """
    # Data
    dataset: str = 'american_gut'
    num_taxa: int = 500
    min_prevalence: float = 0.01
    
    # Model
    embedding_dim: int = 64
    hidden_dim: int = 256
    
    # Training
    num_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-4
    early_stopping_patience: int = 20
    
    # Loss weights
    lambda_kl: float = 0.01
    lambda_sparse: float = 1.0
    lambda_alpha: float = 0.5
    lambda_beta: float = 0.5
    lambda_coex: float = 5.0
    
    # Validation
    validation_frequency: int = 5
    target_ks_pvalue: float = 0.05
    target_coex_compliance: float = 0.7


class RealisticMicrobiomeModel(nn.Module):
    """Complete model for realistic microbiome generation.
    
    Combines encoder, zero-inflated decoder, and all loss components
    to generate microbiome compositions that match real data distributions.
    
    The model uses:
    - VAE-style encoder for learning latent representations
    - Zero-inflated decoder for realistic sparsity patterns
    - Diversity matching losses for ecological patterns
    - Co-exclusion loss for biological constraints
    - Sparsity loss for matching real data sparsity
    
    Attributes:
        num_taxa: Number of taxa
        embedding_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
        encoder: VAE encoder network
        decoder: Zero-inflated decoder
        diversity_loss: Diversity matching loss module
        coexclusion_loss: Co-exclusion penalty module
        sparsity_loss: Sparsity matching loss module
        rare_taxa_loss: Rare taxa absence loss module
    """
    
    def __init__(
        self,
        num_taxa: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        coexclusion_pairs: Optional[List[Tuple[int, int]]] = None,
        target_sparsity: float = 0.7,
        taxon_prevalences: Optional[Tensor] = None,
        # Loss weights
        lambda_kl: float = 0.01,
        lambda_sparse: float = 1.0,
        lambda_alpha: float = 0.5,
        lambda_beta: float = 0.5,
        lambda_coex: float = 5.0,
        lambda_rare: float = 1.0,
        # Loss configuration
        coex_penalty_mode: str = 'soft',
        coex_threshold: float = 0.001,
        diversity_multi_scale: bool = True,
    ):
        """Initialize realistic microbiome model.
        
        Args:
            num_taxa: Number of taxa to generate
            embedding_dim: Dimension of latent space (default 64)
            hidden_dim: Hidden layer dimension (default 256)
            coexclusion_pairs: List of (taxon_i, taxon_j) co-exclusion pairs.
                              If None, uses default pairs.
            target_sparsity: Target mean sparsity (default 0.7)
            taxon_prevalences: Target prevalence for each taxon, shape (num_taxa,).
                              If None, only overall sparsity matching is used.
            lambda_kl: Weight for KL divergence loss (default 0.01)
            lambda_sparse: Weight for sparsity loss (default 1.0)
            lambda_alpha: Weight for alpha diversity loss (default 0.5)
            lambda_beta: Weight for beta diversity loss (default 0.5)
            lambda_coex: Weight for co-exclusion loss (default 5.0)
            lambda_rare: Weight for rare taxa loss (default 1.0)
            coex_penalty_mode: 'soft' or 'hard' penalty mode for co-exclusion
            coex_threshold: Abundance threshold for co-exclusion
            diversity_multi_scale: Use multi-scale RBF kernel for diversity loss
        """
        super().__init__()
        
        self.num_taxa = num_taxa
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Store loss weights
        self.lambda_kl = lambda_kl
        self.lambda_sparse = lambda_sparse
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.lambda_coex = lambda_coex
        self.lambda_rare = lambda_rare
        
        # Encoder
        self.encoder = RealisticMicrobiomeEncoder(
            num_taxa=num_taxa,
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # Zero-inflated decoder
        self.decoder = ZeroInflatedDecoder(
            latent_dim=embedding_dim,
            num_taxa=num_taxa,
            hidden_dim=hidden_dim
        )
        
        # Diversity matching loss
        self.diversity_loss = DiversityMatchingLoss(
            multi_scale=diversity_multi_scale,
            alpha_weight=1.0,
            beta_weight=1.0
        )
        
        # Co-exclusion loss
        if coexclusion_pairs is None:
            coexclusion_pairs = load_default_coexclusion_pairs()
        
        # Filter pairs to valid indices
        valid_pairs = [
            (i, j) for i, j in coexclusion_pairs 
            if i < num_taxa and j < num_taxa
        ]
        
        if valid_pairs:
            self.coexclusion_loss = CoexclusionLoss(
                coexclusion_pairs=valid_pairs,
                penalty_weight=1.0,  # Weight applied separately via lambda_coex
                penalty_mode=coex_penalty_mode,
                threshold=coex_threshold
            )
            self.has_coexclusion = True
        else:
            self.coexclusion_loss = None
            self.has_coexclusion = False
        
        # Sparsity loss
        self.sparsity_loss = SparsityLoss(
            target_sparsity=target_sparsity,
            taxon_prevalences=taxon_prevalences,
            sparsity_weight=1.0,
            prevalence_weight=1.0 if taxon_prevalences is not None else 0.0
        )
        
        # Rare taxa loss (if prevalences provided)
        if taxon_prevalences is not None:
            self.rare_taxa_loss = RareTaxaLoss(
                taxon_prevalences=taxon_prevalences,
                rare_threshold=0.01,
                absence_target=0.95,
                penalty_weight=1.0
            )
            self.has_rare_taxa_loss = True
        else:
            self.rare_taxa_loss = None
            self.has_rare_taxa_loss = False
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode compositions to latent distribution.
        
        Args:
            x: Input compositions (batch, num_taxa)
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        return self.encoder(x)
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        return self.encoder.reparameterize(mu, logvar)
    
    def decode(
        self, 
        z: Tensor, 
        temperature: float = 1.0
    ) -> Tensor:
        """Decode latent vectors to compositions.
        
        Args:
            z: Latent vectors (batch, embedding_dim)
            temperature: Sampling temperature
            
        Returns:
            Generated compositions (batch, num_taxa)
        """
        return self.decoder.sample(z, temperature=temperature)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass returning reconstruction and distribution params.
        
        Args:
            x: Input compositions (batch, num_taxa)
            
        Returns:
            Dict with:
                - reconstruction: Reconstructed compositions
                - mu: Latent mean
                - logvar: Latent log-variance
                - z: Sampled latent vector
                - presence_logits: Presence logits from decoder
                - presence_probs: Presence probabilities
                - abundance_mu: Abundance mean from decoder
                - abundance_logvar: Abundance log-variance from decoder
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        presence_logits, abundance_mu, abundance_logvar = self.decoder(z)
        presence_probs = torch.sigmoid(presence_logits)
        reconstruction = self.decoder.sample(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'presence_logits': presence_logits,
            'presence_probs': presence_probs,
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
        z = torch.randn(num_samples, self.embedding_dim, device=device)
        
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
    
    def compute_loss(
        self, 
        x: Tensor,
        real_data: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute composite loss with all components.
        
        Combines reconstruction, KL, sparsity, diversity, and co-exclusion losses.
        Returns individual loss components for logging.
        
        Args:
            x: Input compositions (batch, num_taxa)
            real_data: Optional real data for diversity matching.
                      If None, uses x as reference.
            
        Returns:
            Dict with:
                - total_loss: Combined weighted loss
                - reconstruction_loss: Reconstruction term
                - kl_loss: KL divergence term
                - sparsity_loss: Sparsity matching term
                - alpha_diversity_loss: Alpha diversity MMD
                - beta_diversity_loss: Beta diversity MMD
                - coexclusion_loss: Co-exclusion penalty
                - rare_taxa_loss: Rare taxa absence penalty (if applicable)
        """
        # Forward pass
        output = self.forward(x)
        
        # Use x as real data reference if not provided
        if real_data is None:
            real_data = x
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x, output)
        
        # KL divergence loss
        kl_loss = self.kl_loss(output)
        
        # Sparsity loss (uses presence probabilities)
        sparse_loss = self.sparsity_loss(output['presence_probs'])
        
        # Diversity losses (uses reconstructed compositions)
        generated = output['reconstruction']
        alpha_loss = self.diversity_loss.alpha_diversity_loss(generated, real_data)
        beta_loss = self.diversity_loss.beta_diversity_loss(generated, real_data)
        
        # Co-exclusion loss
        if self.has_coexclusion:
            coex_loss = self.coexclusion_loss(generated)
        else:
            coex_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Rare taxa loss
        if self.has_rare_taxa_loss:
            rare_loss = self.rare_taxa_loss(output['presence_probs'])
        else:
            rare_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Combine losses with weights
        total_loss = (
            recon_loss +
            self.lambda_kl * kl_loss +
            self.lambda_sparse * sparse_loss +
            self.lambda_alpha * alpha_loss +
            self.lambda_beta * beta_loss +
            self.lambda_coex * coex_loss +
            self.lambda_rare * rare_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'sparsity_loss': sparse_loss,
            'alpha_diversity_loss': alpha_loss,
            'beta_diversity_loss': beta_loss,
            'coexclusion_loss': coex_loss,
            'rare_taxa_loss': rare_loss
        }
    
    def compute_loss_with_real_batch(
        self,
        x: Tensor,
        real_batch: Tensor
    ) -> Dict[str, Tensor]:
        """Compute loss using a separate batch of real data for diversity matching.
        
        This is useful during training when you want to compare generated
        samples against a different batch of real samples.
        
        Args:
            x: Input compositions to reconstruct (batch, num_taxa)
            real_batch: Real data batch for diversity comparison (batch, num_taxa)
            
        Returns:
            Dict with all loss components
        """
        return self.compute_loss(x, real_data=real_batch)
    
    def update_loss_weights(
        self,
        lambda_kl: Optional[float] = None,
        lambda_sparse: Optional[float] = None,
        lambda_alpha: Optional[float] = None,
        lambda_beta: Optional[float] = None,
        lambda_coex: Optional[float] = None,
        lambda_rare: Optional[float] = None
    ) -> None:
        """Update loss weights dynamically.
        
        Useful for curriculum learning or hyperparameter tuning.
        
        Args:
            lambda_kl: New KL weight (if provided)
            lambda_sparse: New sparsity weight (if provided)
            lambda_alpha: New alpha diversity weight (if provided)
            lambda_beta: New beta diversity weight (if provided)
            lambda_coex: New co-exclusion weight (if provided)
            lambda_rare: New rare taxa weight (if provided)
        """
        if lambda_kl is not None:
            self.lambda_kl = lambda_kl
        if lambda_sparse is not None:
            self.lambda_sparse = lambda_sparse
        if lambda_alpha is not None:
            self.lambda_alpha = lambda_alpha
        if lambda_beta is not None:
            self.lambda_beta = lambda_beta
        if lambda_coex is not None:
            self.lambda_coex = lambda_coex
        if lambda_rare is not None:
            self.lambda_rare = lambda_rare
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights.
        
        Returns:
            Dict with all loss weight values
        """
        return {
            'lambda_kl': self.lambda_kl,
            'lambda_sparse': self.lambda_sparse,
            'lambda_alpha': self.lambda_alpha,
            'lambda_beta': self.lambda_beta,
            'lambda_coex': self.lambda_coex,
            'lambda_rare': self.lambda_rare
        }
    
    def compute_metrics(
        self,
        generated: Tensor,
        real: Tensor,
        threshold: float = 1e-6
    ) -> Dict[str, float]:
        """Compute evaluation metrics for generated samples.
        
        Args:
            generated: Generated compositions (batch, num_taxa)
            real: Real compositions for comparison (batch, num_taxa)
            threshold: Abundance threshold for presence/absence
            
        Returns:
            Dict with evaluation metrics
        """
        with torch.no_grad():
            metrics = {}
            
            # Sparsity metrics
            sparsity_metrics = self.sparsity_loss.compute_sparsity_metrics(generated)
            metrics.update({f'gen_{k}': v for k, v in sparsity_metrics.items()})
            
            # Real data sparsity for comparison
            real_sparsity_metrics = self.sparsity_loss.compute_sparsity_metrics(real)
            metrics.update({f'real_{k}': v for k, v in real_sparsity_metrics.items()})
            
            # Co-exclusion compliance
            if self.has_coexclusion:
                compliance = self.coexclusion_loss.compute_compliance(generated)
                metrics['coexclusion_compliance'] = compliance
            
            # Rare taxa metrics
            if self.has_rare_taxa_loss:
                rare_metrics = self.rare_taxa_loss.compute_rare_taxa_metrics(generated)
                metrics.update({f'rare_{k}': v for k, v in rare_metrics.items()})
            
            return metrics
    
    @classmethod
    def from_real_data(
        cls,
        real_data: Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        coexclusion_pairs: Optional[List[Tuple[int, int]]] = None,
        **kwargs
    ) -> 'RealisticMicrobiomeModel':
        """Create model with parameters derived from real data.
        
        Automatically computes target sparsity and prevalences from
        the provided real data.
        
        Args:
            real_data: Real microbiome compositions (n_samples, num_taxa)
            embedding_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
            coexclusion_pairs: Optional co-exclusion pairs
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Configured RealisticMicrobiomeModel instance
        """
        from src.sparsity_loss import (
            compute_target_sparsity_from_data,
            compute_target_prevalence_from_data
        )
        
        num_taxa = real_data.shape[1]
        
        # Compute target statistics from real data
        target_sparsity, _ = compute_target_sparsity_from_data(real_data)
        taxon_prevalences = compute_target_prevalence_from_data(real_data)
        
        return cls(
            num_taxa=num_taxa,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            coexclusion_pairs=coexclusion_pairs,
            target_sparsity=target_sparsity,
            taxon_prevalences=taxon_prevalences,
            **kwargs
        )
