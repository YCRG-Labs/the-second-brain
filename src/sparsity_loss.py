"""Sparsity Loss for realistic microbiome generation.

This module implements a loss function that encourages generated microbiome
compositions to match the sparsity patterns observed in real microbiome data.

Microbiome data is characterized by high sparsity (many zeros) due to:
1. True biological absence of taxa
2. Detection limits of sequencing methods
3. Ecological constraints limiting co-occurrence

The SparsityLoss ensures generated samples have:
- Overall sparsity matching real data distribution
- Per-taxon prevalence matching real data patterns
- Rare taxa appropriately absent in most samples

References:
    Weiss, S., et al. (2017). Normalization and microbial differential 
    abundance strategies depend upon data characteristics.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_sparsity(
    compositions: Tensor,
    threshold: float = 1e-6
) -> Tensor:
    """Compute sparsity (fraction of zeros) for each sample.
    
    Sparsity is defined as the proportion of taxa with abundance
    below the threshold in each sample.
    
    Args:
        compositions: Relative abundance tensor of shape (batch, num_taxa)
        threshold: Abundance threshold below which a taxon is considered absent
    
    Returns:
        Sparsity values of shape (batch,) in range [0, 1]
    """
    # Count taxa below threshold
    is_absent = (compositions < threshold).float()
    sparsity = is_absent.mean(dim=-1)
    return sparsity


def compute_prevalence(
    compositions: Tensor,
    threshold: float = 1e-6
) -> Tensor:
    """Compute prevalence (fraction of samples present) for each taxon.
    
    Prevalence is defined as the proportion of samples where a taxon
    has abundance above the threshold.
    
    Args:
        compositions: Relative abundance tensor of shape (batch, num_taxa)
        threshold: Abundance threshold above which a taxon is considered present
    
    Returns:
        Prevalence values of shape (num_taxa,) in range [0, 1]
    """
    # Count samples where each taxon is present
    is_present = (compositions >= threshold).float()
    prevalence = is_present.mean(dim=0)
    return prevalence


def compute_target_sparsity_from_data(
    real_data: Tensor,
    threshold: float = 1e-6
) -> Tuple[float, float]:
    """Compute target sparsity statistics from real data.
    
    Args:
        real_data: Real microbiome compositions of shape (n_samples, num_taxa)
        threshold: Abundance threshold for presence/absence
    
    Returns:
        Tuple of (mean_sparsity, std_sparsity)
    """
    sparsity = compute_sparsity(real_data, threshold)
    return sparsity.mean().item(), sparsity.std().item()


def compute_target_prevalence_from_data(
    real_data: Tensor,
    threshold: float = 1e-6
) -> Tensor:
    """Compute target per-taxon prevalence from real data.
    
    Args:
        real_data: Real microbiome compositions of shape (n_samples, num_taxa)
        threshold: Abundance threshold for presence/absence
    
    Returns:
        Target prevalence for each taxon, shape (num_taxa,)
    """
    return compute_prevalence(real_data, threshold)


class SparsityLoss(nn.Module):
    """Loss for matching sparsity patterns in generated microbiome data.
    
    This loss encourages generated samples to have:
    1. Overall sparsity matching the target distribution
    2. Per-taxon prevalence matching real data patterns
    
    The loss supports two modes:
    - Overall sparsity matching: Penalizes deviation from target mean sparsity
    - Per-taxon prevalence matching: Penalizes deviation from target prevalence per taxon
    
    Attributes:
        target_sparsity: Target mean sparsity (fraction of zeros)
        taxon_prevalences: Target prevalence for each taxon
        sparsity_weight: Weight for overall sparsity loss
        prevalence_weight: Weight for per-taxon prevalence loss
        threshold: Abundance threshold for presence/absence determination
    """
    
    def __init__(
        self,
        target_sparsity: float,
        taxon_prevalences: Optional[Tensor] = None,
        sparsity_weight: float = 1.0,
        prevalence_weight: float = 1.0,
        threshold: float = 1e-6
    ):
        """Initialize sparsity loss.
        
        Args:
            target_sparsity: Target mean sparsity (fraction of zeros).
                            Should be in range [0, 1]. Typical microbiome
                            data has sparsity around 0.7-0.9.
            taxon_prevalences: Target prevalence for each taxon, shape (num_taxa,).
                              If None, only overall sparsity matching is used.
            sparsity_weight: Weight for overall sparsity loss component (default 1.0)
            prevalence_weight: Weight for per-taxon prevalence loss component (default 1.0)
            threshold: Abundance threshold for presence/absence (default 1e-6)
        
        Raises:
            ValueError: If target_sparsity is not in [0, 1]
        """
        super().__init__()
        
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError(f"target_sparsity must be in [0, 1], got {target_sparsity}")
        
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.prevalence_weight = prevalence_weight
        self.threshold = threshold
        
        # Register taxon prevalences as buffer if provided
        if taxon_prevalences is not None:
            if not isinstance(taxon_prevalences, Tensor):
                taxon_prevalences = torch.tensor(taxon_prevalences, dtype=torch.float32)
            self.register_buffer('taxon_prevalences', taxon_prevalences)
        else:
            self.taxon_prevalences = None
    
    def overall_sparsity_loss(
        self,
        presence_probs: Tensor
    ) -> Tensor:
        """Compute loss for matching overall sparsity.
        
        Uses presence probabilities (soft) for differentiability during training.
        The loss penalizes deviation from target sparsity.
        
        Args:
            presence_probs: Presence probabilities of shape (batch, num_taxa)
                           Values should be in [0, 1]
        
        Returns:
            Overall sparsity loss (scalar tensor)
        """
        # Expected sparsity = 1 - mean(presence_prob)
        # This is differentiable through presence_probs
        expected_presence = presence_probs.mean(dim=-1)  # (batch,)
        expected_sparsity = 1.0 - expected_presence.mean()  # scalar
        
        # L2 loss on sparsity deviation
        sparsity_deviation = expected_sparsity - self.target_sparsity
        loss = sparsity_deviation ** 2
        
        return loss
    
    def prevalence_loss(
        self,
        presence_probs: Tensor
    ) -> Tensor:
        """Compute loss for matching per-taxon prevalence.
        
        Penalizes deviation of generated prevalence from target prevalence
        for each taxon.
        
        Args:
            presence_probs: Presence probabilities of shape (batch, num_taxa)
        
        Returns:
            Prevalence matching loss (scalar tensor)
        """
        if self.taxon_prevalences is None:
            return torch.tensor(0.0, device=presence_probs.device, dtype=presence_probs.dtype)
        
        # Expected prevalence = mean presence probability across batch
        expected_prevalence = presence_probs.mean(dim=0)  # (num_taxa,)
        
        # Ensure target is on same device
        target = self.taxon_prevalences.to(presence_probs.device)
        
        # L2 loss on prevalence deviation per taxon
        prevalence_deviation = expected_prevalence - target
        loss = (prevalence_deviation ** 2).mean()
        
        return loss
    
    def forward(
        self,
        presence_probs: Tensor
    ) -> Tensor:
        """Compute combined sparsity loss.
        
        Args:
            presence_probs: Presence probabilities of shape (batch, num_taxa)
                           Values should be in [0, 1]. Can be obtained from
                           ZeroInflatedDecoder.get_presence_probs() or
                           torch.sigmoid(presence_logits).
        
        Returns:
            Combined sparsity loss (scalar tensor)
        """
        overall_loss = self.overall_sparsity_loss(presence_probs)
        prevalence_loss = self.prevalence_loss(presence_probs)
        
        total_loss = (
            self.sparsity_weight * overall_loss +
            self.prevalence_weight * prevalence_loss
        )
        
        return total_loss
    
    def forward_with_components(
        self,
        presence_probs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute sparsity loss with individual components.
        
        Args:
            presence_probs: Presence probabilities of shape (batch, num_taxa)
        
        Returns:
            Tuple of (total_loss, overall_sparsity_loss, prevalence_loss)
        """
        overall_loss = self.overall_sparsity_loss(presence_probs)
        prevalence_loss = self.prevalence_loss(presence_probs)
        
        total_loss = (
            self.sparsity_weight * overall_loss +
            self.prevalence_weight * prevalence_loss
        )
        
        return total_loss, overall_loss, prevalence_loss
    
    def compute_sparsity_metrics(
        self,
        compositions: Tensor
    ) -> dict:
        """Compute sparsity metrics for generated compositions.
        
        This method uses hard thresholding for evaluation (not training).
        
        Args:
            compositions: Generated compositions of shape (batch, num_taxa)
        
        Returns:
            Dict with:
                - mean_sparsity: Mean sparsity across samples
                - std_sparsity: Std of sparsity across samples
                - target_sparsity: Target sparsity
                - sparsity_deviation: Absolute deviation from target
                - prevalence_mse: MSE of prevalence if taxon_prevalences provided
        """
        with torch.no_grad():
            # Compute actual sparsity
            sparsity = compute_sparsity(compositions, self.threshold)
            mean_sparsity = sparsity.mean().item()
            std_sparsity = sparsity.std().item()
            
            metrics = {
                'mean_sparsity': mean_sparsity,
                'std_sparsity': std_sparsity,
                'target_sparsity': self.target_sparsity,
                'sparsity_deviation': abs(mean_sparsity - self.target_sparsity)
            }
            
            # Compute prevalence metrics if target provided
            if self.taxon_prevalences is not None:
                actual_prevalence = compute_prevalence(compositions, self.threshold)
                target = self.taxon_prevalences.to(compositions.device)
                prevalence_mse = ((actual_prevalence - target) ** 2).mean().item()
                metrics['prevalence_mse'] = prevalence_mse
            
            return metrics
    
    def is_within_tolerance(
        self,
        compositions: Tensor,
        tolerance: float = 0.1
    ) -> bool:
        """Check if generated sparsity is within tolerance of target.
        
        Args:
            compositions: Generated compositions of shape (batch, num_taxa)
            tolerance: Acceptable deviation from target (default 0.1 = 10%)
        
        Returns:
            True if mean sparsity is within tolerance of target
        """
        metrics = self.compute_sparsity_metrics(compositions)
        return metrics['sparsity_deviation'] <= tolerance
    
    @classmethod
    def from_real_data(
        cls,
        real_data: Tensor,
        sparsity_weight: float = 1.0,
        prevalence_weight: float = 1.0,
        threshold: float = 1e-6,
        include_prevalence: bool = True
    ) -> 'SparsityLoss':
        """Create SparsityLoss from real microbiome data.
        
        Computes target sparsity and prevalence statistics from the
        provided real data.
        
        Args:
            real_data: Real microbiome compositions of shape (n_samples, num_taxa)
            sparsity_weight: Weight for overall sparsity loss
            prevalence_weight: Weight for per-taxon prevalence loss
            threshold: Abundance threshold for presence/absence
            include_prevalence: If True, include per-taxon prevalence matching
        
        Returns:
            Configured SparsityLoss instance
        """
        # Compute target sparsity
        target_sparsity, _ = compute_target_sparsity_from_data(real_data, threshold)
        
        # Compute target prevalence if requested
        taxon_prevalences = None
        if include_prevalence:
            taxon_prevalences = compute_target_prevalence_from_data(real_data, threshold)
        
        return cls(
            target_sparsity=target_sparsity,
            taxon_prevalences=taxon_prevalences,
            sparsity_weight=sparsity_weight,
            prevalence_weight=prevalence_weight,
            threshold=threshold
        )


class RareTaxaLoss(nn.Module):
    """Loss for ensuring rare taxa are appropriately absent.
    
    This loss specifically targets taxa with low prevalence in real data,
    ensuring they remain rare in generated samples.
    
    According to Requirements 1.4: When a taxon has prevalence below 1% 
    in real data, the generated samples SHALL have that taxon absent 
    in at least 95% of samples.
    
    Attributes:
        rare_taxa_mask: Boolean mask for rare taxa
        rare_threshold: Prevalence threshold for "rare" (default 0.01 = 1%)
        absence_target: Target absence rate for rare taxa (default 0.95 = 95%)
    """
    
    def __init__(
        self,
        taxon_prevalences: Tensor,
        rare_threshold: float = 0.01,
        absence_target: float = 0.95,
        penalty_weight: float = 1.0
    ):
        """Initialize rare taxa loss.
        
        Args:
            taxon_prevalences: Prevalence for each taxon, shape (num_taxa,)
            rare_threshold: Prevalence threshold below which a taxon is "rare"
            absence_target: Target absence rate for rare taxa
            penalty_weight: Weight for the penalty
        """
        super().__init__()
        
        self.rare_threshold = rare_threshold
        self.absence_target = absence_target
        self.penalty_weight = penalty_weight
        
        # Identify rare taxa
        if not isinstance(taxon_prevalences, Tensor):
            taxon_prevalences = torch.tensor(taxon_prevalences, dtype=torch.float32)
        
        rare_mask = taxon_prevalences < rare_threshold
        self.register_buffer('rare_taxa_mask', rare_mask)
        self.register_buffer('taxon_prevalences', taxon_prevalences)
        
        self.num_rare_taxa = rare_mask.sum().item()
    
    def forward(
        self,
        presence_probs: Tensor
    ) -> Tensor:
        """Compute rare taxa absence loss.
        
        Args:
            presence_probs: Presence probabilities of shape (batch, num_taxa)
        
        Returns:
            Rare taxa loss (scalar tensor)
        """
        if self.num_rare_taxa == 0:
            # Return zero that maintains gradient graph
            return presence_probs.sum() * 0.0
        
        # Get presence probs for rare taxa only
        rare_presence = presence_probs[:, self.rare_taxa_mask]  # (batch, num_rare)
        
        # Expected absence rate = 1 - mean(presence_prob)
        expected_absence = 1.0 - rare_presence.mean()
        
        # Penalize if absence rate is below target
        # Use ReLU to only penalize when below target
        shortfall = F.relu(self.absence_target - expected_absence)
        
        return self.penalty_weight * shortfall
    
    def compute_rare_taxa_metrics(
        self,
        compositions: Tensor,
        threshold: float = 1e-6
    ) -> dict:
        """Compute metrics for rare taxa in generated compositions.
        
        Args:
            compositions: Generated compositions of shape (batch, num_taxa)
            threshold: Abundance threshold for presence
        
        Returns:
            Dict with rare taxa metrics
        """
        with torch.no_grad():
            if self.num_rare_taxa == 0:
                return {
                    'num_rare_taxa': 0,
                    'mean_absence_rate': 1.0,
                    'meets_target': True
                }
            
            # Get compositions for rare taxa
            rare_compositions = compositions[:, self.rare_taxa_mask]
            
            # Compute absence rate
            is_absent = (rare_compositions < threshold).float()
            absence_rate = is_absent.mean().item()
            
            return {
                'num_rare_taxa': self.num_rare_taxa,
                'mean_absence_rate': absence_rate,
                'target_absence_rate': self.absence_target,
                'meets_target': absence_rate >= self.absence_target
            }
