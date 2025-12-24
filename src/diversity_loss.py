"""Diversity Matching Losses for realistic microbiome generation.

This module implements differentiable diversity metrics and MMD-based
distribution matching losses for training microbiome generation models.

The losses ensure generated samples match the alpha and beta diversity
distributions of real microbiome data.

References:
    Gretton, A., et al. (2012). A Kernel Two-Sample Test.
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def differentiable_shannon_entropy(
    compositions: Tensor,
    eps: float = 1e-10
) -> Tensor:
    """Compute differentiable Shannon entropy (alpha diversity).
    
    Shannon entropy measures within-sample diversity:
    H = -sum(p_i * log(p_i)) where p_i are relative abundances.
    
    This implementation is fully differentiable for use in loss functions.
    
    Args:
        compositions: Relative abundance tensor of shape (batch, num_taxa)
                     Must be non-negative and sum to 1.
        eps: Small constant for numerical stability in log computation.
    
    Returns:
        Shannon entropy values of shape (batch,)
    """
    # Clamp to avoid log(0)
    compositions_safe = torch.clamp(compositions, min=eps)
    
    # Compute entropy: -sum(p * log(p))
    log_compositions = torch.log(compositions_safe)
    entropy = -torch.sum(compositions * log_compositions, dim=-1)
    
    return entropy


def differentiable_bray_curtis(
    comp1: Tensor,
    comp2: Tensor,
    eps: float = 1e-10
) -> Tensor:
    """Compute differentiable Bray-Curtis dissimilarity between compositions.
    
    Bray-Curtis dissimilarity is defined as:
    BC = sum(|x_i - y_i|) / sum(x_i + y_i)
    
    It ranges from 0 (identical) to 1 (completely dissimilar).
    
    Args:
        comp1: First composition tensor of shape (batch, num_taxa)
        comp2: Second composition tensor of shape (batch, num_taxa)
        eps: Small constant for numerical stability.
    
    Returns:
        Bray-Curtis dissimilarity values of shape (batch,)
    """
    numerator = torch.sum(torch.abs(comp1 - comp2), dim=-1)
    denominator = torch.sum(comp1 + comp2, dim=-1) + eps
    
    return numerator / denominator



def pairwise_bray_curtis(
    compositions: Tensor,
    eps: float = 1e-10
) -> Tensor:
    """Compute pairwise Bray-Curtis dissimilarity matrix.
    
    Args:
        compositions: Tensor of shape (batch, num_taxa)
        eps: Small constant for numerical stability.
    
    Returns:
        Dissimilarity matrix of shape (batch, batch)
    """
    batch_size = compositions.shape[0]
    
    # Expand for pairwise computation
    # comp1: (batch, 1, num_taxa), comp2: (1, batch, num_taxa)
    comp1 = compositions.unsqueeze(1)
    comp2 = compositions.unsqueeze(0)
    
    # Compute pairwise Bray-Curtis
    numerator = torch.sum(torch.abs(comp1 - comp2), dim=-1)
    denominator = torch.sum(comp1 + comp2, dim=-1) + eps
    
    return numerator / denominator


def differentiable_beta_diversity(
    compositions: Tensor,
    eps: float = 1e-10
) -> Tensor:
    """Compute beta diversity values from pairwise Bray-Curtis matrix.
    
    Returns the upper triangular values (excluding diagonal) as a 1D tensor.
    
    Args:
        compositions: Tensor of shape (batch, num_taxa)
        eps: Small constant for numerical stability.
    
    Returns:
        Beta diversity values of shape (batch * (batch - 1) / 2,)
    """
    bc_matrix = pairwise_bray_curtis(compositions, eps)
    
    # Get upper triangular indices (excluding diagonal)
    batch_size = compositions.shape[0]
    indices = torch.triu_indices(batch_size, batch_size, offset=1, device=compositions.device)
    
    return bc_matrix[indices[0], indices[1]]


class RBFKernel(nn.Module):
    """Radial Basis Function (Gaussian) kernel for MMD computation.
    
    k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    
    Attributes:
        bandwidth: Kernel bandwidth parameter (sigma)
    """
    
    def __init__(self, bandwidth: float = 1.0):
        """Initialize RBF kernel.
        
        Args:
            bandwidth: Kernel bandwidth (sigma). Larger values give smoother kernels.
        """
        super().__init__()
        self.bandwidth = bandwidth
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute kernel matrix between x and y.
        
        Args:
            x: First tensor of shape (n, d)
            y: Second tensor of shape (m, d)
        
        Returns:
            Kernel matrix of shape (n, m)
        """
        # Compute pairwise squared distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
        x_sqnorms = torch.sum(x ** 2, dim=-1, keepdim=True)  # (n, 1)
        y_sqnorms = torch.sum(y ** 2, dim=-1, keepdim=True)  # (m, 1)
        
        sq_distances = x_sqnorms + y_sqnorms.T - 2 * torch.mm(x, y.T)
        
        # Clamp to avoid numerical issues
        sq_distances = torch.clamp(sq_distances, min=0.0)
        
        # Compute RBF kernel
        return torch.exp(-sq_distances / (2 * self.bandwidth ** 2))


class MultiScaleRBFKernel(nn.Module):
    """Multi-scale RBF kernel for more robust MMD computation.
    
    Uses multiple bandwidth values and averages the kernels.
    This is more robust to the choice of bandwidth.
    
    Attributes:
        bandwidths: List of bandwidth values
    """
    
    def __init__(self, bandwidths: Optional[list] = None):
        """Initialize multi-scale RBF kernel.
        
        Args:
            bandwidths: List of bandwidth values. Default: [0.1, 0.5, 1.0, 2.0, 5.0]
        """
        super().__init__()
        if bandwidths is None:
            bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
        self.bandwidths = bandwidths
        self.kernels = nn.ModuleList([RBFKernel(bw) for bw in bandwidths])
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute averaged kernel matrix between x and y.
        
        Args:
            x: First tensor of shape (n, d)
            y: Second tensor of shape (m, d)
        
        Returns:
            Averaged kernel matrix of shape (n, m)
        """
        kernel_sum = torch.zeros(x.shape[0], y.shape[0], device=x.device, dtype=x.dtype)
        
        for kernel in self.kernels:
            kernel_sum = kernel_sum + kernel(x, y)
        
        return kernel_sum / len(self.kernels)



def compute_mmd(
    x: Tensor,
    y: Tensor,
    kernel: nn.Module
) -> Tensor:
    """Compute Maximum Mean Discrepancy between two distributions.
    
    MMD^2 = E[k(x, x')] + E[k(y, y')] - 2 * E[k(x, y)]
    
    where x, x' are samples from P and y, y' are samples from Q.
    
    Args:
        x: Samples from first distribution, shape (n, d)
        y: Samples from second distribution, shape (m, d)
        kernel: Kernel function module
    
    Returns:
        MMD value (scalar tensor)
    """
    n = x.shape[0]
    m = y.shape[0]
    
    # Compute kernel matrices
    k_xx = kernel(x, x)  # (n, n)
    k_yy = kernel(y, y)  # (m, m)
    k_xy = kernel(x, y)  # (n, m)
    
    # Compute unbiased MMD^2 estimate
    # For k_xx and k_yy, we exclude diagonal (self-comparisons)
    # E[k(x, x')] ≈ (sum(k_xx) - trace(k_xx)) / (n * (n - 1))
    
    if n > 1:
        sum_k_xx = (torch.sum(k_xx) - torch.trace(k_xx)) / (n * (n - 1))
    else:
        sum_k_xx = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    if m > 1:
        sum_k_yy = (torch.sum(k_yy) - torch.trace(k_yy)) / (m * (m - 1))
    else:
        sum_k_yy = torch.tensor(0.0, device=y.device, dtype=y.dtype)
    
    sum_k_xy = torch.sum(k_xy) / (n * m)
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd_squared = sum_k_xx + sum_k_yy - 2 * sum_k_xy
    
    # Return MMD (take sqrt, but clamp to avoid negative due to numerical errors)
    return torch.sqrt(torch.clamp(mmd_squared, min=0.0))


class DiversityMatchingLoss(nn.Module):
    """Loss for matching diversity distributions using MMD.
    
    This loss encourages generated microbiome samples to have similar
    alpha and beta diversity distributions as real data.
    
    The loss uses Maximum Mean Discrepancy (MMD) with RBF kernels to
    measure the distance between diversity distributions.
    
    Attributes:
        kernel: Kernel function for MMD computation
        alpha_weight: Weight for alpha diversity loss
        beta_weight: Weight for beta diversity loss
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        bandwidth: float = 1.0,
        multi_scale: bool = True,
        bandwidths: Optional[list] = None,
        alpha_weight: float = 1.0,
        beta_weight: float = 1.0
    ):
        """Initialize diversity matching loss.
        
        Args:
            kernel: Kernel type ('rbf' or 'multi_scale_rbf')
            bandwidth: Bandwidth for single-scale RBF kernel
            multi_scale: If True, use multi-scale RBF kernel
            bandwidths: Bandwidths for multi-scale kernel
            alpha_weight: Weight for alpha diversity loss component
            beta_weight: Weight for beta diversity loss component
        """
        super().__init__()
        
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        
        # Initialize kernel
        if multi_scale or kernel == 'multi_scale_rbf':
            self.kernel = MultiScaleRBFKernel(bandwidths)
        else:
            self.kernel = RBFKernel(bandwidth)
    
    def alpha_diversity_loss(
        self,
        generated: Tensor,
        real: Tensor
    ) -> Tensor:
        """Compute MMD loss for alpha diversity distributions.
        
        Computes Shannon entropy for both generated and real samples,
        then measures the MMD between these distributions.
        
        Args:
            generated: Generated compositions of shape (batch_gen, num_taxa)
            real: Real compositions of shape (batch_real, num_taxa)
        
        Returns:
            Alpha diversity MMD loss (scalar tensor)
        """
        # Compute alpha diversity (Shannon entropy) for both
        gen_alpha = differentiable_shannon_entropy(generated)  # (batch_gen,)
        real_alpha = differentiable_shannon_entropy(real)  # (batch_real,)
        
        # Reshape to (batch, 1) for MMD computation
        gen_alpha = gen_alpha.unsqueeze(-1)
        real_alpha = real_alpha.unsqueeze(-1)
        
        # Compute MMD between alpha diversity distributions
        return compute_mmd(gen_alpha, real_alpha, self.kernel)
    
    def beta_diversity_loss(
        self,
        generated: Tensor,
        real: Tensor
    ) -> Tensor:
        """Compute MMD loss for beta diversity distributions.
        
        Computes pairwise Bray-Curtis dissimilarities for both generated
        and real samples, then measures the MMD between these distributions.
        
        Args:
            generated: Generated compositions of shape (batch_gen, num_taxa)
            real: Real compositions of shape (batch_real, num_taxa)
        
        Returns:
            Beta diversity MMD loss (scalar tensor)
        """
        # Compute beta diversity (pairwise Bray-Curtis) for both
        gen_beta = differentiable_beta_diversity(generated)  # (n_pairs_gen,)
        real_beta = differentiable_beta_diversity(real)  # (n_pairs_real,)
        
        # Handle edge case where we don't have enough samples for pairs
        if gen_beta.numel() == 0 or real_beta.numel() == 0:
            return torch.tensor(0.0, device=generated.device, dtype=generated.dtype)
        
        # Reshape to (n_pairs, 1) for MMD computation
        gen_beta = gen_beta.unsqueeze(-1)
        real_beta = real_beta.unsqueeze(-1)
        
        # Compute MMD between beta diversity distributions
        return compute_mmd(gen_beta, real_beta, self.kernel)
    
    def forward(
        self,
        generated: Tensor,
        real: Tensor
    ) -> Tensor:
        """Compute combined diversity matching loss.
        
        Args:
            generated: Generated compositions of shape (batch_gen, num_taxa)
            real: Real compositions of shape (batch_real, num_taxa)
        
        Returns:
            Combined diversity loss (scalar tensor)
        """
        alpha_loss = self.alpha_diversity_loss(generated, real)
        beta_loss = self.beta_diversity_loss(generated, real)
        
        return self.alpha_weight * alpha_loss + self.beta_weight * beta_loss
    
    def forward_with_components(
        self,
        generated: Tensor,
        real: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute diversity loss with individual components.
        
        Args:
            generated: Generated compositions of shape (batch_gen, num_taxa)
            real: Real compositions of shape (batch_real, num_taxa)
        
        Returns:
            Tuple of (total_loss, alpha_loss, beta_loss)
        """
        alpha_loss = self.alpha_diversity_loss(generated, real)
        beta_loss = self.beta_diversity_loss(generated, real)
        total_loss = self.alpha_weight * alpha_loss + self.beta_weight * beta_loss
        
        return total_loss, alpha_loss, beta_loss
