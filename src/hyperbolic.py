"""Hyperbolic embedding module for phylogenetic trees.

This module implements hyperbolic geometry operations in the Poincaré ball model
and provides functionality to embed phylogenetic trees into hyperbolic space.

The Poincaré ball model represents hyperbolic space as the interior of a unit ball,
where distances grow exponentially near the boundary. This makes it naturally suited
for representing hierarchical structures like phylogenetic trees.

References:
    Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.types import PhylogeneticTree
from src.exceptions import EmptyTreeError


# Numerical stability constants
EPS = 1e-15
MAX_NORM = 1.0 - 1e-5


def _clamp_norm(x: Tensor, max_norm: float = MAX_NORM) -> Tensor:
    """Clamp tensor norm to stay within Poincaré ball.
    
    Args:
        x: Input tensor of shape (..., dim)
        max_norm: Maximum allowed norm (default: 1 - 1e-5)
        
    Returns:
        Tensor with norm clamped to max_norm
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    # Only scale down if norm exceeds max_norm
    scale = torch.clamp(max_norm / (norm + EPS), max=1.0)
    return x * scale


def poincare_distance(u: Tensor, v: Tensor, curvature: float = 1.0) -> Tensor:
    """Compute hyperbolic distance in Poincaré ball model.
    
    The distance formula in the Poincaré ball is:
        d(u, v) = (1/√c) * arcosh(1 + 2c * ||u - v||² / ((1 - c||u||²)(1 - c||v||²)))
    
    For curvature c = 1:
        d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))
    
    Args:
        u: First point(s) in Poincaré ball, shape (..., dim)
        v: Second point(s) in Poincaré ball, shape (..., dim)
        curvature: Negative curvature magnitude (default: 1.0)
        
    Returns:
        Hyperbolic distance(s), shape (...)
    """
    # Compute squared norms
    u_norm_sq = torch.sum(u * u, dim=-1)
    v_norm_sq = torch.sum(v * v, dim=-1)
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)
    
    # Compute the argument of arcosh
    # arcosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
    denom = (1 - curvature * u_norm_sq) * (1 - curvature * v_norm_sq)
    # Clamp denominator to avoid division by zero near boundary
    denom = torch.clamp(denom, min=EPS)
    
    x = 1 + 2 * curvature * diff_norm_sq / denom
    # Clamp x to be >= 1 for numerical stability (arcosh domain)
    x = torch.clamp(x, min=1.0 + EPS)
    
    # arcosh(x) = log(x + sqrt(x^2 - 1))
    distance = torch.acosh(x) / (curvature ** 0.5)
    
    return distance


def exponential_map(x: Tensor, v: Tensor, curvature: float = 1.0) -> Tensor:
    """Map tangent vector to point on Poincaré ball manifold.
    
    The exponential map at point x maps a tangent vector v to a point on the manifold.
    This is used in Riemannian optimization to update points while staying on the manifold.
    
    Formula:
        exp_x(v) = x ⊕ (tanh(√c * λ_x * ||v|| / 2) * v / (√c * ||v||))
    
    where λ_x = 2 / (1 - c||x||²) is the conformal factor and ⊕ is Möbius addition.
    
    Args:
        x: Base point(s) in Poincaré ball, shape (..., dim)
        v: Tangent vector(s) at x, shape (..., dim)
        curvature: Negative curvature magnitude (default: 1.0)
        
    Returns:
        Point(s) on manifold, shape (..., dim)
    """
    sqrt_c = curvature ** 0.5
    
    # Compute conformal factor λ_x = 2 / (1 - c||x||²)
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / torch.clamp(1 - curvature * x_norm_sq, min=EPS)
    
    # Compute ||v||
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=EPS)
    
    # Compute the scaling factor: tanh(√c * λ_x * ||v|| / 2) / (√c * ||v||)
    arg = sqrt_c * lambda_x * v_norm / 2
    # Clamp to avoid numerical issues with tanh
    arg = torch.clamp(arg, max=15.0)
    scale = torch.tanh(arg) / (sqrt_c * v_norm)
    
    # Direction of v
    v_normalized = v * scale
    
    # Möbius addition: x ⊕ y
    result = mobius_add(x, v_normalized, curvature)
    
    # Ensure result stays in ball
    return _clamp_norm(result)


def logarithmic_map(x: Tensor, y: Tensor, curvature: float = 1.0) -> Tensor:
    """Map point on manifold to tangent vector at base point.
    
    The logarithmic map is the inverse of the exponential map.
    It maps a point y on the manifold to a tangent vector at x.
    
    Formula:
        log_x(y) = (2 / (√c * λ_x)) * arctanh(√c * ||-x ⊕ y||) * (-x ⊕ y) / ||-x ⊕ y||
    
    Args:
        x: Base point(s) in Poincaré ball, shape (..., dim)
        y: Target point(s) in Poincaré ball, shape (..., dim)
        curvature: Negative curvature magnitude (default: 1.0)
        
    Returns:
        Tangent vector(s) at x, shape (..., dim)
    """
    sqrt_c = curvature ** 0.5
    
    # Compute conformal factor λ_x = 2 / (1 - c||x||²)
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2.0 / torch.clamp(1 - curvature * x_norm_sq, min=EPS)
    
    # Compute -x ⊕ y (Möbius addition of -x and y)
    neg_x = -x
    diff = mobius_add(neg_x, y, curvature)
    
    # Compute ||diff||
    diff_norm = torch.norm(diff, dim=-1, keepdim=True)
    diff_norm = torch.clamp(diff_norm, min=EPS)
    
    # Compute arctanh(√c * ||diff||)
    # arctanh(x) = 0.5 * log((1+x)/(1-x)) for |x| < 1
    arg = sqrt_c * diff_norm
    # Clamp to stay in valid domain of arctanh
    arg = torch.clamp(arg, max=1.0 - EPS)
    arctanh_val = torch.atanh(arg)
    
    # Scale factor: (2 / (√c * λ_x)) * arctanh(...) / ||diff||
    scale = (2.0 / (sqrt_c * lambda_x)) * arctanh_val / diff_norm
    
    return diff * scale


def mobius_add(x: Tensor, y: Tensor, curvature: float = 1.0) -> Tensor:
    """Möbius addition in the Poincaré ball.
    
    The Möbius addition is the group operation in hyperbolic space:
        x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
                (1 + 2c<x,y> + c²||x||²||y||²)
    
    Args:
        x: First point(s), shape (..., dim)
        y: Second point(s), shape (..., dim)
        curvature: Negative curvature magnitude (default: 1.0)
        
    Returns:
        Result of Möbius addition, shape (..., dim)
    """
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
    
    c = curvature
    
    # Numerator: (1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y
    num = (1 + 2 * c * xy_inner + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
    
    # Denominator: 1 + 2c<x,y> + c²||x||²||y||²
    denom = 1 + 2 * c * xy_inner + c * c * x_norm_sq * y_norm_sq
    denom = torch.clamp(denom, min=EPS)
    
    result = num / denom
    
    return _clamp_norm(result)


class HyperbolicEmbedder(nn.Module):
    """Embeds phylogenetic tree into Poincaré ball.
    
    This class learns hyperbolic embeddings for taxa in a phylogenetic tree,
    optimizing to preserve phylogenetic distances in hyperbolic space.
    
    The optimization uses Riemannian Adam with exponential map retraction
    to ensure embeddings remain within the Poincaré ball.
    
    Attributes:
        num_taxa: Number of OTUs/taxa to embed
        embedding_dim: Dimension of hyperbolic space
        curvature: Negative curvature of hyperbolic space
        embeddings: Learnable embedding parameters
    """
    
    def __init__(
        self, 
        num_taxa: int, 
        embedding_dim: int = 32, 
        curvature: float = 1.0
    ):
        """Initialize hyperbolic embedder.
        
        Args:
            num_taxa: Number of OTUs/taxa to embed
            embedding_dim: Dimension of hyperbolic space (default 32)
            curvature: Negative curvature of hyperbolic space (default 1.0)
            
        Raises:
            EmptyTreeError: If num_taxa < 1
        """
        super().__init__()
        
        if num_taxa < 1:
            raise EmptyTreeError("Number of taxa must be at least 1")
        
        self.num_taxa = num_taxa
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        
        # Initialize embeddings uniformly within a small ball
        # This ensures they start well within the Poincaré ball
        init_scale = 0.001
        embeddings = torch.randn(num_taxa, embedding_dim) * init_scale
        self.embeddings = nn.Parameter(embeddings)
    
    def get_embeddings(self) -> Tensor:
        """Return learned embeddings of shape (num_taxa, embedding_dim).
        
        Returns:
            Tensor of embeddings, clamped to stay within Poincaré ball
        """
        return _clamp_norm(self.embeddings)
    
    def _tree_loss(
        self, 
        tree: PhylogeneticTree, 
        margin: float = 0.1
    ) -> Tensor:
        """Compute tree loss for embedding optimization.
        
        The loss combines:
        1. Edge distance matching: hyperbolic distance should match patristic distance
        2. Non-adjacent separation: non-adjacent pairs should be separated by margin
        
        Args:
            tree: Phylogenetic tree structure
            margin: Margin for non-adjacent pair separation (default 0.1)
            
        Returns:
            Scalar loss tensor
        """
        embeddings = self.get_embeddings()
        
        # Handle edge case: single taxon or no edges
        if len(tree.edges) == 0:
            # Return a small regularization loss to keep embedding near origin
            return torch.sum(embeddings ** 2)
        
        # Edge distance matching loss
        edge_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        for (parent, child), target_dist in zip(tree.edges, tree.edge_lengths):
            u = embeddings[parent]
            v = embeddings[child]
            hyp_dist = poincare_distance(u.unsqueeze(0), v.unsqueeze(0), self.curvature)
            edge_loss = edge_loss + (hyp_dist - target_dist) ** 2
        
        edge_loss = edge_loss / len(tree.edges)
        
        # Non-adjacent separation loss with margin
        # Create set of adjacent pairs for quick lookup
        adjacent_pairs = set()
        for parent, child in tree.edges:
            adjacent_pairs.add((min(parent, child), max(parent, child)))
        
        separation_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        num_non_adjacent = 0
        
        for i in range(self.num_taxa):
            for j in range(i + 1, self.num_taxa):
                pair = (i, j)
                if pair not in adjacent_pairs:
                    u = embeddings[i]
                    v = embeddings[j]
                    hyp_dist = poincare_distance(
                        u.unsqueeze(0), v.unsqueeze(0), self.curvature
                    )
                    # Hinge loss: penalize if distance < margin
                    separation_loss = separation_loss + torch.relu(margin - hyp_dist)
                    num_non_adjacent += 1
        
        if num_non_adjacent > 0:
            separation_loss = separation_loss / num_non_adjacent
        
        return edge_loss + separation_loss
    
    def _riemannian_grad(self, euclidean_grad: Tensor) -> Tensor:
        """Convert Euclidean gradient to Riemannian gradient.
        
        The Riemannian gradient is the Euclidean gradient scaled by the
        inverse of the metric tensor: g^{-1} = ((1 - ||x||²) / 2)²
        
        Args:
            euclidean_grad: Gradient in Euclidean space
            
        Returns:
            Gradient in Riemannian (tangent) space
        """
        embeddings = self.get_embeddings()
        norm_sq = torch.sum(embeddings ** 2, dim=-1, keepdim=True)
        # Conformal factor squared (inverse metric)
        scale = ((1 - self.curvature * norm_sq) / 2) ** 2
        return euclidean_grad * scale

    
    def fit(
        self, 
        tree: PhylogeneticTree, 
        epochs: int = 1000,
        lr: float = 0.01,
        margin: float = 0.1,
        verbose: bool = False
    ) -> None:
        """Optimize embeddings to match phylogenetic distances.
        
        Uses Riemannian Adam optimizer with exponential map retraction
        to maintain points within the Poincaré ball.
        
        Args:
            tree: Phylogenetic tree to embed
            epochs: Number of optimization epochs (default 1000)
            lr: Learning rate (default 0.01)
            margin: Margin for non-adjacent separation (default 0.1)
            verbose: Whether to print progress (default False)
            
        Raises:
            EmptyTreeError: If tree has no taxa
        """
        if tree.num_taxa < 1:
            raise EmptyTreeError("Tree must have at least one taxon")
        
        if tree.num_taxa != self.num_taxa:
            raise ValueError(
                f"Tree has {tree.num_taxa} taxa but embedder has {self.num_taxa}"
            )
        
        # Use Adam optimizer with Riemannian gradient correction
        optimizer = torch.optim.Adam([self.embeddings], lr=lr)
        
        # Adam state for Riemannian optimization
        m = torch.zeros_like(self.embeddings)  # First moment
        v = torch.zeros_like(self.embeddings)  # Second moment
        beta1, beta2 = 0.9, 0.999
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss
            loss = self._tree_loss(tree, margin)
            
            # Backward pass
            loss.backward()
            
            # Get Euclidean gradient
            if self.embeddings.grad is not None:
                euclidean_grad = self.embeddings.grad.clone()
                
                # Convert to Riemannian gradient
                riemannian_grad = self._riemannian_grad(euclidean_grad)
                
                # Update Adam moments
                m = beta1 * m + (1 - beta1) * riemannian_grad
                v = beta2 * v + (1 - beta2) * (riemannian_grad ** 2)
                
                # Bias correction
                m_hat = m / (1 - beta1 ** (epoch + 1))
                v_hat = v / (1 - beta2 ** (epoch + 1))
                
                # Compute update direction (in tangent space)
                update = lr * m_hat / (torch.sqrt(v_hat) + EPS)
                
                # Apply exponential map to update embeddings
                with torch.no_grad():
                    current = self.get_embeddings()
                    new_embeddings = exponential_map(
                        current, -update, self.curvature
                    )
                    self.embeddings.data = new_embeddings
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Final clamp to ensure all embeddings are in ball
        with torch.no_grad():
            self.embeddings.data = _clamp_norm(self.embeddings.data)
    
    def forward(self, indices: Optional[Tensor] = None) -> Tensor:
        """Get embeddings for specified indices.
        
        Args:
            indices: Optional tensor of indices. If None, returns all embeddings.
            
        Returns:
            Embeddings tensor
        """
        embeddings = self.get_embeddings()
        if indices is not None:
            return embeddings[indices]
        return embeddings
