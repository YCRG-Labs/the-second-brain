"""Centered Log-Ratio (CLR) transformation for compositional data.

This module implements the CLR transformation which maps compositional data
from the simplex to unconstrained Euclidean space, enabling standard machine
learning operations while respecting compositional constraints.

The CLR transformation is defined as:
    clr(x)_i = log(x_i / g(x))
where g(x) is the geometric mean of x.

References:
    Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from src.exceptions import InvalidCompositionError


class CLRTransform:
    """Centered log-ratio transformation for compositional data.
    
    This class provides forward and inverse CLR transformations with
    support for zero handling via multiplicative replacement.
    
    Attributes:
        num_taxa: Number of taxa (D) in the composition
        pseudocount: Pseudocount value for multiplicative replacement (ε = 0.5/D)
    """
    
    def __init__(self, num_taxa: int, pseudocount_factor: float = 0.5):
        """Initialize CLR transform.
        
        Args:
            num_taxa: Number of taxa (D)
            pseudocount_factor: Factor for multiplicative replacement (ε = factor/D)
        """
        if num_taxa < 1:
            raise ValueError("num_taxa must be at least 1")
        
        self.num_taxa = num_taxa
        self.pseudocount = pseudocount_factor / num_taxa
    
    def _apply_multiplicative_replacement(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply multiplicative replacement for zero entries.
        
        Replaces zeros with pseudocount ε and adjusts non-zero entries
        to maintain the simplex constraint (sum to 1).
        
        Args:
            x: Composition vector (batch, D) on simplex, may contain zeros
            
        Returns:
            replaced: Composition with zeros replaced (batch, D)
            mask: Binary mask where 1 indicates original zero (batch, D)
        """
        # Create binary mask for zeros (1 where zero, 0 where non-zero)
        zero_mask = (x == 0).float()
        
        # Count zeros per sample
        num_zeros = zero_mask.sum(dim=-1, keepdim=True)
        
        # Calculate adjustment factor for non-zero entries
        # Non-zero entries are scaled down to make room for pseudocounts
        delta = num_zeros * self.pseudocount
        
        # Apply replacement: zeros get pseudocount, non-zeros get scaled
        replaced = torch.where(
            x == 0,
            torch.full_like(x, self.pseudocount),
            x * (1 - delta) / (1 - delta + num_zeros * self.pseudocount - num_zeros * self.pseudocount)
        )
        
        # Simpler approach: replace zeros and renormalize
        replaced = torch.where(
            x == 0,
            torch.full_like(x, self.pseudocount),
            x
        )
        # Renormalize to ensure sum to 1
        replaced = replaced / replaced.sum(dim=-1, keepdim=True)
        
        return replaced, zero_mask
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform composition to CLR space.
        
        The CLR transformation maps a composition vector from the simplex
        to unconstrained Euclidean space. The result is centered (sums to zero).
        
        Args:
            x: Composition vector (batch, D) on simplex
               Must be non-negative and sum to 1
               
        Returns:
            y: CLR-transformed vector (batch, D)
               Note: Returns full D dimensions (not D-1) for simplicity
            mask: Binary zero mask (batch, D) where 1 indicates original zero
            
        Raises:
            InvalidCompositionError: If composition has negative values
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Validate non-negativity
        if torch.any(x < 0):
            raise InvalidCompositionError("Composition must be non-negative")
        
        # Handle zeros with multiplicative replacement
        x_replaced, mask = self._apply_multiplicative_replacement(x)
        
        # Compute geometric mean: exp(mean(log(x)))
        log_x = torch.log(x_replaced)
        log_geometric_mean = log_x.mean(dim=-1, keepdim=True)
        
        # CLR transformation: log(x_i / g(x)) = log(x_i) - log(g(x))
        y = log_x - log_geometric_mean
        
        if squeeze_output:
            y = y.squeeze(0)
            mask = mask.squeeze(0)
        
        return y, mask
    
    def inverse(self, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Transform CLR back to simplex.
        
        The inverse CLR uses softmax to ensure the result is a valid
        composition (non-negative, sums to 1).
        
        Args:
            y: CLR vector (batch, D)
            mask: Optional zero mask to apply after transformation
                  Where mask is 1, the output will be set to 0
                  
        Returns:
            x: Composition on simplex (batch, D)
               All components non-negative, sum equals 1
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply softmax to get valid composition
        # softmax(y)_i = exp(y_i) / sum(exp(y_j))
        x = torch.softmax(y, dim=-1)
        
        # Apply zero mask if provided
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            # Set masked positions to zero
            x = x * (1 - mask)
            # Renormalize to maintain simplex constraint
            x_sum = x.sum(dim=-1, keepdim=True)
            # Avoid division by zero
            x = torch.where(x_sum > 0, x / x_sum, x)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x
