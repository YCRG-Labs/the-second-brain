"""Differentiable rasterization module for microbiome compositions.

This module converts microbiome compositions into spatial images using
hyperbolic embeddings and kernel density estimation. The rasterization
is fully differentiable to allow gradient flow back to embeddings.

The pipeline:
1. Stereographic projection: Poincaré ball → 2D plane
2. Kernel density estimation: Point abundances → Spatial density
3. Channel normalization: Raw density → [0, 1] range

References:
    Cannon, J.W., et al. (1997). Hyperbolic Geometry.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.exceptions import DimensionMismatchError


# Numerical stability constants
EPS = 1e-15
MAX_NORM = 1.0 - 1e-5


def stereographic_project(embeddings: Tensor, curvature: float = 1.0) -> Tensor:
    """Project from Poincaré ball to 2D plane via stereographic projection.
    
    The stereographic projection maps points from the Poincaré ball model
    of hyperbolic space to the Euclidean plane. For a point x in the ball,
    we project from the "south pole" (0, 0, ..., -1) through x onto the
    plane at z=0.
    
    For the 2D case (embedding_dim >= 2), we use the first two coordinates
    and apply the standard stereographic projection formula:
        (x, y) → (2x/(1-r²), 2y/(1-r²))
    where r² = x² + y² (using only first 2 dims for projection).
    
    This ensures finite outputs for all points strictly inside the ball.
    
    Args:
        embeddings: Points in Poincaré ball, shape (..., dim) where dim >= 2
                   All points must have norm < 1
        curvature: Negative curvature magnitude (default: 1.0)
        
    Returns:
        2D coordinates on the plane, shape (..., 2)
        
    Note:
        For points very close to the boundary (norm → 1), the projection
        maps to infinity. We clamp the norm to ensure finite outputs.
    """
    # Extract first two dimensions for 2D projection
    x = embeddings[..., 0]
    y = embeddings[..., 1] if embeddings.shape[-1] > 1 else torch.zeros_like(x)
    
    # Compute squared norm of the 2D projection
    r_sq = x * x + y * y
    
    # Clamp to ensure we stay strictly inside the ball
    # This prevents division by zero and infinite outputs
    r_sq = torch.clamp(r_sq, max=MAX_NORM ** 2)
    
    # Stereographic projection formula: scale = 2 / (1 - r²)
    # For curvature c, the formula becomes: scale = 2 / (1 - c*r²)
    denom = 1.0 - curvature * r_sq
    denom = torch.clamp(denom, min=EPS)  # Ensure positive denominator
    scale = 2.0 / denom
    
    # Apply projection
    proj_x = x * scale
    proj_y = y * scale
    
    # Stack into 2D coordinates
    result = torch.stack([proj_x, proj_y], dim=-1)
    
    return result



class DifferentiableRasterizer(nn.Module):
    """Converts microbiome compositions to spatial images.
    
    This class implements a differentiable rasterization pipeline that:
    1. Projects hyperbolic embeddings to 2D via stereographic projection
    2. Applies adaptive kernel density estimation with phylogenetic depth-based bandwidth
    3. Produces multi-channel images with functional annotations
    4. Normalizes each channel to [0, 1]
    
    The rasterization is fully differentiable, allowing gradients to flow
    back to both embeddings and abundances during training.
    
    Attributes:
        embeddings: Hyperbolic embeddings (num_taxa, embed_dim)
        image_size: Output image resolution (H=W)
        num_channels: Number of functional channels
        functional_annotations: Per-taxon functional features
        curvature: Hyperbolic space curvature
    """
    
    def __init__(
        self,
        embeddings: Tensor,
        image_size: int = 256,
        num_channels: int = 16,
        functional_annotations: Optional[Tensor] = None,
        curvature: float = 1.0,
        base_bandwidth: float = 0.1,
        phylogenetic_depths: Optional[Tensor] = None
    ):
        """Initialize differentiable rasterizer.
        
        Args:
            embeddings: Hyperbolic embeddings (num_taxa, embed_dim)
            image_size: Output image resolution (default 256)
            num_channels: Number of functional channels (default 16)
            functional_annotations: Per-taxon functional features (num_taxa, num_channels)
                                   If None, uses one-hot encoding up to num_channels
            curvature: Hyperbolic space curvature (default 1.0)
            base_bandwidth: Base bandwidth for KDE (default 0.1)
            phylogenetic_depths: Per-taxon phylogenetic depths for adaptive bandwidth
                                If None, uses uniform bandwidth
        """
        super().__init__()
        
        if embeddings.dim() != 2:
            raise DimensionMismatchError(
                f"Embeddings must be 2D (num_taxa, embed_dim), got shape {embeddings.shape}"
            )
        
        self.num_taxa = embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        self.image_size = image_size
        self.num_channels = num_channels
        self.curvature = curvature
        self.base_bandwidth = base_bandwidth
        
        # Register embeddings as buffer (not trainable by this module)
        self.register_buffer('embeddings', embeddings)
        
        # Set up functional annotations
        if functional_annotations is not None:
            if functional_annotations.shape[0] != self.num_taxa:
                raise DimensionMismatchError(
                    f"Functional annotations must have {self.num_taxa} rows, "
                    f"got {functional_annotations.shape[0]}"
                )
            if functional_annotations.shape[1] != num_channels:
                raise DimensionMismatchError(
                    f"Functional annotations must have {num_channels} columns, "
                    f"got {functional_annotations.shape[1]}"
                )
            self.register_buffer('functional_annotations', functional_annotations)
        else:
            # Default: distribute taxa across channels
            # Each taxon contributes to channel (taxon_idx % num_channels)
            annotations = torch.zeros(self.num_taxa, num_channels)
            for i in range(self.num_taxa):
                annotations[i, i % num_channels] = 1.0
            self.register_buffer('functional_annotations', annotations)
        
        # Set up phylogenetic depths for adaptive bandwidth
        if phylogenetic_depths is not None:
            if phylogenetic_depths.shape[0] != self.num_taxa:
                raise DimensionMismatchError(
                    f"Phylogenetic depths must have {self.num_taxa} elements, "
                    f"got {phylogenetic_depths.shape[0]}"
                )
            self.register_buffer('phylogenetic_depths', phylogenetic_depths)
        else:
            # Default: uniform depths
            self.register_buffer(
                'phylogenetic_depths', 
                torch.ones(self.num_taxa)
            )
        
        # Pre-compute pixel grid coordinates
        # Grid spans [-1, 1] in both dimensions (will be scaled based on projections)
        coords = torch.linspace(-1, 1, image_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        # Shape: (image_size, image_size, 2)
        self.register_buffer('pixel_grid', torch.stack([grid_x, grid_y], dim=-1))
    
    def _compute_adaptive_bandwidth(self) -> Tensor:
        """Compute per-taxon bandwidth based on phylogenetic depth.
        
        Taxa at greater phylogenetic depth (more derived) get smaller bandwidth,
        while taxa at shallower depth (more basal) get larger bandwidth.
        
        Returns:
            Per-taxon bandwidth values, shape (num_taxa,)
        """
        # Normalize depths to [0, 1]
        depths = self.phylogenetic_depths
        min_depth = depths.min()
        max_depth = depths.max()
        
        if max_depth > min_depth:
            normalized_depths = (depths - min_depth) / (max_depth - min_depth)
        else:
            normalized_depths = torch.ones_like(depths) * 0.5
        
        # Bandwidth inversely proportional to depth
        # Deeper taxa get smaller bandwidth (more localized)
        # Range: [0.5 * base, 1.5 * base]
        bandwidth = self.base_bandwidth * (1.5 - normalized_depths)
        
        return bandwidth
    
    def _scale_projections(self, projections: Tensor) -> Tensor:
        """Scale stereographic projections to fit within image grid.
        
        Args:
            projections: 2D coordinates from stereographic projection, shape (..., 2)
            
        Returns:
            Scaled coordinates in [-1, 1] range, shape (..., 2)
        """
        # Find the range of projections
        proj_min = projections.min()
        proj_max = projections.max()
        proj_range = proj_max - proj_min
        
        if proj_range > EPS:
            # Scale to [-0.9, 0.9] to leave margin at edges
            scaled = 1.8 * (projections - proj_min) / proj_range - 0.9
        else:
            # All points at same location, center them
            scaled = torch.zeros_like(projections)
        
        return scaled
    
    def rasterize(self, abundances: Tensor) -> Tensor:
        """Convert abundance vector to spatial image.
        
        Uses adaptive kernel density estimation to spread each taxon's
        abundance across the image based on its projected location and
        phylogenetic depth-based bandwidth.
        
        Args:
            abundances: Composition vector (batch, num_taxa) or (num_taxa,)
                       Must be non-negative
                       
        Returns:
            Spatial image (batch, channels, H, W) with values in [0, 1]
            
        Raises:
            DimensionMismatchError: If abundances dimension doesn't match num_taxa
        """
        # Handle 1D input
        if abundances.dim() == 1:
            abundances = abundances.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = abundances.shape[0]
        
        if abundances.shape[1] != self.num_taxa:
            raise DimensionMismatchError(
                f"Abundances must have {self.num_taxa} taxa, got {abundances.shape[1]}"
            )
        
        # Project embeddings to 2D
        projections = stereographic_project(self.embeddings, self.curvature)
        # Scale to fit image grid
        projections = self._scale_projections(projections)  # (num_taxa, 2)
        
        # Compute adaptive bandwidths
        bandwidths = self._compute_adaptive_bandwidth()  # (num_taxa,)
        
        # Pixel grid: (H, W, 2)
        grid = self.pixel_grid  # (H, W, 2)
        
        # Vectorized KDE computation
        # Expand grid for broadcasting: (1, H, W, 2)
        grid_expanded = grid.unsqueeze(0)
        # Expand projections: (num_taxa, 1, 1, 2)
        proj_expanded = projections.unsqueeze(1).unsqueeze(1)
        
        # Compute squared distances: (num_taxa, H, W)
        diff = grid_expanded - proj_expanded  # (num_taxa, H, W, 2)
        dist_sq = (diff * diff).sum(dim=-1)  # (num_taxa, H, W)
        
        # Compute Gaussian kernels with adaptive bandwidth
        # bandwidths: (num_taxa,) -> (num_taxa, 1, 1)
        bw_expanded = bandwidths.unsqueeze(-1).unsqueeze(-1)
        kernels = torch.exp(-dist_sq / (2 * bw_expanded * bw_expanded + EPS))  # (num_taxa, H, W)
        
        # Weight kernels by abundances
        # abundances: (batch, num_taxa) -> (batch, num_taxa, 1, 1)
        abundances_expanded = abundances.unsqueeze(-1).unsqueeze(-1)
        # kernels: (num_taxa, H, W) -> (1, num_taxa, H, W)
        kernels_expanded = kernels.unsqueeze(0)
        
        # Weighted kernels: (batch, num_taxa, H, W)
        weighted_kernels = abundances_expanded * kernels_expanded
        
        # Apply functional annotations to map taxa to channels
        # functional_annotations: (num_taxa, channels)
        # weighted_kernels: (batch, num_taxa, H, W)
        # Result: (batch, channels, H, W)
        
        # Reshape for matrix multiplication
        # weighted_kernels: (batch, num_taxa, H*W)
        wk_flat = weighted_kernels.view(batch_size, self.num_taxa, -1)
        # functional_annotations.T: (channels, num_taxa)
        func_t = self.functional_annotations.t()
        
        # Matrix multiply: (batch, channels, H*W)
        image_flat = torch.matmul(func_t.unsqueeze(0), wk_flat)
        
        # Reshape back: (batch, channels, H, W)
        image = image_flat.view(batch_size, self.num_channels, self.image_size, self.image_size)
        
        # Normalize each channel independently to [0, 1]
        image = self._normalize_channels(image)
        
        if squeeze_output:
            image = image.squeeze(0)
        
        return image
    
    def _normalize_channels(self, image: Tensor) -> Tensor:
        """Normalize each channel independently to [0, 1].
        
        Args:
            image: Raw image tensor (batch, channels, H, W)
            
        Returns:
            Normalized image with each channel in [0, 1]
        """
        # Compute min and max per channel (across H, W dimensions)
        # image: (batch, channels, H, W)
        # Flatten spatial dimensions for min/max computation
        image_flat = image.view(image.shape[0], image.shape[1], -1)  # (batch, channels, H*W)
        
        c_min = image_flat.min(dim=-1, keepdim=True)[0]  # (batch, channels, 1)
        c_max = image_flat.max(dim=-1, keepdim=True)[0]  # (batch, channels, 1)
        c_range = c_max - c_min
        
        # Normalize, handling constant channels
        # Where range is ~0, output 0
        normalized_flat = torch.where(
            c_range > EPS,
            (image_flat - c_min) / (c_range + EPS),
            torch.zeros_like(image_flat)
        )
        
        # Reshape back
        normalized = normalized_flat.view_as(image)
        
        return normalized
    
    def unrasterize(self, image: Tensor) -> Tensor:
        """Inverse operation: image back to abundance vector.
        
        This is an approximate inverse that estimates abundances by
        sampling the image at each taxon's projected location and
        weighting by functional annotations.
        
        Args:
            image: Spatial image (batch, channels, H, W) or (channels, H, W)
                  Values should be in [0, 1]
                  
        Returns:
            Estimated abundances (batch, num_taxa) on simplex
            
        Note:
            This is not an exact inverse due to the lossy nature of
            rasterization. The output is normalized to sum to 1.
        """
        # Handle 3D input
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = image.shape[0]
        
        # Project embeddings to 2D and scale
        projections = stereographic_project(self.embeddings, self.curvature)
        projections = self._scale_projections(projections)  # (num_taxa, 2)
        
        # Convert projections to pixel coordinates
        # projections are in [-1, 1], convert to [0, image_size-1]
        pixel_coords = ((projections + 1) / 2 * (self.image_size - 1)).long()
        pixel_coords = torch.clamp(pixel_coords, 0, self.image_size - 1)
        
        # Initialize abundances
        abundances = torch.zeros(
            batch_size, self.num_taxa,
            device=image.device, dtype=image.dtype
        )
        
        # For each taxon, sample image at its location
        for t in range(self.num_taxa):
            px, py = pixel_coords[t]
            func = self.functional_annotations[t]  # (channels,)
            
            for b in range(batch_size):
                # Sample all channels at this location
                pixel_values = image[b, :, py, px]  # (channels,)
                # Weight by functional annotations
                abundance = (pixel_values * func).sum()
                abundances[b, t] = abundance
        
        # Normalize to simplex (sum to 1)
        abundances = F.relu(abundances)  # Ensure non-negative
        abundances_sum = abundances.sum(dim=-1, keepdim=True)
        abundances = torch.where(
            abundances_sum > EPS,
            abundances / abundances_sum,
            torch.ones_like(abundances) / self.num_taxa  # Uniform if all zero
        )
        
        if squeeze_output:
            abundances = abundances.squeeze(0)
        
        return abundances
    
    def forward(self, abundances: Tensor) -> Tensor:
        """Forward pass: rasterize abundances to image.
        
        Args:
            abundances: Composition vector (batch, num_taxa)
            
        Returns:
            Spatial image (batch, channels, H, W)
        """
        return self.rasterize(abundances)
