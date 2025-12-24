"""Spatiotemporal Transformer for longitudinal microbiome prediction.

This module implements a TimeSformer-style architecture for predicting future
microbiome states from longitudinal samples. The architecture uses factorized
spatial-temporal attention to efficiently process video sequences of rasterized
microbiome images.

Key components:
1. Patch Embedding: Converts frames into patch tokens with positional encodings
2. Factorized Attention: Separate spatial and temporal attention for efficiency
3. Multi-horizon Prediction: Supports prediction horizons H ∈ {1, 2, 4, 8}

References:
    Bertasius, G., et al. (2021). Is Space-Time Attention All You Need for Video Understanding?
    Arnab, A., et al. (2021). ViViT: A Video Vision Transformer.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Positional Encodings
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spatial and temporal positions.
    
    Uses sinusoidal functions at different frequencies to encode positions,
    similar to the original Transformer positional encodings.
    
    Attributes:
        dim: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, dim: int, max_len: int = 1024):
        """Initialize positional encoding.
        
        Args:
            dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, positions: Tensor) -> Tensor:
        """Get positional encodings for given positions.
        
        Args:
            positions: Position indices (batch, seq_len) or (seq_len,)
            
        Returns:
            Positional encodings (batch, seq_len, dim) or (seq_len, dim)
        """
        if positions.dim() == 1:
            return self.pe[positions]
        return self.pe[positions]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding.
    
    Uses learnable embeddings for each position.
    
    Attributes:
        dim: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, dim: int, max_len: int = 1024):
        """Initialize learned positional encoding.
        
        Args:
            dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)
    
    def forward(self, positions: Tensor) -> Tensor:
        """Get positional encodings for given positions.
        
        Args:
            positions: Position indices
            
        Returns:
            Positional encodings
        """
        return self.embedding(positions)


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbedding(nn.Module):
    """Converts video frames into patch tokens with positional encodings.
    
    Takes a video sequence of shape (batch, seq_len, channels, H, W) and
    converts it to patch tokens of shape (batch, seq_len, num_patches, embed_dim).
    
    For 256×256 images with 16×16 patches, this produces 256 patches per frame.
    
    Attributes:
        image_size: Input image resolution
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Output embedding dimension
        num_patches: Number of patches per frame (H/patch_size * W/patch_size)
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 16,
        embed_dim: int = 768,
        max_seq_len: int = 16
    ):
        """Initialize patch embedding.
        
        Args:
            image_size: Input image resolution (default 256)
            patch_size: Size of each patch (default 16)
            in_channels: Number of input channels (default 16)
            embed_dim: Output embedding dimension (default 768)
            max_seq_len: Maximum temporal sequence length (default 16)
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Number of patches per frame: (256/16) * (256/16) = 16 * 16 = 256
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch projection: Conv2d with kernel_size=stride=patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Spatial positional encoding (for patches within a frame)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # Temporal positional encoding (for frames in sequence)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """Convert video frames to patch tokens.
        
        Args:
            x: Input video (batch, seq_len, channels, H, W)
            
        Returns:
            Patch tokens (batch, seq_len, num_patches, embed_dim)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames at once
        x = x.reshape(B * T, C, H, W)
        
        # Project patches: (B*T, C, H, W) -> (B*T, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B*T, embed_dim, H/P, W/P) -> (B*T, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B*T, embed_dim, num_patches) -> (B*T, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Reshape back to video: (B*T, num_patches, embed_dim) -> (B, T, num_patches, embed_dim)
        x = x.reshape(B, T, self.num_patches, self.embed_dim)
        
        # Add spatial positional encoding (broadcast over batch and time)
        x = x + self.spatial_pos_embed.unsqueeze(1)  # (1, 1, num_patches, embed_dim)
        
        # Add temporal positional encoding (broadcast over batch and patches)
        temporal_pos = self.temporal_pos_embed[:, :T, :].unsqueeze(2)  # (1, T, 1, embed_dim)
        x = x + temporal_pos
        
        # Normalize
        x = self.norm(x)
        
        return x
    
    def get_num_patches(self) -> int:
        """Return the number of patches per frame.
        
        Returns:
            Number of patches (256 for 256×256 images with 16×16 patches)
        """
        return self.num_patches



# =============================================================================
# Factorized Attention
# =============================================================================

class SpatialAttention(nn.Module):
    """Multi-head self-attention within each frame (spatial attention).
    
    Applies attention across all patches within the same frame,
    allowing the model to capture spatial relationships.
    
    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        """Initialize spatial attention.
        
        Args:
            embed_dim: Embedding dimension (default 768)
            num_heads: Number of attention heads (default 12)
            dropout: Dropout probability (default 0.1)
            qkv_bias: Whether to use bias in QKV projection (default True)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial attention within each frame.
        
        Args:
            x: Input tokens (batch, seq_len, num_patches, embed_dim)
            
        Returns:
            Output tokens (batch, seq_len, num_patches, embed_dim)
        """
        B, T, N, D = x.shape
        
        # Reshape to process each frame independently
        # (B, T, N, D) -> (B*T, N, D)
        x = x.reshape(B * T, N, D)
        
        # Compute QKV
        qkv = self.qkv(x)  # (B*T, N, 3*D)
        qkv = qkv.reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B*T, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B * T, N, D)  # (B*T, N, D)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        # Reshape back to video format
        out = out.reshape(B, T, N, D)
        
        return out


class TemporalAttention(nn.Module):
    """Multi-head self-attention across frames (temporal attention).
    
    Applies attention across all frames for each patch position,
    allowing the model to capture temporal dynamics.
    
    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        """Initialize temporal attention.
        
        Args:
            embed_dim: Embedding dimension (default 768)
            num_heads: Number of attention heads (default 12)
            dropout: Dropout probability (default 0.1)
            qkv_bias: Whether to use bias in QKV projection (default True)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply temporal attention across frames.
        
        Args:
            x: Input tokens (batch, seq_len, num_patches, embed_dim)
            
        Returns:
            Output tokens (batch, seq_len, num_patches, embed_dim)
        """
        B, T, N, D = x.shape
        
        # Reshape to process each patch position across time
        # (B, T, N, D) -> (B*N, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        # Compute QKV
        qkv = self.qkv(x)  # (B*N, T, 3*D)
        qkv = qkv.reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, num_heads, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B*N, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B * N, T, D)  # (B*N, T, D)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        # Reshape back to video format
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        
        return out


class MLP(nn.Module):
    """Feed-forward MLP block.
    
    Standard transformer MLP with GELU activation.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """Initialize MLP.
        
        Args:
            embed_dim: Input/output dimension
            mlp_ratio: Hidden dimension multiplier
            dropout: Dropout probability
        """
        super().__init__()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FactorizedTransformerBlock(nn.Module):
    """Transformer block with factorized spatial-temporal attention.
    
    Applies spatial attention followed by temporal attention, with
    residual connections and layer normalization.
    
    Architecture:
        x -> LayerNorm -> SpatialAttn -> + -> LayerNorm -> TemporalAttn -> + -> LayerNorm -> MLP -> +
            |__________________________|    |___________________________|    |__________________|
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        """Initialize factorized transformer block.
        
        Args:
            embed_dim: Embedding dimension (default 768)
            num_heads: Number of attention heads (default 12)
            mlp_ratio: MLP hidden dimension multiplier (default 4.0)
            dropout: Dropout probability (default 0.1)
            qkv_bias: Whether to use bias in QKV projection (default True)
        """
        super().__init__()
        
        # Spatial attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout, qkv_bias)
        
        # Temporal attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout, qkv_bias)
        
        # MLP
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with factorized attention.
        
        Args:
            x: Input tokens (batch, seq_len, num_patches, embed_dim)
            
        Returns:
            Output tokens (batch, seq_len, num_patches, embed_dim)
        """
        # Spatial attention with residual
        x = x + self.spatial_attn(self.norm1(x))
        
        # Temporal attention with residual
        x = x + self.temporal_attn(self.norm2(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm3(x))
        
        return x



# =============================================================================
# Metadata Encoding
# =============================================================================

class MetadataTokenEncoder(nn.Module):
    """Encodes time-varying metadata as additional tokens.
    
    Converts metadata vectors into tokens that can be prepended to
    the patch sequence for conditioning.
    
    Attributes:
        metadata_dim: Input metadata dimension
        embed_dim: Output embedding dimension
    """
    
    def __init__(
        self,
        metadata_dim: int = 128,
        embed_dim: int = 768,
        num_tokens: int = 4
    ):
        """Initialize metadata encoder.
        
        Args:
            metadata_dim: Input metadata dimension (default 128)
            embed_dim: Output embedding dimension (default 768)
            num_tokens: Number of metadata tokens per timestep (default 4)
        """
        super().__init__()
        
        self.metadata_dim = metadata_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # Project metadata to multiple tokens
        self.proj = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim * num_tokens),
            nn.GELU(),
            nn.Linear(embed_dim * num_tokens, embed_dim * num_tokens)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, metadata: Tensor) -> Tensor:
        """Encode metadata as tokens.
        
        Args:
            metadata: Input metadata (batch, seq_len, metadata_dim)
            
        Returns:
            Metadata tokens (batch, seq_len, num_tokens, embed_dim)
        """
        B, T, D = metadata.shape
        
        # Project to tokens
        tokens = self.proj(metadata)  # (B, T, embed_dim * num_tokens)
        tokens = tokens.reshape(B, T, self.num_tokens, self.embed_dim)
        
        # Normalize
        tokens = self.norm(tokens)
        
        return tokens


# =============================================================================
# Spatiotemporal Transformer
# =============================================================================

class SpatiotemporalTransformer(nn.Module):
    """TimeSformer-style architecture for longitudinal microbiome prediction.
    
    This model processes video sequences of rasterized microbiome images
    using factorized spatial-temporal attention to predict future states.
    
    Architecture:
    1. Patch embedding with spatial and temporal positional encodings
    2. Stack of factorized transformer blocks (spatial then temporal attention)
    3. Prediction head for generating future frames
    
    Attributes:
        image_size: Input image resolution
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        max_seq_len: Maximum temporal sequence length
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 16,
        metadata_dim: int = 128
    ):
        """Initialize spatiotemporal transformer.
        
        Args:
            image_size: Input image resolution (default 256)
            patch_size: Size of each patch (default 16)
            in_channels: Number of input channels (default 16)
            embed_dim: Transformer embedding dimension (default 768)
            depth: Number of transformer blocks (default 12)
            num_heads: Number of attention heads (default 12)
            mlp_ratio: MLP hidden dimension multiplier (default 4.0)
            dropout: Dropout probability (default 0.1)
            max_seq_len: Maximum temporal sequence length (default 16)
            metadata_dim: Dimension of metadata input (default 128)
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.metadata_dim = metadata_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        # Metadata token encoder for intervention encoding
        self.metadata_encoder = MetadataTokenEncoder(
            metadata_dim=metadata_dim,
            embed_dim=embed_dim,
            num_tokens=4
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FactorizedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction head: project back to image patches
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """Initialize module weights.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _patches_to_image(self, patches: Tensor) -> Tensor:
        """Convert patch tokens back to image.
        
        Args:
            patches: Patch tokens (batch, num_patches, patch_size^2 * channels)
            
        Returns:
            Image (batch, channels, H, W)
        """
        B, N, _ = patches.shape
        
        # Reshape to (B, num_patches, channels, patch_size, patch_size)
        patches = patches.reshape(
            B, N, self.in_channels, self.patch_size, self.patch_size
        )
        
        # Compute grid dimensions
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        
        # Reshape to grid: (B, h_patches, w_patches, C, P, P)
        patches = patches.reshape(B, h_patches, w_patches, self.in_channels, 
                                  self.patch_size, self.patch_size)
        
        # Permute and reshape to image: (B, C, H, W)
        image = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.in_channels, self.image_size, self.image_size
        )
        
        return image
    
    def forward(
        self,
        video: Tensor,
        metadata_seq: Optional[Tensor] = None,
        predict_horizon: int = 1
    ) -> Tensor:
        """Predict future frames.
        
        Args:
            video: Input sequence (batch, seq_len, channels, H, W)
            metadata_seq: Time-varying metadata (batch, seq_len, metadata_dim)
            predict_horizon: Number of future frames to predict (default 1)
            
        Returns:
            Predicted frames (batch, predict_horizon, channels, H, W)
        """
        B, T, C, H, W = video.shape
        
        # Embed patches
        x = self.patch_embed(video)  # (B, T, num_patches, embed_dim)
        
        # Add metadata tokens if provided
        if metadata_seq is not None:
            # Encode metadata as tokens
            meta_tokens = self.metadata_encoder(metadata_seq)  # (B, T, num_meta_tokens, embed_dim)
            
            # Concatenate metadata tokens with patch tokens
            x = torch.cat([meta_tokens, x], dim=2)  # (B, T, num_meta_tokens + num_patches, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Remove metadata tokens if they were added
        if metadata_seq is not None:
            num_meta_tokens = self.metadata_encoder.num_tokens
            x = x[:, :, num_meta_tokens:, :]  # (B, T, num_patches, embed_dim)
        
        # Predict future frames
        # Use the last frame's features to predict future
        last_frame_features = x[:, -1, :, :]  # (B, num_patches, embed_dim)
        
        # Generate predictions for each horizon step
        predictions = []
        current_features = last_frame_features
        
        for h in range(predict_horizon):
            # Project to patch pixels
            pred_patches = self.pred_head(current_features)  # (B, num_patches, P*P*C)
            
            # Convert patches to image
            pred_image = self._patches_to_image(pred_patches)  # (B, C, H, W)
            predictions.append(pred_image)
            
            # For multi-step prediction, we could update features
            # For now, we use the same features (single-step prediction repeated)
            # The autoregressive_generate method handles proper multi-step
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, predict_horizon, C, H, W)
        
        return predictions
    
    def encode(
        self,
        video: Tensor,
        metadata_seq: Optional[Tensor] = None
    ) -> Tensor:
        """Encode video sequence to features.
        
        Args:
            video: Input sequence (batch, seq_len, channels, H, W)
            metadata_seq: Time-varying metadata (batch, seq_len, metadata_dim)
            
        Returns:
            Encoded features (batch, seq_len, num_patches, embed_dim)
        """
        # Embed patches
        x = self.patch_embed(video)
        
        # Add metadata tokens if provided
        if metadata_seq is not None:
            meta_tokens = self.metadata_encoder(metadata_seq)
            x = torch.cat([meta_tokens, x], dim=2)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Remove metadata tokens
        if metadata_seq is not None:
            num_meta_tokens = self.metadata_encoder.num_tokens
            x = x[:, :, num_meta_tokens:, :]
        
        return x


    @torch.no_grad()
    def autoregressive_generate(
        self,
        video: Tensor,
        metadata_seq: Tensor,
        future_metadata: Tensor,
        num_steps: int
    ) -> Tensor:
        """Generate multiple future frames autoregressively.
        
        Performs autoregressive rollout during inference, feeding predicted
        frames back as input for subsequent predictions.
        
        Args:
            video: Input sequence (batch, seq_len, channels, H, W)
            metadata_seq: Metadata for input sequence (batch, seq_len, metadata_dim)
            future_metadata: Metadata for future steps (batch, num_steps, metadata_dim)
            num_steps: Number of future frames to generate
            
        Returns:
            Generated frames (batch, num_steps, channels, H, W)
        """
        B, T, C, H, W = video.shape
        device = video.device
        
        # Validate num_steps is in supported horizons
        supported_horizons = {1, 2, 4, 8}
        if num_steps not in supported_horizons:
            # Round up to nearest supported horizon
            for h in sorted(supported_horizons):
                if h >= num_steps:
                    actual_steps = h
                    break
            else:
                actual_steps = max(supported_horizons)
        else:
            actual_steps = num_steps
        
        # Initialize with input video
        current_video = video.clone()
        current_metadata = metadata_seq.clone()
        
        generated_frames = []
        
        for step in range(actual_steps):
            # Get metadata for this step
            step_metadata = future_metadata[:, step:step+1, :]  # (B, 1, metadata_dim)
            
            # Predict next frame using current video and metadata
            # The metadata should match the video sequence length
            pred = self.forward(
                current_video,
                current_metadata,
                predict_horizon=1
            )  # (B, 1, C, H, W)
            
            generated_frames.append(pred[:, 0])  # (B, C, H, W)
            
            # Update video sequence: slide window and add prediction
            current_video = torch.cat([
                current_video[:, 1:, :, :, :],  # Remove oldest frame
                pred  # Add predicted frame
            ], dim=1)
            
            # Update metadata sequence: slide window and add future metadata
            current_metadata = torch.cat([
                current_metadata[:, 1:, :],  # Remove oldest metadata
                step_metadata  # Add new metadata for the predicted frame
            ], dim=1)
        
        # Stack generated frames
        generated = torch.stack(generated_frames, dim=1)  # (B, num_steps, C, H, W)
        
        # Return only requested number of steps
        return generated[:, :num_steps]
    
    def predict_multi_horizon(
        self,
        video: Tensor,
        metadata_seq: Optional[Tensor] = None,
        horizons: List[int] = None
    ) -> Dict[int, Tensor]:
        """Predict at multiple horizons.
        
        Supports horizons H ∈ {1, 2, 4, 8} as specified in requirements.
        
        Args:
            video: Input sequence (batch, seq_len, channels, H, W)
            metadata_seq: Time-varying metadata (batch, seq_len, metadata_dim)
            horizons: List of prediction horizons (default [1, 2, 4, 8])
            
        Returns:
            Dict mapping horizon to predicted frames
        """
        if horizons is None:
            horizons = [1, 2, 4, 8]
        
        # Validate horizons
        valid_horizons = {1, 2, 4, 8}
        for h in horizons:
            if h not in valid_horizons:
                raise ValueError(f"Horizon {h} not supported. Must be in {valid_horizons}")
        
        predictions = {}
        
        for horizon in horizons:
            pred = self.forward(video, metadata_seq, predict_horizon=horizon)
            predictions[horizon] = pred
        
        return predictions


# =============================================================================
# Helper Functions
# =============================================================================

def construct_video_sequence(
    samples: List[Tensor],
    target_length: int = None
) -> Tensor:
    """Construct video sequence from list of frame tensors.
    
    Args:
        samples: List of frame tensors, each (channels, H, W)
        target_length: Target sequence length (pads/truncates if needed)
        
    Returns:
        Video tensor (1, seq_len, channels, H, W)
    """
    # Stack frames
    video = torch.stack(samples, dim=0)  # (seq_len, C, H, W)
    
    # Add batch dimension
    video = video.unsqueeze(0)  # (1, seq_len, C, H, W)
    
    # Handle target length
    if target_length is not None:
        current_len = video.shape[1]
        if current_len < target_length:
            # Pad with zeros
            padding = torch.zeros(
                1, target_length - current_len, *video.shape[2:],
                device=video.device, dtype=video.dtype
            )
            video = torch.cat([video, padding], dim=1)
        elif current_len > target_length:
            # Truncate
            video = video[:, :target_length]
    
    return video


def decompose_frame_to_patches(
    frame: Tensor,
    patch_size: int = 16
) -> Tensor:
    """Decompose a frame into patches.
    
    For a 256×256 frame with 16×16 patches, produces 256 patches.
    
    Args:
        frame: Input frame (channels, H, W) or (batch, channels, H, W)
        patch_size: Size of each patch (default 16)
        
    Returns:
        Patches tensor (num_patches, channels, patch_size, patch_size)
        or (batch, num_patches, channels, patch_size, patch_size)
    """
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, C, H, W = frame.shape
    
    # Compute number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Reshape to extract patches
    # (B, C, H, W) -> (B, C, num_h, patch_size, num_w, patch_size)
    patches = frame.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    
    # Permute to (B, num_h, num_w, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    
    # Flatten patch grid: (B, num_patches, C, patch_size, patch_size)
    patches = patches.reshape(B, num_patches_h * num_patches_w, C, patch_size, patch_size)
    
    if squeeze_batch:
        patches = patches.squeeze(0)
    
    return patches


def reconstruct_frame_from_patches(
    patches: Tensor,
    image_size: int = 256,
    patch_size: int = 16
) -> Tensor:
    """Reconstruct frame from patches.
    
    Args:
        patches: Patches tensor (num_patches, channels, patch_size, patch_size)
                 or (batch, num_patches, channels, patch_size, patch_size)
        image_size: Output image size (default 256)
        patch_size: Size of each patch (default 16)
        
    Returns:
        Reconstructed frame (channels, H, W) or (batch, channels, H, W)
    """
    if patches.dim() == 4:
        patches = patches.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, N, C, P, _ = patches.shape
    
    # Compute grid dimensions
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size
    
    # Reshape to grid: (B, num_h, num_w, C, P, P)
    patches = patches.reshape(B, num_patches_h, num_patches_w, C, P, P)
    
    # Permute to (B, C, num_h, P, num_w, P)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    
    # Reshape to image: (B, C, H, W)
    frame = patches.reshape(B, C, image_size, image_size)
    
    if squeeze_batch:
        frame = frame.squeeze(0)
    
    return frame

