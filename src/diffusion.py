"""Compositional Diffusion Model for microbiome generation.

This module implements a diffusion model that operates in CLR (Centered Log-Ratio)
space to generate valid microbiome compositions while respecting simplex constraints.

The diffusion process:
1. Forward: Gradually add Gaussian noise in CLR space
2. Reverse: Learn to denoise using a U-Net with metadata conditioning
3. Output: Apply inverse CLR to get valid compositions on the simplex

References:
    Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models.
    Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.clr_transform import CLRTransform


# =============================================================================
# Noise Schedule
# =============================================================================

class NoiseSchedule:
    """Linear beta schedule for diffusion process.
    
    Implements the noise schedule β_t ∈ [beta_start, beta_end] with linear
    interpolation, and computes all derived quantities needed for diffusion.
    
    Attributes:
        num_timesteps: Number of diffusion steps (T)
        beta_start: Starting noise level
        beta_end: Ending noise level
        betas: Noise schedule β_t for each timestep
        alphas: 1 - β_t
        alphas_cumprod: Cumulative product of alphas (ᾱ_t)
        alphas_cumprod_prev: ᾱ_{t-1}
        sqrt_alphas_cumprod: √ᾱ_t
        sqrt_one_minus_alphas_cumprod: √(1 - ᾱ_t)
        sqrt_recip_alphas: 1/√α_t
        posterior_variance: Variance for posterior q(x_{t-1}|x_t, x_0)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None
    ):
        """Initialize noise schedule.
        
        Args:
            num_timesteps: Number of diffusion steps (default 1000)
            beta_start: Starting noise level (default 0.0001)
            beta_end: Ending noise level (default 0.02)
            device: Device to place tensors on
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device or torch.device('cpu')
        
        # Linear beta schedule: β_t linearly interpolates from beta_start to beta_end
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=self.device
        )
        
        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # ᾱ_{t-1} with ᾱ_0 = 1
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        
        # Useful precomputed quantities for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod + 1e-8)
        )
        
        # Clamp posterior variance to avoid numerical issues at t=0
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
        # Log variance for numerical stability
        self.posterior_log_variance = torch.log(self.posterior_variance)
        
        # Coefficients for posterior mean
        # μ̃_t = (√ᾱ_{t-1} * β_t / (1 - ᾱ_t)) * x_0 + (√α_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * x_t
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod + 1e-8)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod + 1e-8)
        )
    
    def to(self, device: torch.device) -> 'NoiseSchedule':
        """Move all tensors to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def get_index(self, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
        """Extract values at timestep t and reshape for broadcasting.
        
        Args:
            t: Timestep indices (batch_size,)
            x_shape: Shape of the tensor to broadcast to
            
        Returns:
            Values at timestep t, reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
        return t.view(*out_shape)



# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps.
    
    Uses sinusoidal functions at different frequencies to encode
    the diffusion timestep, similar to transformer positional encodings.
    
    Attributes:
        dim: Output embedding dimension
        num_frequencies: Number of frequency bands (default 256)
    """
    
    def __init__(self, dim: int, num_frequencies: int = 256):
        """Initialize time embedding.
        
        Args:
            dim: Output embedding dimension
            num_frequencies: Number of frequency bands
        """
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies
        
        # MLP to project sinusoidal features to desired dimension
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t: Tensor) -> Tensor:
        """Compute time embedding.
        
        Args:
            t: Timestep indices (batch_size,) or scalar
            
        Returns:
            Time embeddings (batch_size, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        device = t.device
        half_dim = self.num_frequencies
        
        # Compute frequencies: exp(-log(10000) * i / half_dim)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Compute sinusoidal features
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Project through MLP
        return self.mlp(embedding)


# =============================================================================
# Attention Modules
# =============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial features.
    
    Applies self-attention across spatial dimensions of feature maps.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        """Initialize self-attention.
        
        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention.
        
        Args:
            x: Input features (batch, channels, H, W)
            
        Returns:
            Output features (batch, channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project and residual
        return x + self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for metadata conditioning.
    
    Allows the model to attend to metadata embeddings for conditioning.
    """
    
    def __init__(self, channels: int, context_dim: int = 512, num_heads: int = 8):
        """Initialize cross-attention.
        
        Args:
            channels: Number of input/output channels
            context_dim: Dimension of context (metadata) embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Apply cross-attention with context.
        
        Args:
            x: Input features (batch, channels, H, W)
            context: Context embeddings (batch, context_dim) or (batch, seq_len, context_dim)
            
        Returns:
            Output features (batch, channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Query from spatial features
        q = self.q(h)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        
        # Key, Value from context
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, context_dim)
        
        kv = self.kv(context)  # (B, seq_len, channels * 2)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, heads, seq_len, head_dim)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project and residual
        return x + self.proj(out)


# =============================================================================
# U-Net Building Blocks
# =============================================================================

class ResBlock(nn.Module):
    """Residual block with time embedding injection.
    
    Standard residual block with GroupNorm, SiLU activation,
    and additive time embedding conditioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1
    ):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_dim: Dimension of time embedding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Forward pass with time conditioning.
        
        Args:
            x: Input features (batch, in_channels, H, W)
            t_emb: Time embedding (batch, time_dim)
            
        Returns:
            Output features (batch, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder.
    
    Contains residual blocks, optional attention, and downsampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
        context_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_dim: Dimension of time embedding
            num_res_blocks: Number of residual blocks
            use_attention: Whether to use self-attention
            use_cross_attention: Whether to use cross-attention
            context_dim: Dimension of context for cross-attention
            dropout: Dropout probability
        """
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim, dropout))
            
            if use_attention:
                self.attentions.append(SelfAttention(out_channels))
            else:
                self.attentions.append(nn.Identity())
            
            if use_cross_attention:
                self.cross_attentions.append(CrossAttention(out_channels, context_dim))
            else:
                self.cross_attentions.append(None)
        
        # Downsampling
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self, 
        x: Tensor, 
        t_emb: Tensor, 
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass with skip connections.
        
        Args:
            x: Input features (batch, in_channels, H, W)
            t_emb: Time embedding (batch, time_dim)
            context: Optional context for cross-attention
            
        Returns:
            Tuple of (downsampled output, list of skip features)
        """
        skips = []
        
        for res_block, attn, cross_attn in zip(
            self.res_blocks, self.attentions, self.cross_attentions
        ):
            x = res_block(x, t_emb)
            x = attn(x)
            if cross_attn is not None and context is not None:
                x = cross_attn(x, context)
            skips.append(x)
        
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder.
    
    Contains upsampling, skip connection concatenation, residual blocks,
    and optional attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_cross_attention: bool = False,
        context_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels (after concat with skip)
            out_channels: Number of output channels
            time_dim: Dimension of time embedding
            num_res_blocks: Number of residual blocks
            use_attention: Whether to use self-attention
            use_cross_attention: Whether to use cross-attention
            context_dim: Dimension of context for cross-attention
            dropout: Dropout probability
        """
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        
        for i in range(num_res_blocks):
            # First block takes concatenated features
            in_ch = in_channels + out_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(in_ch, out_channels, time_dim, dropout))
            
            if use_attention:
                self.attentions.append(SelfAttention(out_channels))
            else:
                self.attentions.append(nn.Identity())
            
            if use_cross_attention:
                self.cross_attentions.append(CrossAttention(out_channels, context_dim))
            else:
                self.cross_attentions.append(None)
    
    def forward(
        self,
        x: Tensor,
        skips: List[Tensor],
        t_emb: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with skip connections.
        
        Args:
            x: Input features (batch, in_channels, H, W)
            skips: List of skip features from encoder
            t_emb: Time embedding (batch, time_dim)
            context: Optional context for cross-attention
            
        Returns:
            Output features (batch, out_channels, H*2, W*2)
        """
        x = self.upsample(x)
        
        for i, (res_block, attn, cross_attn) in enumerate(zip(
            self.res_blocks, self.attentions, self.cross_attentions
        )):
            if i == 0 and skips:
                # Concatenate skip connection
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            
            x = res_block(x, t_emb)
            x = attn(x)
            if cross_attn is not None and context is not None:
                x = cross_attn(x, context)
        
        return x


class MiddleBlock(nn.Module):
    """Middle block of U-Net (bottleneck).
    
    Contains residual blocks with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        channels: int,
        time_dim: int,
        context_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize middle block.
        
        Args:
            channels: Number of channels
            time_dim: Dimension of time embedding
            context_dim: Dimension of context for cross-attention
            dropout: Dropout probability
        """
        super().__init__()
        
        self.res1 = ResBlock(channels, channels, time_dim, dropout)
        self.attn = SelfAttention(channels)
        self.cross_attn = CrossAttention(channels, context_dim)
        self.res2 = ResBlock(channels, channels, time_dim, dropout)
    
    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        context: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input features
            t_emb: Time embedding
            context: Optional context for cross-attention
            
        Returns:
            Output features
        """
        x = self.res1(x, t_emb)
        x = self.attn(x)
        if context is not None:
            x = self.cross_attn(x, context)
        x = self.res2(x, t_emb)
        return x



# =============================================================================
# Metadata Encoder
# =============================================================================

class MetadataEncoder(nn.Module):
    """Encodes metadata into embeddings for cross-attention.
    
    Uses a 3-layer MLP to project metadata into 512-dimensional
    keys/values for cross-attention conditioning.
    """
    
    def __init__(self, metadata_dim: int, hidden_dim: int = 256, output_dim: int = 512):
        """Initialize metadata encoder.
        
        Args:
            metadata_dim: Dimension of input metadata
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (default 512)
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, metadata: Tensor) -> Tensor:
        """Encode metadata.
        
        Args:
            metadata: Input metadata (batch, metadata_dim)
            
        Returns:
            Encoded metadata (batch, output_dim)
        """
        return self.mlp(metadata)


# =============================================================================
# U-Net Denoiser
# =============================================================================

class UNetDenoiser(nn.Module):
    """U-Net architecture for denoising diffusion.
    
    Implements a U-Net with:
    - 4 downsampling blocks (256→128→64→32→16)
    - Self-attention at 16×16 resolution
    - Cross-attention for metadata injection
    - Sinusoidal time embedding (256 frequencies)
    - Skip connections between encoder and decoder
    
    Attributes:
        image_size: Input/output image resolution
        in_channels: Number of input channels
        model_channels: Base channel count
        channel_mult: Channel multipliers for each level
        num_res_blocks: Residual blocks per level
        attention_resolutions: Resolutions to apply self-attention
        metadata_dim: Dimension of metadata input
        context_dim: Dimension of metadata embeddings for cross-attention
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 16,
        model_channels: int = 64,
        channel_mult: List[int] = None,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        metadata_dim: int = 128,
        context_dim: int = 512,
        dropout: float = 0.1,
        num_time_frequencies: int = 256
    ):
        """Initialize U-Net denoiser.
        
        Args:
            image_size: Input image resolution (default 256)
            in_channels: Number of input channels (default 16)
            model_channels: Base channel count (default 64)
            channel_mult: Channel multipliers (default [1, 2, 4, 8, 8])
            num_res_blocks: Residual blocks per level (default 2)
            attention_resolutions: Resolutions for self-attention (default [16])
            metadata_dim: Dimension of metadata input (default 128)
            context_dim: Dimension of context embeddings (default 512)
            dropout: Dropout probability (default 0.1)
            num_time_frequencies: Frequencies for time embedding (default 256)
        """
        super().__init__()
        
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8, 8]  # 5 levels: 256→128→64→32→16
        if attention_resolutions is None:
            attention_resolutions = [16]  # Self-attention at 16×16
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        
        time_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim, num_time_frequencies)
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(metadata_dim, output_dim=context_dim)
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        current_res = image_size
        
        for i, mult in enumerate(channel_mult[:-1]):  # Don't downsample at last level
            out_ch = model_channels * mult
            use_attn = current_res in attention_resolutions
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    time_dim=time_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=True,
                    context_dim=context_dim,
                    dropout=dropout
                )
            )
            ch = out_ch
            current_res //= 2
        
        # Middle block (at lowest resolution)
        middle_ch = model_channels * channel_mult[-1]
        self.middle_conv = nn.Conv2d(ch, middle_ch, 3, padding=1)
        self.middle_block = MiddleBlock(
            channels=middle_ch,
            time_dim=time_dim,
            context_dim=context_dim,
            dropout=dropout
        )
        ch = middle_ch
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult[:-1])):
            out_ch = model_channels * mult
            use_attn = current_res in attention_resolutions
            
            self.up_blocks.append(
                UpBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    time_dim=time_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    use_cross_attention=True,
                    context_dim=context_dim,
                    dropout=dropout
                )
            )
            ch = out_ch
            current_res *= 2
        
        # Output convolution
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        
        # Initialize output conv to zero for stable training
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        metadata: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass: predict noise.
        
        Args:
            x: Noisy input (batch, channels, H, W)
            t: Timestep indices (batch,)
            metadata: Optional metadata for conditioning (batch, metadata_dim)
            
        Returns:
            Predicted noise (batch, channels, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Metadata embedding
        if metadata is not None:
            context = self.metadata_encoder(metadata)
        else:
            context = None
        
        # Input
        h = self.input_conv(x)
        
        # Encoder with skip connections
        all_skips = []
        for down_block in self.down_blocks:
            h, skips = down_block(h, t_emb, context)
            all_skips.extend(skips)
        
        # Middle
        h = self.middle_conv(h)
        h = self.middle_block(h, t_emb, context)
        
        # Decoder with skip connections
        for up_block in self.up_blocks:
            # Get skips for this level
            num_skips = up_block.res_blocks.__len__()
            level_skips = [all_skips.pop() for _ in range(num_skips)]
            h = up_block(h, level_skips, t_emb, context)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h



# =============================================================================
# Compositional Diffusion Model
# =============================================================================

class CompositionalDiffusion(nn.Module):
    """Diffusion model operating in CLR space for microbiome generation.
    
    This model performs diffusion in the CLR (Centered Log-Ratio) transformed
    space, ensuring that generated samples satisfy simplex constraints when
    transformed back via inverse CLR.
    
    The diffusion process:
    1. Forward: Add Gaussian noise in CLR space with linear schedule
    2. Reverse: Denoise using U-Net with metadata conditioning
    3. Output: Apply inverse CLR to get valid compositions
    
    Attributes:
        image_size: Spatial resolution of rasterized images
        in_channels: Number of functional channels
        num_timesteps: Number of diffusion steps
        clr_transform: CLR transformation module
        noise_schedule: Noise schedule parameters
        denoiser: U-Net denoiser network
    """
    
    def __init__(
        self,
        num_taxa: int,
        image_size: int = 256,
        in_channels: int = 16,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = None,
        channel_mult: List[int] = None,
        metadata_dim: int = 128,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        lambda_comp: float = 0.1,
        lambda_phylo: float = 0.05,
        phylogenetic_distances: Optional[Tensor] = None
    ):
        """Initialize compositional diffusion model.
        
        Args:
            num_taxa: Number of taxa in compositions
            image_size: Spatial resolution (default 256)
            in_channels: Number of channels (default 16)
            model_channels: Base U-Net channels (default 64)
            num_res_blocks: Residual blocks per level (default 2)
            attention_resolutions: Resolutions for self-attention (default [16])
            channel_mult: Channel multipliers (default [1, 2, 4, 8, 8])
            metadata_dim: Dimension of metadata (default 128)
            num_timesteps: Diffusion steps (default 1000)
            beta_start: Starting noise level (default 0.0001)
            beta_end: Ending noise level (default 0.02)
            lambda_comp: Compositional constraint loss weight (default 0.1)
            lambda_phylo: Phylogenetic coherence loss weight (default 0.05)
            phylogenetic_distances: Pairwise phylogenetic distances for coherence loss
        """
        super().__init__()
        
        if attention_resolutions is None:
            attention_resolutions = [16]
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8, 8]
        
        self.num_taxa = num_taxa
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.lambda_comp = lambda_comp
        self.lambda_phylo = lambda_phylo
        
        # CLR transformation
        self.clr_transform = CLRTransform(num_taxa)
        
        # Noise schedule
        self.noise_schedule = NoiseSchedule(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # U-Net denoiser
        self.denoiser = UNetDenoiser(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            metadata_dim=metadata_dim,
            context_dim=512,
            dropout=0.1
        )
        
        # Phylogenetic distances for coherence loss
        if phylogenetic_distances is not None:
            self.register_buffer('phylogenetic_distances', phylogenetic_distances)
        else:
            self.phylogenetic_distances = None
    
    def to(self, device: torch.device) -> 'CompositionalDiffusion':
        """Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        super().to(device)
        self.noise_schedule = self.noise_schedule.to(device)
        return self
    
    # =========================================================================
    # Forward Diffusion (q_sample)
    # =========================================================================
    
    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None
    ) -> Tensor:
        """Forward diffusion: add noise at timestep t.
        
        Implements the forward process q(x_t | x_0):
            x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        
        where ε ~ N(0, I) and ᾱ_t is the cumulative product of alphas.
        
        Args:
            x_0: Clean data in CLR space (batch, channels, H, W)
            t: Timestep indices (batch,)
            noise: Optional pre-sampled noise (batch, channels, H, W)
            
        Returns:
            Noisy data x_t (batch, channels, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get schedule values at timestep t
        sqrt_alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: (batch,) -> (batch, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # Forward diffusion: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t
    
    # =========================================================================
    # Reverse Diffusion (p_sample)
    # =========================================================================
    
    def _predict_x0_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        predicted_noise: Tensor
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - √(1 - ᾱ_t) * ε) / √ᾱ_t
        
        Args:
            x_t: Noisy data at timestep t
            t: Timestep indices
            predicted_noise: Predicted noise from denoiser
            
        Returns:
            Predicted clean data x_0
        """
        sqrt_alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / (sqrt_alphas_cumprod_t + 1e-8)
        return x_0
    
    def _posterior_mean_variance(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute posterior mean and variance q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Predicted clean data
            x_t: Noisy data at timestep t
            t: Timestep indices
            
        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        coef1 = self.noise_schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.noise_schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        variance = self.noise_schedule.posterior_variance[t].view(-1, 1, 1, 1)
        
        posterior_mean = coef1 * x_0 + coef2 * x_t
        return posterior_mean, variance
    
    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        metadata: Optional[Tensor] = None
    ) -> Tensor:
        """Reverse diffusion: denoise one step.
        
        Samples from p(x_{t-1} | x_t) using the learned denoiser.
        
        Args:
            x_t: Noisy data at timestep t (batch, channels, H, W)
            t: Timestep indices (batch,)
            metadata: Optional metadata for conditioning (batch, metadata_dim)
            
        Returns:
            Denoised data x_{t-1} (batch, channels, H, W)
        """
        # Predict noise
        predicted_noise = self.denoiser(x_t, t, metadata)
        
        # Predict x_0
        x_0_pred = self._predict_x0_from_noise(x_t, t, predicted_noise)
        
        # Compute posterior mean and variance
        posterior_mean, posterior_variance = self._posterior_mean_variance(x_0_pred, x_t, t)
        
        # Sample x_{t-1}
        noise = torch.randn_like(x_t)
        
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        x_t_minus_1 = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(
        self,
        metadata: Tensor,
        num_samples: Optional[int] = None,
        return_intermediates: bool = False
    ) -> Tensor:
        """Generate new samples conditioned on metadata.
        
        Performs full reverse diffusion from pure noise to generate
        new microbiome compositions.
        
        Args:
            metadata: Metadata for conditioning (batch, metadata_dim) or (metadata_dim,)
            num_samples: Number of samples (inferred from metadata if not provided)
            return_intermediates: Whether to return intermediate steps
            
        Returns:
            Generated samples in CLR space (batch, channels, H, W)
            If return_intermediates, returns list of all timesteps
        """
        # Handle metadata shape
        if metadata.dim() == 1:
            if num_samples is None:
                num_samples = 1
            metadata = metadata.unsqueeze(0).expand(num_samples, -1)
        else:
            num_samples = metadata.shape[0]
        
        device = next(self.parameters()).device
        metadata = metadata.to(device)
        
        # Start from pure noise
        x = torch.randn(
            num_samples, self.in_channels, self.image_size, self.image_size,
            device=device
        )
        
        intermediates = [x] if return_intermediates else None
        
        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, metadata)
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return intermediates
        return x
    
    # =========================================================================
    # Training Loss
    # =========================================================================
    
    def _simple_loss(
        self,
        predicted_noise: Tensor,
        target_noise: Tensor
    ) -> Tensor:
        """Compute simple denoising loss (MSE on predicted noise).
        
        Args:
            predicted_noise: Noise predicted by denoiser
            target_noise: Actual noise added during forward diffusion
            
        Returns:
            MSE loss
        """
        return F.mse_loss(predicted_noise, target_noise)
    
    def _compositional_loss(
        self,
        x_0_pred: Tensor
    ) -> Tensor:
        """Compute compositional constraint loss.
        
        Encourages the predicted clean data to be valid in CLR space,
        which means it should be centered (sum to zero along taxa dimension).
        
        Args:
            x_0_pred: Predicted clean data in CLR space
            
        Returns:
            Compositional constraint loss
        """
        # CLR vectors should sum to zero (centered property)
        # We penalize deviation from zero-sum
        channel_sums = x_0_pred.mean(dim=(2, 3))  # Average over spatial dims
        return torch.mean(channel_sums ** 2)
    
    def _phylogenetic_loss(
        self,
        x_0_pred: Tensor
    ) -> Tensor:
        """Compute phylogenetic coherence loss.
        
        Encourages phylogenetically related taxa to have similar abundances.
        Uses the phylogenetic distance matrix to weight the coherence.
        
        Args:
            x_0_pred: Predicted clean data in CLR space
            
        Returns:
            Phylogenetic coherence loss (0 if no phylogenetic distances provided)
        """
        if self.phylogenetic_distances is None:
            return torch.tensor(0.0, device=x_0_pred.device)
        
        # Compute spatial average to get per-channel values
        channel_values = x_0_pred.mean(dim=(2, 3))  # (batch, channels)
        
        # Compute pairwise differences
        batch_size, num_channels = channel_values.shape
        
        # Expand for pairwise computation
        v1 = channel_values.unsqueeze(2)  # (batch, channels, 1)
        v2 = channel_values.unsqueeze(1)  # (batch, 1, channels)
        
        # Pairwise squared differences
        diff_sq = (v1 - v2) ** 2  # (batch, channels, channels)
        
        # Weight by inverse phylogenetic distance (closer taxa should be more similar)
        # Normalize distances to [0, 1]
        if self.phylogenetic_distances.numel() > 0:
            max_dist = self.phylogenetic_distances.max() + 1e-8
            weights = 1.0 - (self.phylogenetic_distances / max_dist)
            
            # Truncate or pad weights to match channels
            if weights.shape[0] > num_channels:
                weights = weights[:num_channels, :num_channels]
            elif weights.shape[0] < num_channels:
                # Pad with zeros (no coherence constraint for extra channels)
                pad_size = num_channels - weights.shape[0]
                weights = F.pad(weights, (0, pad_size, 0, pad_size), value=0.0)
            
            # Apply weights
            weighted_diff = diff_sq * weights.unsqueeze(0)
            return weighted_diff.mean()
        
        return torch.tensor(0.0, device=x_0_pred.device)
    
    def training_loss(
        self,
        x_0: Tensor,
        metadata: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute training losses.
        
        Computes the combined loss:
            total = simple_loss + λ_c * comp_loss + λ_p * phylo_loss
        
        Args:
            x_0: Clean data in CLR space (batch, channels, H, W)
            metadata: Optional metadata for conditioning (batch, metadata_dim)
            
        Returns:
            Dict with:
                - simple_loss: MSE on predicted noise
                - comp_loss: Compositional constraint violation
                - phylo_loss: Phylogenetic coherence loss
                - total_loss: Weighted combination
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.denoiser(x_t, t, metadata)
        
        # Predict x_0 for auxiliary losses
        x_0_pred = self._predict_x0_from_noise(x_t, t, predicted_noise)
        
        # Compute losses
        simple_loss = self._simple_loss(predicted_noise, noise)
        comp_loss = self._compositional_loss(x_0_pred)
        phylo_loss = self._phylogenetic_loss(x_0_pred)
        
        # Total loss with weights
        total_loss = simple_loss + self.lambda_comp * comp_loss + self.lambda_phylo * phylo_loss
        
        return {
            'simple_loss': simple_loss,
            'comp_loss': comp_loss,
            'phylo_loss': phylo_loss,
            'total_loss': total_loss
        }
    
    def forward(
        self,
        x_0: Tensor,
        metadata: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass: compute training loss.
        
        Args:
            x_0: Clean data in CLR space (batch, channels, H, W)
            metadata: Optional metadata for conditioning
            
        Returns:
            Dict of losses
        """
        return self.training_loss(x_0, metadata)
