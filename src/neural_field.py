"""Neural Microbiome Field module.

This module implements a continuous function mapping metadata and time to
microbiome composition, enabling counterfactual queries and intervention
optimization.
"""

import math
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous inputs.
    
    Applies positional encoding with multiple frequency bands to metadata
    and time inputs, enabling the network to capture high-frequency variations.
    
    Attributes:
        num_frequencies: Number of frequency bands (default 10)
    """
    
    def __init__(self, num_frequencies: int = 10):
        """Initialize positional encoding.
        
        Args:
            num_frequencies: Number of frequency bands for encoding
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Create frequency bands: [2^0, 2^1, ..., 2^(num_frequencies-1)]
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Encoded tensor of shape (batch, input_dim * (2 * num_frequencies + 1))
            Original features concatenated with sin and cos encodings
        """
        # x: (batch, input_dim)
        batch_size, input_dim = x.shape
        
        # Expand for broadcasting: (batch, input_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute frequencies: (batch, input_dim, num_frequencies)
        scaled = x_expanded * self.frequencies.view(1, 1, -1) * math.pi
        
        # Apply sin and cos
        sin_features = torch.sin(scaled)  # (batch, input_dim, num_frequencies)
        cos_features = torch.cos(scaled)  # (batch, input_dim, num_frequencies)
        
        # Flatten frequency dimension
        sin_features = sin_features.reshape(batch_size, input_dim * self.num_frequencies)
        cos_features = cos_features.reshape(batch_size, input_dim * self.num_frequencies)
        
        # Concatenate: original + sin + cos
        encoded = torch.cat([x, sin_features, cos_features], dim=-1)
        
        return encoded


class NeuralMicrobiomeField(nn.Module):
    """Continuous function mapping metadata and time to microbiome composition.
    
    Implements an 8-layer MLP with skip connections that outputs valid
    compositions on the simplex. Supports counterfactual queries and
    intervention optimization.
    
    Attributes:
        metadata_dim: Dimension of metadata input
        num_taxa: Number of taxa in output composition
        hidden_dim: Hidden layer dimension (default 512)
        num_layers: Number of MLP layers (default 8)
        skip_layer: Layer index for skip connection (default 4)
        num_frequencies: Number of positional encoding frequencies (default 10)
    """
    
    def __init__(
        self,
        metadata_dim: int,
        num_taxa: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        skip_layer: int = 4,
        num_frequencies: int = 10
    ):
        """Initialize neural microbiome field.
        
        Args:
            metadata_dim: Dimension of metadata input
            num_taxa: Number of taxa in output composition
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            skip_layer: Layer index for skip connection
            num_frequencies: Number of positional encoding frequencies
        """
        super().__init__()
        
        self.metadata_dim = metadata_dim
        self.num_taxa = num_taxa
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layer = skip_layer
        self.num_frequencies = num_frequencies
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_frequencies)
        
        # Input dimension after positional encoding
        # metadata_dim + 1 (time) -> each encoded to (2 * num_frequencies + 1) * original_dim
        input_features = metadata_dim + 1  # metadata + time
        encoded_dim = input_features * (2 * num_frequencies + 1)
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(encoded_dim, hidden_dim))
        
        # Middle layers
        for i in range(1, num_layers - 1):
            if i == skip_layer:
                # Skip connection: concatenate with input
                self.layers.append(nn.Linear(hidden_dim + encoded_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, num_taxa))
    
    def forward(self, metadata: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Query composition at given metadata and time.
        
        Args:
            metadata: Host characteristics of shape (batch, metadata_dim)
            time: Time point of shape (batch, 1) or (batch,)
            
        Returns:
            Composition on simplex of shape (batch, num_taxa)
        """
        # Ensure time has correct shape
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        
        # Concatenate metadata and time
        x = torch.cat([metadata, time], dim=-1)  # (batch, metadata_dim + 1)
        
        # Apply positional encoding
        x = self.pos_encoding(x)  # (batch, encoded_dim)
        
        # Store for skip connection
        skip_input = x
        
        # Forward through MLP
        for i, layer in enumerate(self.layers[:-1]):
            if i == self.skip_layer:
                # Apply skip connection
                x = torch.cat([x, skip_input], dim=-1)
            
            x = layer(x)
            x = F.relu(x)
        
        # Output layer (no activation yet)
        logits = self.layers[-1](x)
        
        # Apply softmax to ensure valid composition
        composition = F.softmax(logits, dim=-1)
        
        return composition
    
    def counterfactual(
        self,
        metadata: torch.Tensor,
        time: torch.Tensor,
        intervention: Dict[str, Any]
    ) -> torch.Tensor:
        """Evaluate composition under modified metadata.
        
        Args:
            metadata: Original host characteristics (batch, metadata_dim)
            time: Time point (batch, 1) or (batch,)
            intervention: Dictionary mapping metadata indices to new values
                         e.g., {0: 25.0, 3: 1.5} modifies dimensions 0 and 3
            
        Returns:
            Counterfactual composition (batch, num_taxa)
        """
        # Clone metadata to avoid modifying original
        modified_metadata = metadata.clone()
        
        # Apply interventions
        for idx, value in intervention.items():
            if isinstance(idx, int):
                modified_metadata[:, idx] = value
            else:
                raise ValueError(f"Intervention key must be int, got {type(idx)}")
        
        # Query field with modified metadata
        return self.forward(modified_metadata, time)
    
    def optimize_intervention(
        self,
        target_composition: torch.Tensor,
        time: torch.Tensor,
        fixed_metadata: Dict[int, torch.Tensor],
        optimizable_indices: List[int],
        num_steps: int = 100,
        learning_rate: float = 0.01
    ) -> torch.Tensor:
        """Find metadata values to achieve target composition.
        
        Uses gradient descent to optimize specified metadata dimensions
        while keeping others fixed.
        
        Args:
            target_composition: Desired composition (batch, num_taxa)
            time: Time point (batch, 1) or (batch,)
            fixed_metadata: Dictionary mapping indices to fixed values
            optimizable_indices: List of metadata indices to optimize
            num_steps: Number of optimization steps
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Optimized metadata (batch, metadata_dim)
        """
        batch_size = target_composition.shape[0]
        device = target_composition.device
        
        # Initialize metadata
        metadata = torch.zeros(batch_size, self.metadata_dim, device=device)
        
        # Set fixed values
        for idx, value in fixed_metadata.items():
            if isinstance(value, torch.Tensor):
                metadata[:, idx] = value
            else:
                metadata[:, idx] = float(value)
        
        # Initialize optimizable parameters
        optimizable_params = torch.zeros(
            batch_size, len(optimizable_indices),
            device=device, requires_grad=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam([optimizable_params], lr=learning_rate)
        
        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Update metadata with current optimizable values
            current_metadata = metadata.clone()
            for i, idx in enumerate(optimizable_indices):
                current_metadata[:, idx] = optimizable_params[:, i]
            
            # Predict composition
            predicted = self.forward(current_metadata, time)
            
            # Compute loss (KL divergence)
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            loss = F.kl_div(
                torch.log(predicted + eps),
                target_composition,
                reduction='batchmean'
            )
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        # Construct final metadata
        final_metadata = metadata.clone()
        for i, idx in enumerate(optimizable_indices):
            final_metadata[:, idx] = optimizable_params[:, i].detach()
        
        return final_metadata
    
    def training_loss(
        self,
        metadata: torch.Tensor,
        time: torch.Tensor,
        observed_composition: torch.Tensor,
        lambda_r: float = 1e-5
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with KL divergence and L2 regularization.
        
        Args:
            metadata: Host characteristics (batch, metadata_dim)
            time: Time point (batch, 1) or (batch,)
            observed_composition: Ground truth composition (batch, num_taxa)
            lambda_r: L2 regularization weight
            
        Returns:
            Dictionary containing:
                - kl_loss: KL divergence between predicted and observed
                - reg_loss: L2 regularization on parameters
                - total_loss: Combined loss
        """
        # Predict composition
        predicted = self.forward(metadata, time)
        
        # KL divergence loss
        eps = 1e-8
        kl_loss = F.kl_div(
            torch.log(predicted + eps),
            observed_composition,
            reduction='batchmean'
        )
        
        # L2 regularization
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss = reg_loss + torch.sum(param ** 2)
        reg_loss = lambda_r * reg_loss
        
        # Total loss
        total_loss = kl_loss + reg_loss
        
        return {
            'kl_loss': kl_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }
