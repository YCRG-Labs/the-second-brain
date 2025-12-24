"""Model serialization and checkpoint management.

This module provides functionality for saving and loading model checkpoints,
preserving numerical precision for hyperbolic embeddings and ensuring
architecture compatibility.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
import json


class ModelSerializer:
    """Save and load model checkpoints with validation."""
    
    @staticmethod
    def save_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Save model state, optimizer, and config to file.
        
        This method preserves full numerical precision for all model parameters,
        including hyperbolic embeddings which require high precision to maintain
        points within the Poincaré ball.
        
        Args:
            path: File path to save checkpoint
            model: PyTorch model to save
            optimizer: Optional optimizer state to save
            config: Optional configuration dictionary
            epoch: Optional epoch number
            **kwargs: Additional metadata to save
            
        Raises:
            IOError: If unable to write to path
            RuntimeError: If model state cannot be serialized
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': _extract_model_config(model),
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['optimizer_class'] = optimizer.__class__.__name__
        
        # Add training metadata
        if config is not None:
            checkpoint['config'] = config
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # Add any additional metadata
        checkpoint.update(kwargs)
        
        # Save with full precision (default for torch.save)
        try:
            torch.save(checkpoint, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint to {path}: {e}")
    
    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        strict: bool = True,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load checkpoint and restore model state.
        
        This method validates architecture compatibility before loading and
        restores the exact model state for reproducible inference.
        
        Args:
            path: File path to load checkpoint from
            model: PyTorch model to load state into
            optimizer: Optional optimizer to restore state
            strict: Whether to strictly enforce state dict keys match
            device: Device to load tensors to (default: same as saved)
            
        Returns:
            Dictionary containing config and training state metadata
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible with model
            ValueError: If checkpoint format is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        
        # Load checkpoint
        try:
            if device is not None:
                checkpoint = torch.load(path, map_location=device)
            else:
                checkpoint = torch.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
        
        # Validate checkpoint format
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint format: missing 'model_state_dict'")
        
        # Validate architecture compatibility
        if not ModelSerializer.validate_checkpoint(path, model):
            raise RuntimeError(
                f"Checkpoint architecture incompatible with model. "
                f"Expected {model.__class__.__name__}, "
                f"got {checkpoint.get('model_class', 'unknown')}"
            )
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}")
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                raise RuntimeError(f"Failed to load optimizer state: {e}")
        
        # Return metadata
        metadata = {
            'config': checkpoint.get('config'),
            'epoch': checkpoint.get('epoch'),
            'model_class': checkpoint.get('model_class'),
            'optimizer_class': checkpoint.get('optimizer_class'),
        }
        
        # Include any additional keys
        for key in checkpoint:
            if key not in ['model_state_dict', 'optimizer_state_dict', 
                          'model_class', 'model_config', 'optimizer_class',
                          'config', 'epoch']:
                metadata[key] = checkpoint[key]
        
        return metadata
    
    @staticmethod
    def validate_checkpoint(path: str, model: nn.Module) -> bool:
        """Check if checkpoint is compatible with model architecture.
        
        This method performs validation without loading the full checkpoint,
        checking that layer shapes and model class match.
        
        Args:
            path: File path to checkpoint
            model: PyTorch model to validate against
            
        Returns:
            True if checkpoint is compatible, False otherwise
        """
        if not os.path.exists(path):
            return False
        
        try:
            # Load checkpoint metadata only
            checkpoint = torch.load(path, map_location='cpu')
            
            # Check model class
            if 'model_class' in checkpoint:
                if checkpoint['model_class'] != model.__class__.__name__:
                    return False
            
            # Check state dict keys and shapes
            if 'model_state_dict' not in checkpoint:
                return False
            
            checkpoint_state = checkpoint['model_state_dict']
            model_state = model.state_dict()
            
            # Check all keys exist
            checkpoint_keys = set(checkpoint_state.keys())
            model_keys = set(model_state.keys())
            
            if checkpoint_keys != model_keys:
                return False
            
            # Check tensor shapes match
            for key in checkpoint_keys:
                if checkpoint_state[key].shape != model_state[key].shape:
                    return False
            
            return True
            
        except Exception:
            return False


def _extract_model_config(model: nn.Module) -> Dict[str, Any]:
    """Extract configuration from model for serialization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model configuration parameters
    """
    config = {}
    
    # Try to extract common configuration attributes
    config_attrs = [
        'num_taxa', 'embedding_dim', 'curvature',  # HyperbolicEmbedder
        'image_size', 'num_channels', 'patch_size',  # Rasterizer, Transformer
        'hidden_dim', 'num_layers', 'num_heads',  # Various models
        'num_timesteps', 'beta_start', 'beta_end',  # Diffusion
        'metadata_dim', 'embed_dim', 'depth',  # Transformer
    ]
    
    for attr in config_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            # Only save serializable types
            if isinstance(value, (int, float, str, bool, list, tuple)):
                config[attr] = value
    
    return config
