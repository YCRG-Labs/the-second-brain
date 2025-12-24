"""Experiment runner infrastructure for training and evaluation.

This module provides the ExperimentRunner class for managing experiments,
including configuration management, logging setup, and checkpoint management.
"""

import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from .exceptions import MicrobiomeSimulationError


class ExperimentError(MicrobiomeSimulationError):
    """Raised when experiment execution fails."""
    pass


def _run_sensitivity_experiment_worker(
    exp_config: Dict[str, Any],
    model_factory: Callable[[Any], nn.Module],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader]
) -> Dict[str, Any]:
    """Worker function for parallel sensitivity analysis.
    
    This function is designed to be run in a separate process via
    ProcessPoolExecutor. It creates a new experiment runner and
    trains/evaluates a model with the specified configuration.
    
    Args:
        exp_config: Experiment configuration dictionary containing:
            - param_name: Parameter being varied
            - param_value: Value for this experiment
            - base_config: Base configuration dictionary
            - experiment_name: Name for this experiment
        model_factory: Function that creates model from config
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        
    Returns:
        Dictionary with experiment results
    """
    try:
        param_name = exp_config['param_name']
        param_value = exp_config['param_value']
        config_dict = exp_config['base_config'].copy()
        experiment_name = exp_config['experiment_name']
        
        # Modify config with parameter value
        if '.' in param_name:
            parts = param_name.split('.')
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = param_value
        else:
            config_dict[param_name] = param_value
        
        # Create config and model
        sensitivity_config = ExperimentConfig(**config_dict)
        model = model_factory(sensitivity_config)
        
        # Create runner
        runner = ExperimentRunner(sensitivity_config, experiment_name)
        
        # Train
        train_results = runner.train_model(model, train_loader, val_loader)
        
        # Evaluate
        if test_loader is not None:
            test_results = runner.evaluate_model(model, test_loader)
        else:
            test_results = {}
        
        return {
            'train': train_results,
            'test': test_results,
            'config': sensitivity_config.to_dict()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'config': exp_config.get('base_config', {})
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.
    
    Attributes:
        # Data configuration
        dataset: Dataset name
        train_split: Training set fraction
        val_split: Validation set fraction
        test_split: Test set fraction
        
        # Model configuration
        model_type: Type of model ('diffusion', 'temporal', 'field', 'vae', 'gan', 'lstm', 'transformer')
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        
        # Training configuration
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        weight_decay: L2 regularization weight
        
        # Loss weights
        lambda_comp: Compositional constraint loss weight
        lambda_phylo: Phylogenetic coherence loss weight
        
        # Reproducibility
        seed: Random seed
        
        # Output configuration
        output_dir: Directory for outputs
        checkpoint_freq: Checkpoint saving frequency (epochs)
        log_freq: Logging frequency (batches)
        
        # Additional parameters
        extra_params: Additional model-specific parameters
    """
    # Data
    dataset: str
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Model
    model_type: str = 'diffusion'
    embedding_dim: int = 32
    hidden_dim: int = 512
    num_layers: int = 8
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-2
    
    # Loss weights
    lambda_comp: float = 0.1
    lambda_phylo: float = 0.05
    
    # Reproducibility
    seed: int = 42
    
    # Output
    output_dir: str = './experiments'
    checkpoint_freq: int = 10
    log_freq: int = 100
    
    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Load config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Load config from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            ExperimentConfig instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class CheckpointManager:
    """Manages model checkpoints during training.
    
    Handles saving, loading, and managing checkpoints with support for
    keeping only the best checkpoints and periodic checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_best_n: int = 3,
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
            metric_name: Metric to use for determining best checkpoints
            mode: 'min' or 'max' for metric optimization
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best checkpoints
        self.best_checkpoints: List[Tuple[float, Path]] = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint so far
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'is_best': is_best
        }
        
        if extra_state:
            checkpoint.update(extra_state)
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        
        # Update best checkpoints list
        if self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            self._update_best_checkpoints(metric_value, checkpoint_path)
        
        return checkpoint_path
    
    def _update_best_checkpoints(self, metric_value: float, checkpoint_path: Path) -> None:
        """Update list of best checkpoints.
        
        Args:
            metric_value: Value of the metric
            checkpoint_path: Path to checkpoint
        """
        # Add new checkpoint
        self.best_checkpoints.append((metric_value, checkpoint_path))
        
        # Sort by metric value
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x[0])
        else:
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only best N
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove worst checkpoint
            _, worst_path = self.best_checkpoints.pop()
            if worst_path.exists() and 'best_model' not in worst_path.name:
                worst_path.unlink()
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint dictionary with epoch, metrics, etc.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint, or None if no checkpoints saved
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return best_path
        
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        
        return None


class ExperimentLogger:
    """Handles logging for experiments.
    
    Provides structured logging to console and file, with support for
    metrics tracking and progress reporting.
    """
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        
        # Set up logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f'{experiment_name}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics history
        self.metrics_history: Dict[str, List[float]] = {}
        self.metrics_file = self.log_dir / f'{experiment_name}_metrics.json'
    
    def log(self, message: str, level: str = 'info') -> None:
        """Log a message.
        
        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error', 'debug')
        """
        log_func = getattr(self.logger, level.lower())
        log_func(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step/epoch number
        """
        # Add to history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
        
        # Log to console
        metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
        self.log(f'Step {step}: {metrics_str}')
        
        # Save to file
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save metrics history to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get complete metrics history.
        
        Returns:
            Dictionary mapping metric names to lists of values
        """
        return self.metrics_history


class ExperimentRunner:
    """Manages experiment execution and result tracking.
    
    Provides a unified interface for:
    - Configuration management
    - Reproducibility (seed setting)
    - Training loop with logging and checkpointing
    - Evaluation pipeline
    - Result tracking and saving
    
    Example:
        >>> config = ExperimentConfig(
        ...     dataset='american_gut',
        ...     model_type='vae',
        ...     num_epochs=50
        ... )
        >>> runner = ExperimentRunner(config)
        >>> results = runner.train_model(model, train_loader, val_loader)
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        experiment_name: Optional[str] = None
    ):
        """Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            experiment_name: Optional experiment name (auto-generated if None)
        """
        self.config = config
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            experiment_name = f'{config.model_type}_{config.dataset}_{timestamp}'
        self.experiment_name = experiment_name
        
        # Set up output directory
        self.output_dir = Path(config.output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.results_dir = self.output_dir / 'results'
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.logger = ExperimentLogger(self.log_dir, experiment_name)
        
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # Save configuration
        config.to_yaml(self.output_dir / 'config.yaml')
        self.logger.log(f'Initialized experiment: {experiment_name}')
        self.logger.log(f'Output directory: {self.output_dir}')
        
        # Log environment information
        self._log_environment()
    
    def _set_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.log(f'Set random seed to {seed}')
    
    def _log_environment(self) -> None:
        """Log environment information for reproducibility."""
        env_info = {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cpu_count': os.cpu_count()
        }
        
        # Save to file
        env_file = self.output_dir / 'environment.json'
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        self.logger.log('Environment information:')
        for key, value in env_info.items():
            self.logger.log(f'  {key}: {value}')
    
    def get_device(self) -> torch.device:
        """Get device for training.
        
        Returns:
            torch.device (cuda if available, else cpu)
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def save_results(self, results: Dict[str, Any], filename: str = 'results.json') -> Path:
        """Save experiment results to file.
        
        Args:
            results: Results dictionary
            filename: Output filename
            
        Returns:
            Path to saved results file
        """
        results_path = self.results_dir / filename
        
        # Convert numpy arrays, tensors, and DataFrames to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.log(f'Results saved to {results_path}')
        return results_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def load_results(self, filename: str = 'results.json') -> Dict[str, Any]:
        """Load experiment results from file.
        
        Args:
            filename: Results filename
            
        Returns:
            Results dictionary
        """
        results_path = self.results_dir / filename
        
        if not results_path.exists():
            raise FileNotFoundError(f'Results file not found: {results_path}')
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train model with logging and checkpointing.
        
        Implements a complete training loop with:
        - Progress bars (tqdm)
        - Metric logging
        - Checkpoint saving
        - Early stopping (optional)
        - Learning rate scheduling (optional)
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            optimizer: Optional optimizer (creates Adam if None)
            scheduler: Optional learning rate scheduler
            loss_fn: Optional loss function (uses model's loss if None)
            early_stopping_patience: Epochs to wait before early stopping (None to disable)
            
        Returns:
            Dictionary with training history and final metrics
        """
        device = self.get_device()
        model = model.to(device)
        
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        self.logger.log(f'Starting training for {self.config.num_epochs} epochs')
        self.logger.log(f'Device: {device}')
        self.logger.log(f'Batch size: {self.config.batch_size}')
        self.logger.log(f'Learning rate: {self.config.learning_rate}')
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, loss_fn, epoch, device
            )
            
            # Validate if validation loader provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(
                    model, val_loader, loss_fn, epoch, device
                )
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                else:
                    scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            history['train_loss'].append(train_metrics['train_loss'])
            history['epoch_time'].append(epoch_time)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics, 'lr': optimizer.param_groups[0]['lr']}
            self.logger.log_metrics(all_metrics, epoch)
            
            # Save checkpoint
            is_best = False
            if val_loader is not None:
                current_val_loss = val_metrics['val_loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if epoch % self.config.checkpoint_freq == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, all_metrics, is_best
                )
            
            # Early stopping check
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                self.logger.log(
                    f'Early stopping triggered after {epoch + 1} epochs '
                    f'(patience={early_stopping_patience})'
                )
                break
        
        # Save final checkpoint
        final_metrics = {**train_metrics, **val_metrics}
        self.checkpoint_manager.save_checkpoint(
            model, optimizer, self.config.num_epochs - 1, final_metrics
        )
        
        # Prepare results
        results = {
            'history': history,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss if val_loader is not None else None,
            'total_epochs': epoch + 1,
            'config': self.config.to_dict()
        }
        
        # Save results
        self.save_results(results, 'training_results.json')
        
        self.logger.log('Training completed')
        self.logger.log(f'Best validation loss: {best_val_loss:.4f}' if val_loader else 'No validation performed')
        
        return results
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable],
        epoch: int,
        device: torch.device
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            epoch: Current epoch number
            device: Device to train on
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        try:
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
        except ImportError:
            pbar = train_loader
            self.logger.log(f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if loss_fn is not None:
                # Use provided loss function
                if isinstance(batch, (list, tuple)):
                    loss = loss_fn(model, *batch)
                else:
                    loss = loss_fn(model, batch)
            else:
                # Assume model returns loss
                if isinstance(batch, (list, tuple)):
                    output = model(*batch)
                else:
                    output = model(batch)
                
                if isinstance(output, dict) and 'loss' in output:
                    loss = output['loss']
                elif isinstance(output, tuple) and len(output) > 0:
                    loss = output[0] if isinstance(output[0], torch.Tensor) and output[0].dim() == 0 else output[0].mean()
                else:
                    loss = output
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': loss.item()})
            
            # Log batch metrics periodically
            if batch_idx % self.config.log_freq == 0:
                self.logger.log(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                    f'loss={loss.item():.4f}',
                    level='debug'
                )
        
        avg_loss = total_loss / num_batches
        
        return {'train_loss': avg_loss}
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Optional[Callable],
        epoch: int,
        device: torch.device
    ) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            loss_fn: Loss function
            epoch: Current epoch number
            device: Device to validate on
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        try:
            from tqdm import tqdm
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', leave=False)
        except ImportError:
            pbar = val_loader
            self.logger.log(f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                else:
                    batch = batch.to(device)
                
                # Forward pass
                if loss_fn is not None:
                    if isinstance(batch, (list, tuple)):
                        loss = loss_fn(model, *batch)
                    else:
                        loss = loss_fn(model, batch)
                else:
                    if isinstance(batch, (list, tuple)):
                        output = model(*batch)
                    else:
                        output = model(batch)
                    
                    if isinstance(output, dict) and 'loss' in output:
                        loss = output['loss']
                    elif isinstance(output, tuple) and len(output) > 0:
                        loss = output[0] if isinstance(output[0], torch.Tensor) and output[0].dim() == 0 else output[0].mean()
                    else:
                        loss = output
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        return {'val_loss': avg_loss}

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        evaluator: Optional[Any] = None,
        generate_samples: bool = False,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """Evaluate model on test set with comprehensive metrics.
        
        Runs inference on test data and computes all relevant metrics:
        - Generation quality (if applicable)
        - Prediction accuracy (if applicable)
        - Custom metrics (via evaluator)
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            evaluator: Optional MicrobiomeEvaluator for comprehensive metrics
            generate_samples: Whether to generate samples for quality evaluation
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with all evaluation metrics
        """
        device = self.get_device()
        model = model.to(device)
        model.eval()
        
        self.logger.log('Starting evaluation on test set')
        
        results = {
            'test_metrics': {},
            'predictions': [],
            'ground_truth': [],
            'generated_samples': None
        }
        
        # Run inference on test set
        self.logger.log('Running inference on test data...')
        
        try:
            from tqdm import tqdm
            pbar = tqdm(test_loader, desc='Evaluating', leave=False)
        except ImportError:
            pbar = test_loader
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch_data = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                else:
                    batch_data = batch.to(device)
                
                # Forward pass
                if isinstance(batch_data, (list, tuple)):
                    output = model(*batch_data)
                else:
                    output = model(batch_data)
                
                # Extract predictions and ground truth
                if isinstance(output, dict):
                    pred = output.get('predictions', output.get('output', None))
                elif isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output
                
                if pred is not None:
                    results['predictions'].append(pred.cpu().numpy())
                
                # Get ground truth (assume last element of batch is target)
                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    target = batch[-1]
                    if isinstance(target, torch.Tensor):
                        results['ground_truth'].append(target.cpu().numpy())
        
        # Convert lists to arrays
        if results['predictions']:
            results['predictions'] = np.concatenate(results['predictions'], axis=0)
        if results['ground_truth']:
            results['ground_truth'] = np.concatenate(results['ground_truth'], axis=0)
        
        # Generate samples if requested
        if generate_samples:
            self.logger.log(f'Generating {num_samples} samples...')
            
            if hasattr(model, 'sample'):
                generated = model.sample(num_samples, device=device)
                results['generated_samples'] = generated.cpu().numpy()
            elif hasattr(model, 'generate'):
                generated = model.generate(num_samples, device=device)
                results['generated_samples'] = generated.cpu().numpy()
            else:
                self.logger.log('Model does not have sample() or generate() method', level='warning')
        
        # Compute metrics using evaluator if provided
        if evaluator is not None:
            self.logger.log('Computing comprehensive metrics...')
            
            # Generation metrics
            if results['generated_samples'] is not None and results['ground_truth']:
                gen_metrics = evaluator.evaluate_generation(
                    results['ground_truth'],
                    results['generated_samples']
                )
                results['test_metrics'].update(gen_metrics)
            
            # Prediction metrics
            if results['predictions'] and results['ground_truth']:
                pred_metrics = evaluator.evaluate_prediction(
                    results['predictions'],
                    results['ground_truth']
                )
                results['test_metrics'].update(pred_metrics)
        
        # Save results
        self.save_results(results, 'evaluation_results.json')
        
        # Log summary
        self.logger.log('Evaluation completed')
        if results['test_metrics']:
            self.logger.log('Test metrics:')
            for metric_name, metric_value in results['test_metrics'].items():
                self.logger.log(f'  {metric_name}: {metric_value:.4f}')
        
        return results
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """Load best model checkpoint.
        
        Args:
            model: Model instance to load weights into
            
        Returns:
            Model with best checkpoint loaded
        """
        best_checkpoint_path = self.checkpoint_manager.get_best_checkpoint_path()
        
        if best_checkpoint_path is None:
            self.logger.log('No best checkpoint found', level='warning')
            return model
        
        self.logger.log(f'Loading best checkpoint from {best_checkpoint_path}')
        self.checkpoint_manager.load_checkpoint(best_checkpoint_path, model)
        
        return model
    
    def resume_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Resume training from a checkpoint.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            checkpoint_path: Path to checkpoint (uses best if None)
            
        Returns:
            Training results
        """
        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_best_checkpoint_path()
        
        if checkpoint_path is None:
            self.logger.log('No checkpoint found, starting from scratch', level='warning')
            return self.train_model(model, train_loader, val_loader)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer
        )
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.logger.log(f'Resuming training from epoch {start_epoch}')
        
        # Update config to start from checkpoint epoch
        original_epochs = self.config.num_epochs
        self.config.num_epochs = original_epochs - start_epoch
        
        # Train
        results = self.train_model(model, train_loader, val_loader, optimizer)
        
        # Restore original config
        self.config.num_epochs = original_epochs
        
        return results

    def run_ablation_study(
        self,
        ablation_configs: Dict[str, Dict[str, Any]],
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run ablation study with multiple configurations.
        
        Trains and evaluates models with different ablations to understand
        the contribution of each component.
        
        Args:
            ablation_configs: Dictionary mapping ablation names to config modifications
                             e.g., {'no_hyperbolic': {'use_hyperbolic': False}}
            model_factory: Function that creates model from config
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Dictionary mapping ablation names to their results
        """
        self.logger.log(f'Starting ablation study with {len(ablation_configs)} configurations')
        
        ablation_results = {}
        
        # Run baseline (full model)
        self.logger.log('Running baseline (full model)...')
        baseline_config = ExperimentConfig(**self.config.to_dict())
        baseline_model = model_factory(baseline_config)
        
        baseline_runner = ExperimentRunner(
            baseline_config,
            experiment_name=f'{self.experiment_name}_baseline'
        )
        
        baseline_train_results = baseline_runner.train_model(
            baseline_model, train_loader, val_loader
        )
        
        if test_loader is not None:
            baseline_test_results = baseline_runner.evaluate_model(
                baseline_model, test_loader
            )
        else:
            baseline_test_results = {}
        
        ablation_results['baseline'] = {
            'train': baseline_train_results,
            'test': baseline_test_results,
            'config': baseline_config.to_dict()
        }
        
        # Run each ablation
        for ablation_name, config_modifications in ablation_configs.items():
            self.logger.log(f'Running ablation: {ablation_name}')
            
            # Create modified config
            ablation_config_dict = self.config.to_dict()
            ablation_config_dict.update(config_modifications)
            ablation_config = ExperimentConfig(**ablation_config_dict)
            
            # Create model with ablation
            ablation_model = model_factory(ablation_config)
            
            # Create runner for this ablation
            ablation_runner = ExperimentRunner(
                ablation_config,
                experiment_name=f'{self.experiment_name}_{ablation_name}'
            )
            
            # Train
            try:
                train_results = ablation_runner.train_model(
                    ablation_model, train_loader, val_loader
                )
                
                # Evaluate
                if test_loader is not None:
                    test_results = ablation_runner.evaluate_model(
                        ablation_model, test_loader
                    )
                else:
                    test_results = {}
                
                ablation_results[ablation_name] = {
                    'train': train_results,
                    'test': test_results,
                    'config': ablation_config.to_dict()
                }
                
                self.logger.log(f'Completed ablation: {ablation_name}')
                
            except Exception as e:
                self.logger.log(
                    f'Ablation {ablation_name} failed: {str(e)}',
                    level='error'
                )
                ablation_results[ablation_name] = {
                    'error': str(e),
                    'config': ablation_config.to_dict()
                }
        
        # Aggregate results
        aggregated_results = self._aggregate_ablation_results(ablation_results)
        
        # Save results
        self.save_results(ablation_results, 'ablation_study_results.json')
        self.save_results(aggregated_results, 'ablation_study_summary.json')
        
        self.logger.log('Ablation study completed')
        self._log_ablation_summary(aggregated_results)
        
        return ablation_results
    
    def _aggregate_ablation_results(
        self,
        ablation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate ablation study results for comparison.
        
        Args:
            ablation_results: Raw ablation results
            
        Returns:
            Aggregated results with comparisons
        """
        aggregated = {
            'ablations': {},
            'baseline_metrics': {},
            'performance_degradation': {}
        }
        
        # Extract baseline metrics
        if 'baseline' in ablation_results:
            baseline = ablation_results['baseline']
            if 'train' in baseline and 'final_metrics' in baseline['train']:
                aggregated['baseline_metrics'] = baseline['train']['final_metrics']
        
        # Compare each ablation to baseline
        for ablation_name, results in ablation_results.items():
            if ablation_name == 'baseline' or 'error' in results:
                continue
            
            ablation_summary = {
                'final_train_loss': None,
                'final_val_loss': None,
                'test_metrics': {}
            }
            
            # Extract metrics
            if 'train' in results and 'final_metrics' in results['train']:
                metrics = results['train']['final_metrics']
                ablation_summary['final_train_loss'] = metrics.get('train_loss')
                ablation_summary['final_val_loss'] = metrics.get('val_loss')
            
            if 'test' in results and 'test_metrics' in results['test']:
                ablation_summary['test_metrics'] = results['test']['test_metrics']
            
            aggregated['ablations'][ablation_name] = ablation_summary
            
            # Compute performance degradation
            if aggregated['baseline_metrics']:
                degradation = {}
                
                baseline_val_loss = aggregated['baseline_metrics'].get('val_loss')
                ablation_val_loss = ablation_summary['final_val_loss']
                
                if baseline_val_loss is not None and ablation_val_loss is not None:
                    degradation['val_loss_increase'] = ablation_val_loss - baseline_val_loss
                    degradation['val_loss_relative'] = (
                        (ablation_val_loss - baseline_val_loss) / baseline_val_loss * 100
                    )
                
                aggregated['performance_degradation'][ablation_name] = degradation
        
        return aggregated
    
    def _log_ablation_summary(self, aggregated_results: Dict[str, Any]) -> None:
        """Log summary of ablation study results.
        
        Args:
            aggregated_results: Aggregated ablation results
        """
        self.logger.log('Ablation Study Summary:')
        self.logger.log('=' * 60)
        
        # Baseline
        if aggregated_results['baseline_metrics']:
            self.logger.log('Baseline:')
            for metric, value in aggregated_results['baseline_metrics'].items():
                self.logger.log(f'  {metric}: {value:.4f}')
        
        # Ablations
        self.logger.log('\nAblations:')
        for ablation_name, summary in aggregated_results['ablations'].items():
            self.logger.log(f'\n{ablation_name}:')
            if summary['final_val_loss'] is not None:
                self.logger.log(f'  val_loss: {summary["final_val_loss"]:.4f}')
            
            # Performance degradation
            if ablation_name in aggregated_results['performance_degradation']:
                degradation = aggregated_results['performance_degradation'][ablation_name]
                if 'val_loss_relative' in degradation:
                    self.logger.log(
                        f'  degradation: {degradation["val_loss_relative"]:.2f}%'
                    )
        
        self.logger.log('=' * 60)

    def run_sensitivity_analysis(
        self,
        param_name: str,
        param_values: List[Any],
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run sensitivity analysis over parameter range.
        
        Trains and evaluates models with different parameter values to
        understand model robustness and optimal hyperparameters.
        
        Supports both sequential and parallel execution. Parallel execution
        uses ProcessPoolExecutor to run multiple experiments concurrently.
        
        Args:
            param_name: Name of parameter to vary (must be in config)
            param_values: List of values to test
            model_factory: Function that creates model from config
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers (None = CPU count)
            
        Returns:
            Dictionary with sensitivity analysis results including:
                - param_name: Parameter being varied
                - param_values: List of tested values
                - experiments: Results for each parameter value
                - aggregated: Aggregated statistics and best value
        
        Example:
            >>> runner = ExperimentRunner(config)
            >>> results = runner.run_sensitivity_analysis(
            ...     'learning_rate',
            ...     [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            ...     model_factory,
            ...     train_loader,
            ...     val_loader,
            ...     parallel=True,
            ...     max_workers=4
            ... )
        """
        self.logger.log(
            f'Starting sensitivity analysis for {param_name} '
            f'with {len(param_values)} values'
        )
        
        if parallel:
            self.logger.log(f'Running experiments in parallel with max_workers={max_workers}')
            return self._run_sensitivity_parallel(
                param_name,
                param_values,
                model_factory,
                train_loader,
                val_loader,
                test_loader,
                max_workers
            )
        else:
            self.logger.log('Running experiments sequentially')
            return self._run_sensitivity_sequential(
                param_name,
                param_values,
                model_factory,
                train_loader,
                val_loader,
                test_loader
            )
    
    def _run_sensitivity_sequential(
        self,
        param_name: str,
        param_values: List[Any],
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader]
    ) -> Dict[str, Any]:
        """Run sensitivity analysis sequentially.
        
        Args:
            param_name: Parameter name
            param_values: Parameter values to test
            model_factory: Model factory function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {
            'param_name': param_name,
            'param_values': param_values,
            'experiments': {}
        }
        
        # Run experiment for each parameter value
        for param_value in param_values:
            self.logger.log(f'Testing {param_name}={param_value}')
            
            result = self._run_single_sensitivity_experiment(
                param_name,
                param_value,
                model_factory,
                train_loader,
                val_loader,
                test_loader
            )
            
            sensitivity_results['experiments'][str(param_value)] = result
            
            if 'error' not in result:
                self.logger.log(f'Completed {param_name}={param_value}')
            else:
                self.logger.log(
                    f'Experiment with {param_name}={param_value} failed: {result["error"]}',
                    level='error'
                )
        
        # Aggregate results
        aggregated_results = self._aggregate_sensitivity_results(sensitivity_results)
        sensitivity_results['aggregated'] = aggregated_results
        
        # Save results
        self.save_results(sensitivity_results, f'sensitivity_{param_name}_results.json')
        self.save_results(aggregated_results, f'sensitivity_{param_name}_summary.json')
        
        self.logger.log(f'Sensitivity analysis for {param_name} completed')
        self._log_sensitivity_summary(param_name, aggregated_results)
        
        return sensitivity_results
    
    def _run_sensitivity_parallel(
        self,
        param_name: str,
        param_values: List[Any],
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        max_workers: Optional[int]
    ) -> Dict[str, Any]:
        """Run sensitivity analysis in parallel using ProcessPoolExecutor.
        
        Args:
            param_name: Parameter name
            param_values: Parameter values to test
            model_factory: Model factory function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            max_workers: Maximum number of parallel workers
            
        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {
            'param_name': param_name,
            'param_values': param_values,
            'experiments': {}
        }
        
        # Prepare experiment configurations
        experiment_configs = []
        for param_value in param_values:
            experiment_configs.append({
                'param_name': param_name,
                'param_value': param_value,
                'base_config': self.config.to_dict(),
                'experiment_name': f'{self.experiment_name}_{param_name}_{param_value}'
            })
        
        # Run experiments in parallel
        self.logger.log(f'Submitting {len(experiment_configs)} experiments to parallel executor')
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {}
            for exp_config in experiment_configs:
                future = executor.submit(
                    _run_sensitivity_experiment_worker,
                    exp_config,
                    model_factory,
                    train_loader,
                    val_loader,
                    test_loader
                )
                future_to_config[future] = exp_config
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_config):
                exp_config = future_to_config[future]
                param_value = exp_config['param_value']
                
                try:
                    result = future.result()
                    sensitivity_results['experiments'][str(param_value)] = result
                    
                    completed += 1
                    self.logger.log(
                        f'Completed {param_name}={param_value} '
                        f'({completed}/{len(param_values)})'
                    )
                    
                except Exception as e:
                    self.logger.log(
                        f'Experiment with {param_name}={param_value} failed: {str(e)}',
                        level='error'
                    )
                    sensitivity_results['experiments'][str(param_value)] = {
                        'error': str(e),
                        'config': exp_config
                    }
                    completed += 1
        
        # Aggregate results
        aggregated_results = self._aggregate_sensitivity_results(sensitivity_results)
        sensitivity_results['aggregated'] = aggregated_results
        
        # Save results
        self.save_results(sensitivity_results, f'sensitivity_{param_name}_results.json')
        self.save_results(aggregated_results, f'sensitivity_{param_name}_summary.json')
        
        self.logger.log(f'Sensitivity analysis for {param_name} completed')
        self._log_sensitivity_summary(param_name, aggregated_results)
        
        return sensitivity_results
    
    def _run_single_sensitivity_experiment(
        self,
        param_name: str,
        param_value: Any,
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader]
    ) -> Dict[str, Any]:
        """Run a single sensitivity experiment.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
            model_factory: Model factory function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Experiment results
        """
        try:
            # Create modified config
            config_dict = self.config.to_dict()
            
            # Handle nested parameters (e.g., 'extra_params.beta')
            if '.' in param_name:
                parts = param_name.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = param_value
            else:
                config_dict[param_name] = param_value
            
            sensitivity_config = ExperimentConfig(**config_dict)
            
            # Create model
            model = model_factory(sensitivity_config)
            
            # Create runner for this experiment
            experiment_name = f'{self.experiment_name}_{param_name}_{param_value}'
            runner = ExperimentRunner(sensitivity_config, experiment_name)
            
            # Train
            train_results = runner.train_model(model, train_loader, val_loader)
            
            # Evaluate
            if test_loader is not None:
                test_results = runner.evaluate_model(model, test_loader)
            else:
                test_results = {}
            
            return {
                'train': train_results,
                'test': test_results,
                'config': sensitivity_config.to_dict()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'config': config_dict
            }
    
    def run_multi_param_sensitivity(
        self,
        param_grid: Dict[str, List[Any]],
        model_factory: Callable[[ExperimentConfig], nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run sensitivity analysis for multiple parameters.
        
        Performs independent sensitivity analysis for each parameter to
        understand their individual effects on model performance.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       e.g., {'learning_rate': [1e-5, 1e-4, 1e-3],
                              'embedding_dim': [16, 32, 64]}
            model_factory: Function that creates model from config
            train_loader: Training data loader
            val_loader: Optional validation data loader
            test_loader: Optional test data loader
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping parameter names to their sensitivity results,
            plus a 'summary' key with aggregated statistics across all parameters
        
        Example:
            >>> param_grid = {
            ...     'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            ...     'embedding_dim': [16, 32, 64, 128],
            ...     'num_layers': [4, 6, 8, 10]
            ... }
            >>> results = runner.run_multi_param_sensitivity(
            ...     param_grid,
            ...     model_factory,
            ...     train_loader,
            ...     val_loader,
            ...     parallel=True
            ... )
        """
        self.logger.log(
            f'Starting multi-parameter sensitivity analysis '
            f'for {len(param_grid)} parameters'
        )
        
        all_results = {}
        
        # Run sensitivity analysis for each parameter independently
        for param_name, param_values in param_grid.items():
            self.logger.log(f'\nAnalyzing parameter: {param_name}')
            
            results = self.run_sensitivity_analysis(
                param_name,
                param_values,
                model_factory,
                train_loader,
                val_loader,
                test_loader,
                parallel=parallel,
                max_workers=max_workers
            )
            
            all_results[param_name] = results
        
        # Create summary across all parameters
        summary = self._create_multi_param_summary(all_results)
        all_results['summary'] = summary
        
        # Save combined results
        self.save_results(all_results, 'multi_param_sensitivity_results.json')
        self.save_results(summary, 'multi_param_sensitivity_summary.json')
        
        self.logger.log('Multi-parameter sensitivity analysis completed')
        self._log_multi_param_summary(summary)
        
        return all_results
    
    def _create_multi_param_summary(
        self,
        all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of multi-parameter sensitivity analysis.
        
        Args:
            all_results: Results for all parameters
            
        Returns:
            Summary dictionary with best values and comparisons
        """
        summary = {
            'parameters': {},
            'best_configurations': {},
            'sensitivity_ranking': []
        }
        
        # Extract best value and sensitivity for each parameter
        sensitivities = []
        
        for param_name, results in all_results.items():
            if 'aggregated' not in results:
                continue
            
            aggregated = results['aggregated']
            
            param_summary = {
                'best_value': aggregated.get('best_value'),
                'best_metric': aggregated.get('best_metric'),
                'worst_value': aggregated.get('worst_value'),
                'worst_metric': aggregated.get('worst_metric'),
                'statistics': aggregated.get('statistics', {})
            }
            
            summary['parameters'][param_name] = param_summary
            
            # Compute sensitivity (range of validation loss)
            if 'val_loss' in aggregated.get('statistics', {}):
                val_loss_stats = aggregated['statistics']['val_loss']
                sensitivity = val_loss_stats['max'] - val_loss_stats['min']
                sensitivities.append((param_name, sensitivity))
        
        # Rank parameters by sensitivity
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        summary['sensitivity_ranking'] = [
            {'parameter': name, 'sensitivity': sens}
            for name, sens in sensitivities
        ]
        
        # Create best configuration
        best_config = {}
        for param_name, param_summary in summary['parameters'].items():
            if param_summary['best_value'] is not None:
                best_config[param_name] = param_summary['best_value']
        
        summary['best_configurations']['individual_best'] = best_config
        
        return summary
    
    def _log_multi_param_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary of multi-parameter sensitivity analysis.
        
        Args:
            summary: Multi-parameter summary
        """
        self.logger.log('\nMulti-Parameter Sensitivity Analysis Summary:')
        self.logger.log('=' * 60)
        
        # Log sensitivity ranking
        self.logger.log('\nParameter Sensitivity Ranking (by validation loss range):')
        for i, item in enumerate(summary['sensitivity_ranking'], 1):
            self.logger.log(
                f'{i}. {item["parameter"]}: '
                f'sensitivity={item["sensitivity"]:.4f}'
            )
        
        # Log best values for each parameter
        self.logger.log('\nBest Values for Each Parameter:')
        for param_name, param_summary in summary['parameters'].items():
            if param_summary['best_value'] is not None:
                self.logger.log(
                    f'{param_name}={param_summary["best_value"]} '
                    f'(val_loss={param_summary["best_metric"]:.4f})'
                )
        
        self.logger.log('=' * 60)
    
    def _aggregate_sensitivity_results(
        self,
        sensitivity_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate sensitivity analysis results with comprehensive statistics.
        
        Computes statistics across all parameter values including:
        - Mean, std, min, max for each metric
        - Best parameter value based on validation loss
        - Convergence information (epochs to convergence)
        - Training time statistics
        
        Args:
            sensitivity_results: Raw sensitivity results
            
        Returns:
            Aggregated results with statistics and DataFrame
        """
        param_name = sensitivity_results['param_name']
        param_values = sensitivity_results['param_values']
        
        aggregated = {
            'param_name': param_name,
            'param_values': param_values,
            'metrics': {
                'train_loss': [],
                'val_loss': [],
                'training_time': [],
                'total_epochs': [],
                'test_metrics': {}
            },
            'statistics': {},
            'best_value': None,
            'best_metric': None,
            'worst_value': None,
            'worst_metric': None
        }
        
        # Extract metrics for each parameter value
        for param_value in param_values:
            param_str = str(param_value)
            
            if param_str not in sensitivity_results['experiments']:
                aggregated['metrics']['train_loss'].append(None)
                aggregated['metrics']['val_loss'].append(None)
                aggregated['metrics']['training_time'].append(None)
                aggregated['metrics']['total_epochs'].append(None)
                continue
            
            experiment = sensitivity_results['experiments'][param_str]
            
            if 'error' in experiment:
                aggregated['metrics']['train_loss'].append(None)
                aggregated['metrics']['val_loss'].append(None)
                aggregated['metrics']['training_time'].append(None)
                aggregated['metrics']['total_epochs'].append(None)
                continue
            
            # Extract training metrics
            if 'train' in experiment and 'final_metrics' in experiment['train']:
                metrics = experiment['train']['final_metrics']
                aggregated['metrics']['train_loss'].append(metrics.get('train_loss'))
                aggregated['metrics']['val_loss'].append(metrics.get('val_loss'))
            else:
                aggregated['metrics']['train_loss'].append(None)
                aggregated['metrics']['val_loss'].append(None)
            
            # Extract training time and epochs
            if 'train' in experiment:
                train_info = experiment['train']
                if 'history' in train_info and 'epoch_time' in train_info['history']:
                    total_time = sum(train_info['history']['epoch_time'])
                    aggregated['metrics']['training_time'].append(total_time)
                else:
                    aggregated['metrics']['training_time'].append(None)
                
                aggregated['metrics']['total_epochs'].append(
                    train_info.get('total_epochs', None)
                )
            else:
                aggregated['metrics']['training_time'].append(None)
                aggregated['metrics']['total_epochs'].append(None)
            
            # Extract test metrics
            if 'test' in experiment and 'test_metrics' in experiment['test']:
                test_metrics = experiment['test']['test_metrics']
                for metric_name, metric_value in test_metrics.items():
                    if metric_name not in aggregated['metrics']['test_metrics']:
                        aggregated['metrics']['test_metrics'][metric_name] = []
                    aggregated['metrics']['test_metrics'][metric_name].append(metric_value)
        
        # Compute statistics for each metric
        for metric_name, metric_values in aggregated['metrics'].items():
            if metric_name == 'test_metrics':
                continue
            
            valid_values = [v for v in metric_values if v is not None]
            if valid_values:
                aggregated['statistics'][metric_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'median': float(np.median(valid_values))
                }
        
        # Find best and worst parameter values (based on validation loss)
        val_losses = aggregated['metrics']['val_loss']
        valid_indices = [i for i, v in enumerate(val_losses) if v is not None]
        
        if valid_indices:
            valid_losses = [val_losses[i] for i in valid_indices]
            valid_params = [param_values[i] for i in valid_indices]
            
            best_idx = np.argmin(valid_losses)
            worst_idx = np.argmax(valid_losses)
            
            aggregated['best_value'] = valid_params[best_idx]
            aggregated['best_metric'] = valid_losses[best_idx]
            aggregated['worst_value'] = valid_params[worst_idx]
            aggregated['worst_metric'] = valid_losses[worst_idx]
        
        # Create DataFrame for easy analysis
        aggregated['dataframe'] = self._create_sensitivity_dataframe(
            param_name, param_values, aggregated['metrics']
        )
        
        return aggregated
    
    def _create_sensitivity_dataframe(
        self,
        param_name: str,
        param_values: List[Any],
        metrics: Dict[str, List]
    ) -> pd.DataFrame:
        """Create pandas DataFrame from sensitivity results.
        
        Args:
            param_name: Parameter name
            param_values: Parameter values
            metrics: Dictionary of metric lists
            
        Returns:
            DataFrame with one row per parameter value
        """
        data = {param_name: param_values}
        
        # Add basic metrics
        for metric_name in ['train_loss', 'val_loss', 'training_time', 'total_epochs']:
            if metric_name in metrics:
                data[metric_name] = metrics[metric_name]
        
        # Add test metrics
        if 'test_metrics' in metrics:
            for test_metric_name, test_metric_values in metrics['test_metrics'].items():
                data[f'test_{test_metric_name}'] = test_metric_values
        
        df = pd.DataFrame(data)
        
        # Sort by parameter value
        try:
            df = df.sort_values(by=param_name)
        except:
            pass  # Skip sorting if parameter values are not comparable
        
        return df
    
    def _log_sensitivity_summary(
        self,
        param_name: str,
        aggregated_results: Dict[str, Any]
    ) -> None:
        """Log summary of sensitivity analysis results.
        
        Args:
            param_name: Parameter name
            aggregated_results: Aggregated sensitivity results
        """
        self.logger.log(f'Sensitivity Analysis Summary for {param_name}:')
        self.logger.log('=' * 60)
        
        param_values = aggregated_results['param_values']
        val_losses = aggregated_results['metrics']['val_loss']
        
        # Log results for each value
        for param_value, val_loss in zip(param_values, val_losses):
            if val_loss is not None:
                self.logger.log(f'{param_name}={param_value}: val_loss={val_loss:.4f}')
            else:
                self.logger.log(f'{param_name}={param_value}: FAILED')
        
        # Log best value
        if aggregated_results['best_value'] is not None:
            self.logger.log(
                f'\nBest value: {param_name}={aggregated_results["best_value"]} '
                f'(val_loss={aggregated_results["best_metric"]:.4f})'
            )
        
        self.logger.log('=' * 60)
