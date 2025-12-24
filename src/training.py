"""Training loop for realistic microbiome generation models.

This module implements the training pipeline with:
- Early stopping based on validation MFD
- Metric logging (diversity, sparsity, co-exclusion)
- Best checkpoint saving
- Support for American Gut and HMP datasets

References:
    Requirements 5.3, 5.4, 5.5 from realistic-microbiome-generation spec
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.realistic_model import RealisticMicrobiomeModel, RealisticTrainingConfig
from src.microbiome_datasets import (
    ProcessedMicrobiomeDataset,
    AmericanGutDataset,
    HMPDataset,
    create_train_val_split,
    load_dataset,
)
from src.evaluation import compute_frechet_distance


def compute_mfd(generated: np.ndarray, real: np.ndarray) -> float:
    """Compute simplified MFD without phylogenetic kernel.
    
    Uses raw compositions as features for Fréchet distance computation.
    
    Args:
        generated: Generated compositions (num_gen, num_taxa)
        real: Real compositions (num_real, num_taxa)
        
    Returns:
        Fréchet distance value
    """
    import numpy as np
    
    # Compute statistics
    mu_gen = np.mean(generated, axis=0)
    mu_real = np.mean(real, axis=0)
    
    sigma_gen = np.cov(generated, rowvar=False)
    sigma_real = np.cov(real, rowvar=False)
    
    # Handle 1D case
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[sigma_gen]])
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    
    return compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


@dataclass
class TrainingState:
    """State of training for checkpointing and resumption.
    
    Attributes:
        epoch: Current epoch number
        best_val_loss: Best validation loss seen
        best_val_mfd: Best validation MFD seen
        epochs_without_improvement: Epochs since last improvement
        training_history: List of metrics per epoch
    """
    epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_mfd: float = float('inf')
    epochs_without_improvement: int = 0
    training_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result of training run.
    
    Attributes:
        final_epoch: Last epoch trained
        best_epoch: Epoch with best validation metrics
        best_val_loss: Best validation loss achieved
        best_val_mfd: Best validation MFD achieved
        training_history: Full training history
        final_metrics: Metrics from final epoch
        training_time: Total training time in seconds
    """
    final_epoch: int
    best_epoch: int
    best_val_loss: float
    best_val_mfd: float
    training_history: List[Dict[str, float]]
    final_metrics: Dict[str, float]
    training_time: float


class EarlyStopping:
    """Early stopping handler based on validation metric.
    
    Stops training when validation metric doesn't improve for
    a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss-like metrics, 'max' for accuracy-like
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current validation metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.should_stop = False


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(
        self,
        save_dir: str,
        model_name: str = 'realistic_model',
        keep_best_only: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            model_name: Base name for checkpoint files
            keep_best_only: If True, only keep best checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.keep_best_only = keep_best_only
        self.best_path: Optional[Path] = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainingState,
        config: RealisticTrainingConfig,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            state: Training state
            config: Training configuration
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_state': {
                'epoch': state.epoch,
                'best_val_loss': state.best_val_loss,
                'best_val_mfd': state.best_val_mfd,
                'epochs_without_improvement': state.epochs_without_improvement,
            },
            'config': vars(config),
            'training_history': state.training_history
        }
        
        if is_best:
            path = self.save_dir / f'{self.model_name}_best.pt'
            torch.save(checkpoint, path)
            self.best_path = path
        elif not self.keep_best_only:
            path = self.save_dir / f'{self.model_name}_epoch{state.epoch}.pt'
            torch.save(checkpoint, path)
        else:
            path = self.save_dir / f'{self.model_name}_latest.pt'
            torch.save(checkpoint, path)
        
        return path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[str] = None
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], TrainingState]:
        """Load checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            path: Path to checkpoint (uses best if None)
            
        Returns:
            Tuple of (model, optimizer, training_state)
        """
        if path is None:
            path = self.best_path or self.save_dir / f'{self.model_name}_best.pt'
        
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        ts = checkpoint['training_state']
        state = TrainingState(
            epoch=ts['epoch'],
            best_val_loss=ts['best_val_loss'],
            best_val_mfd=ts['best_val_mfd'],
            epochs_without_improvement=ts['epochs_without_improvement'],
            training_history=checkpoint.get('training_history', [])
        )
        
        return model, optimizer, state



class RealisticMicrobiomeTrainer:
    """Trainer for RealisticMicrobiomeModel.
    
    Handles the full training loop including:
    - Data loading and batching
    - Training and validation steps
    - Early stopping based on validation MFD
    - Metric logging
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: RealisticMicrobiomeModel,
        config: RealisticTrainingConfig,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on (auto-detected if None)
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.config = config
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode='min'
        )
        
        # Checkpoint manager
        if checkpoint_dir is None:
            checkpoint_dir = f'checkpoints/{config.dataset}'
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_dir,
            model_name=f'realistic_{config.dataset}'
        )
        
        # Training state
        self.state = TrainingState()
        
        # Data loaders (set by prepare_data)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None
    
    def prepare_data(
        self,
        train_dataset: ProcessedMicrobiomeDataset,
        val_dataset: Optional[ProcessedMicrobiomeDataset] = None,
        val_fraction: float = 0.2
    ) -> None:
        """Prepare data loaders for training.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (split from train if None)
            val_fraction: Fraction for validation if splitting
        """
        if val_dataset is None:
            train_dataset, val_dataset = create_train_val_split(
                train_dataset, val_fraction=val_fraction
            )
        
        # Convert to tensors
        self.train_data = torch.tensor(
            train_dataset.compositions, dtype=torch.float32
        )
        self.val_data = torch.tensor(
            val_dataset.compositions, dtype=torch.float32
        )
        
        # Create data loaders
        train_ds = TensorDataset(self.train_data)
        val_ds = TensorDataset(self.val_data)
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.
        
        Returns:
            Dict of average training metrics
        """
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'sparsity_loss': 0.0,
            'alpha_diversity_loss': 0.0,
            'beta_diversity_loss': 0.0,
            'coexclusion_loss': 0.0,
            'rare_taxa_loss': 0.0
        }
        num_batches = 0
        
        for batch in self.train_loader:
            x = batch[0].to(self.device)
            
            # Forward pass and loss computation
            self.optimizer.zero_grad()
            losses = self.model.compute_loss(x)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        
        val_losses = {
            'val_total_loss': 0.0,
            'val_reconstruction_loss': 0.0,
            'val_kl_loss': 0.0,
            'val_sparsity_loss': 0.0,
            'val_alpha_diversity_loss': 0.0,
            'val_beta_diversity_loss': 0.0,
            'val_coexclusion_loss': 0.0,
        }
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch[0].to(self.device)
            losses = self.model.compute_loss(x)
            
            for key in val_losses:
                loss_key = key.replace('val_', '')
                if loss_key in losses:
                    val_losses[key] += losses[loss_key].item()
            num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        # Compute MFD on validation set
        generated = self.model.generate(
            num_samples=min(500, len(self.val_data)),
            device=self.device
        )
        
        try:
            mfd = compute_mfd(
                generated.cpu().numpy(),
                self.val_data.numpy()
            )
            val_losses['val_mfd'] = mfd
        except Exception:
            val_losses['val_mfd'] = float('inf')
        
        # Compute additional metrics
        metrics = self.model.compute_metrics(generated, self.val_data.to(self.device))
        val_losses.update({f'val_{k}': v for k, v in metrics.items()})
        
        return val_losses
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        verbose: bool = True,
        log_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ) -> TrainingResult:
        """Run full training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
            verbose: Whether to print progress
            log_callback: Optional callback for logging metrics
            
        Returns:
            TrainingResult with training history and final metrics
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        if self.train_loader is None:
            raise RuntimeError("Call prepare_data() before training")
        
        start_time = time.time()
        best_epoch = 0
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch + 1
            
            # Training step
            train_metrics = self.train_epoch()
            
            # Validation step
            if (epoch + 1) % self.config.validation_frequency == 0:
                val_metrics = self.validate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['val_total_loss'])
                
                # Check for improvement
                val_mfd = val_metrics.get('val_mfd', float('inf'))
                is_best = val_mfd < self.state.best_val_mfd
                
                if is_best:
                    self.state.best_val_mfd = val_mfd
                    self.state.best_val_loss = val_metrics['val_total_loss']
                    self.state.epochs_without_improvement = 0
                    best_epoch = epoch + 1
                    
                    # Save best checkpoint
                    self.checkpoint_manager.save(
                        self.model, self.optimizer, self.state,
                        self.config, is_best=True
                    )
                else:
                    self.state.epochs_without_improvement += 1
                
                # Early stopping check
                if self.early_stopping(val_mfd):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            self.state.training_history.append(all_metrics)
            
            # Logging
            if verbose and (epoch + 1) % self.config.validation_frequency == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Loss: {train_metrics['total_loss']:.4f} - "
                    f"Val Loss: {val_metrics.get('val_total_loss', 0):.4f} - "
                    f"Val MFD: {val_metrics.get('val_mfd', 0):.4f}"
                )
            
            if log_callback is not None:
                log_callback(all_metrics)
        
        training_time = time.time() - start_time
        
        # Load best model
        if self.checkpoint_manager.best_path:
            self.model, _, _ = self.checkpoint_manager.load(self.model)
        
        # Final validation
        final_metrics = self.validate()
        
        return TrainingResult(
            final_epoch=self.state.epoch,
            best_epoch=best_epoch,
            best_val_loss=self.state.best_val_loss,
            best_val_mfd=self.state.best_val_mfd,
            training_history=self.state.training_history,
            final_metrics=final_metrics,
            training_time=training_time
        )



def train_on_dataset(
    dataset_name: str,
    config: Optional[RealisticTrainingConfig] = None,
    data_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    use_real_data: bool = False,
    verbose: bool = True
) -> Tuple[RealisticMicrobiomeModel, TrainingResult]:
    """Train a model on a specified dataset.
    
    Convenience function that handles data loading, model creation,
    and training in one call.
    
    Args:
        dataset_name: Name of dataset ('american_gut' or 'hmp')
        config: Training configuration (uses defaults if None)
        data_path: Path to data file (uses synthetic if None)
        checkpoint_dir: Directory for checkpoints
        use_real_data: If True, download and use real data from public sources
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_result)
        
    Example:
        >>> # Train on synthetic data (fast, for testing)
        >>> model, result = train_on_dataset('american_gut')
        >>> 
        >>> # Train on real data (downloads ~100MB)
        >>> model, result = train_on_dataset('american_gut', use_real_data=True)
    """
    # Default config
    if config is None:
        config = RealisticTrainingConfig(dataset=dataset_name)
    
    # Load dataset
    if verbose:
        data_type = "real" if use_real_data else "synthetic"
        print(f"Loading {dataset_name} dataset ({data_type} data)...")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = load_dataset(
            dataset_name,
            data_dir=data_path,
            use_real_data=use_real_data,
            max_taxa=config.num_taxa,
            min_prevalence=config.min_prevalence
        )
    
    if verbose:
        print(f"Loaded {dataset.stats.num_samples} samples, {dataset.stats.num_taxa} taxa")
        print(f"Mean sparsity: {dataset.stats.mean_sparsity:.3f}")
        print(f"Alpha diversity: {dataset.stats.alpha_diversity_mean:.3f} ± {dataset.stats.alpha_diversity_std:.3f}")
    
    # Split data
    train_data, val_data = create_train_val_split(dataset, val_fraction=0.2, seed=42)
    
    # Create model from real data
    train_tensor = torch.tensor(train_data.compositions, dtype=torch.float32)
    model = RealisticMicrobiomeModel.from_real_data(
        real_data=train_tensor,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        lambda_kl=config.lambda_kl,
        lambda_sparse=config.lambda_sparse,
        lambda_alpha=config.lambda_alpha,
        lambda_beta=config.lambda_beta,
        lambda_coex=config.lambda_coex
    )
    
    if verbose:
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = RealisticMicrobiomeTrainer(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir
    )
    
    # Prepare data
    trainer.prepare_data(train_data, val_data)
    
    # Train
    if verbose:
        print(f"Starting training for {config.num_epochs} epochs...")
    
    result = trainer.train(verbose=verbose)
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"Best epoch: {result.best_epoch}")
        print(f"Best validation MFD: {result.best_val_mfd:.4f}")
        print(f"Training time: {result.training_time:.1f}s")
    
    return model, result


def train_american_gut(
    config: Optional[RealisticTrainingConfig] = None,
    data_path: Optional[str] = None,
    checkpoint_dir: str = 'checkpoints/american_gut',
    verbose: bool = True
) -> Tuple[RealisticMicrobiomeModel, TrainingResult]:
    """Train model on American Gut dataset.
    
    Args:
        config: Training configuration
        data_path: Path to AGP data file
        checkpoint_dir: Directory for checkpoints
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_result)
    """
    if config is None:
        config = RealisticTrainingConfig(dataset='american_gut')
    
    return train_on_dataset(
        dataset_name='american_gut',
        config=config,
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )


def train_hmp(
    config: Optional[RealisticTrainingConfig] = None,
    data_path: Optional[str] = None,
    checkpoint_dir: str = 'checkpoints/hmp',
    verbose: bool = True
) -> Tuple[RealisticMicrobiomeModel, TrainingResult]:
    """Train model on HMP dataset.
    
    Args:
        config: Training configuration
        data_path: Path to HMP data file
        checkpoint_dir: Directory for checkpoints
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, training_result)
    """
    if config is None:
        config = RealisticTrainingConfig(dataset='hmp')
    
    return train_on_dataset(
        dataset_name='hmp',
        config=config,
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )


def save_training_results(
    result: TrainingResult,
    output_path: str
) -> None:
    """Save training results to JSON file.
    
    Args:
        result: Training result to save
        output_path: Path to output JSON file
    """
    output = {
        'final_epoch': result.final_epoch,
        'best_epoch': result.best_epoch,
        'best_val_loss': result.best_val_loss,
        'best_val_mfd': result.best_val_mfd,
        'training_time': result.training_time,
        'final_metrics': result.final_metrics,
        'training_history': result.training_history
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_trained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> RealisticMicrobiomeModel:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model with saved config
    model = RealisticMicrobiomeModel(
        num_taxa=config['num_taxa'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        lambda_kl=config['lambda_kl'],
        lambda_sparse=config['lambda_sparse'],
        lambda_alpha=config['lambda_alpha'],
        lambda_beta=config['lambda_beta'],
        lambda_coex=config['lambda_coex']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
