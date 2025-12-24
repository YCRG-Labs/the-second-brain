"""Hyperparameter optimization for realistic microbiome generation.

This module implements automated hyperparameter search using Optuna
to find optimal configurations for the RealisticMicrobiomeModel.

The optimization objective combines:
- Minimize MFD (Microbiome Fréchet Distance)
- Maximize KS test p-values for diversity distributions

References:
    Requirements 8.1-8.4 from realistic-microbiome-generation spec
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import warnings

import numpy as np
import torch

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # Type hint fallback

from src.realistic_model import RealisticMicrobiomeModel, RealisticTrainingConfig
from src.training import (
    RealisticMicrobiomeTrainer,
    TrainingResult,
    train_on_dataset,
)
from src.microbiome_datasets import (
    ProcessedMicrobiomeDataset,
    load_dataset,
    create_train_val_split,
)


@dataclass
class HyperparameterSearchSpace:
    """Definition of hyperparameter search space.
    
    Attributes:
        learning_rate: (min, max) for log-uniform sampling
        embedding_dim: List of values to try
        hidden_dim: List of values to try
        lambda_kl: (min, max) for log-uniform sampling
        lambda_sparse: (min, max) for uniform sampling
        lambda_alpha: (min, max) for uniform sampling
        lambda_beta: (min, max) for uniform sampling
        lambda_coex: (min, max) for uniform sampling
        batch_size: List of values to try
    """
    learning_rate: Tuple[float, float] = (1e-5, 1e-3)
    embedding_dim: List[int] = field(default_factory=lambda: [32, 64, 128])
    hidden_dim: List[int] = field(default_factory=lambda: [128, 256, 512])
    lambda_kl: Tuple[float, float] = (0.001, 0.1)
    lambda_sparse: Tuple[float, float] = (0.1, 5.0)
    lambda_alpha: Tuple[float, float] = (0.1, 2.0)
    lambda_beta: Tuple[float, float] = (0.1, 2.0)
    lambda_coex: Tuple[float, float] = (1.0, 10.0)
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization.
    
    Attributes:
        best_params: Best hyperparameter configuration found
        best_value: Best objective value achieved
        best_trial_number: Trial number of best result
        n_trials: Total number of trials run
        study_name: Name of the Optuna study
        all_trials: List of all trial results
    """
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    study_name: str
    all_trials: List[Dict[str, Any]] = field(default_factory=list)


def sample_hyperparameters(
    trial: Trial,
    search_space: HyperparameterSearchSpace
) -> Dict[str, Any]:
    """Sample hyperparameters from search space using Optuna trial.
    
    Args:
        trial: Optuna trial object
        search_space: Search space definition
        
    Returns:
        Dict of sampled hyperparameters
    """
    params = {
        'learning_rate': trial.suggest_float(
            'learning_rate',
            search_space.learning_rate[0],
            search_space.learning_rate[1],
            log=True
        ),
        'embedding_dim': trial.suggest_categorical(
            'embedding_dim',
            search_space.embedding_dim
        ),
        'hidden_dim': trial.suggest_categorical(
            'hidden_dim',
            search_space.hidden_dim
        ),
        'lambda_kl': trial.suggest_float(
            'lambda_kl',
            search_space.lambda_kl[0],
            search_space.lambda_kl[1],
            log=True
        ),
        'lambda_sparse': trial.suggest_float(
            'lambda_sparse',
            search_space.lambda_sparse[0],
            search_space.lambda_sparse[1]
        ),
        'lambda_alpha': trial.suggest_float(
            'lambda_alpha',
            search_space.lambda_alpha[0],
            search_space.lambda_alpha[1]
        ),
        'lambda_beta': trial.suggest_float(
            'lambda_beta',
            search_space.lambda_beta[0],
            search_space.lambda_beta[1]
        ),
        'lambda_coex': trial.suggest_float(
            'lambda_coex',
            search_space.lambda_coex[0],
            search_space.lambda_coex[1]
        ),
        'batch_size': trial.suggest_categorical(
            'batch_size',
            search_space.batch_size
        ),
    }
    return params


def compute_objective(
    val_mfd: float,
    alpha_ks_pvalue: float,
    beta_ks_pvalue: float,
    coex_compliance: float,
    mfd_weight: float = 1.0,
    ks_weight: float = 0.5,
    coex_weight: float = 0.3
) -> float:
    """Compute optimization objective from validation metrics.
    
    The objective combines:
    - MFD (lower is better)
    - KS p-values (higher is better, want > 0.05)
    - Co-exclusion compliance (higher is better)
    
    Args:
        val_mfd: Validation MFD
        alpha_ks_pvalue: KS test p-value for alpha diversity
        beta_ks_pvalue: KS test p-value for beta diversity
        coex_compliance: Co-exclusion compliance score
        mfd_weight: Weight for MFD term
        ks_weight: Weight for KS p-value terms
        coex_weight: Weight for co-exclusion term
        
    Returns:
        Objective value (lower is better)
    """
    # MFD term (lower is better)
    mfd_term = mfd_weight * val_mfd
    
    # KS p-value term (want to maximize, so negate)
    # Clip to avoid extreme values
    alpha_ks = min(alpha_ks_pvalue, 1.0)
    beta_ks = min(beta_ks_pvalue, 1.0)
    ks_term = -ks_weight * (alpha_ks + beta_ks) / 2
    
    # Co-exclusion term (want to maximize, so negate)
    coex_term = -coex_weight * coex_compliance
    
    return mfd_term + ks_term + coex_term



class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna.
    
    Searches over model and training hyperparameters to find
    configurations that minimize MFD and maximize KS p-values.
    """
    
    def __init__(
        self,
        dataset_name: str = 'american_gut',
        search_space: Optional[HyperparameterSearchSpace] = None,
        n_trials: int = 50,
        epochs_per_trial: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize hyperparameter optimizer.
        
        Args:
            dataset_name: Dataset to optimize on
            search_space: Search space definition
            n_trials: Number of optimization trials
            epochs_per_trial: Training epochs per trial
            study_name: Name for Optuna study
            storage: Optuna storage URL (for persistence)
            seed: Random seed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )
        
        self.dataset_name = dataset_name
        self.search_space = search_space or HyperparameterSearchSpace()
        self.n_trials = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.study_name = study_name or f'realistic_microbiome_{dataset_name}'
        self.storage = storage
        self.seed = seed
        
        # Load and prepare data once
        self._prepare_data()
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='minimize',
            load_if_exists=True
        )
    
    def _prepare_data(self) -> None:
        """Load and prepare dataset for optimization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dataset = load_dataset(self.dataset_name)
        
        self.train_data, self.val_data = create_train_val_split(
            self.dataset, val_fraction=0.2, seed=self.seed
        )
        
        self.num_taxa = self.train_data.compositions.shape[1]
    
    def _create_config(self, params: Dict[str, Any]) -> RealisticTrainingConfig:
        """Create training config from sampled parameters."""
        return RealisticTrainingConfig(
            dataset=self.dataset_name,
            num_taxa=self.num_taxa,
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            num_epochs=self.epochs_per_trial,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            early_stopping_patience=10,
            lambda_kl=params['lambda_kl'],
            lambda_sparse=params['lambda_sparse'],
            lambda_alpha=params['lambda_alpha'],
            lambda_beta=params['lambda_beta'],
            lambda_coex=params['lambda_coex'],
            validation_frequency=5
        )
    
    def _objective(self, trial: Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        params = sample_hyperparameters(trial, self.search_space)
        
        # Create config
        config = self._create_config(params)
        
        # Create model
        train_tensor = torch.tensor(
            self.train_data.compositions, dtype=torch.float32
        )
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
        
        # Create trainer
        trainer = RealisticMicrobiomeTrainer(
            model=model,
            config=config,
            checkpoint_dir=f'checkpoints/optuna/{self.study_name}/trial_{trial.number}'
        )
        
        # Prepare data
        trainer.prepare_data(self.train_data, self.val_data)
        
        # Train
        try:
            result = trainer.train(verbose=False)
        except Exception as e:
            # Return bad objective on training failure
            return float('inf')
        
        # Extract metrics
        val_mfd = result.best_val_mfd
        
        # Get KS p-values and compliance from final metrics
        final_metrics = result.final_metrics
        alpha_ks = final_metrics.get('val_alpha_ks_pvalue', 0.0)
        beta_ks = final_metrics.get('val_beta_ks_pvalue', 0.0)
        coex_compliance = final_metrics.get('val_coexclusion_compliance', 0.0)
        
        # Compute objective
        objective = compute_objective(
            val_mfd=val_mfd,
            alpha_ks_pvalue=alpha_ks,
            beta_ks_pvalue=beta_ks,
            coex_compliance=coex_compliance
        )
        
        # Report intermediate values for pruning
        trial.report(objective, self.epochs_per_trial)
        
        return objective
    
    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = True
    ) -> OptimizationResult:
        """Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials (uses default if None)
            timeout: Maximum time in seconds
            callbacks: Optional Optuna callbacks
            show_progress_bar: Whether to show progress
            
        Returns:
            OptimizationResult with best configuration
        """
        if n_trials is None:
            n_trials = self.n_trials
        
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar
        )
        
        # Collect all trial results
        all_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                })
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            best_trial_number=self.study.best_trial.number,
            n_trials=len(self.study.trials),
            study_name=self.study_name,
            all_trials=all_trials
        )
    
    def get_best_config(self) -> RealisticTrainingConfig:
        """Get training config with best hyperparameters.
        
        Returns:
            RealisticTrainingConfig with optimal parameters
        """
        return self._create_config(self.study.best_params)


def run_hyperparameter_search(
    dataset_name: str = 'american_gut',
    n_trials: int = 50,
    epochs_per_trial: int = 50,
    output_dir: str = 'optimization_results',
    seed: int = 42
) -> OptimizationResult:
    """Run hyperparameter search and save results.
    
    Args:
        dataset_name: Dataset to optimize on
        n_trials: Number of optimization trials
        epochs_per_trial: Training epochs per trial
        output_dir: Directory to save results
        seed: Random seed
        
    Returns:
        OptimizationResult with best configuration
    """
    print(f"Starting hyperparameter optimization on {dataset_name}")
    print(f"Running {n_trials} trials with {epochs_per_trial} epochs each")
    
    optimizer = HyperparameterOptimizer(
        dataset_name=dataset_name,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        seed=seed
    )
    
    result = optimizer.optimize()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f'{dataset_name}_optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': result.best_params,
            'best_value': result.best_value,
            'best_trial_number': result.best_trial_number,
            'n_trials': result.n_trials,
            'all_trials': result.all_trials
        }, f, indent=2)
    
    print(f"\nOptimization complete!")
    print(f"Best objective value: {result.best_value:.4f}")
    print(f"Best parameters: {result.best_params}")
    print(f"Results saved to: {results_file}")
    
    return result


def retrain_with_best_params(
    optimization_result: OptimizationResult,
    dataset_name: str = 'american_gut',
    num_epochs: int = 200,
    checkpoint_dir: str = 'checkpoints/best_model'
) -> Tuple[RealisticMicrobiomeModel, TrainingResult]:
    """Retrain model with best hyperparameters.
    
    Args:
        optimization_result: Result from hyperparameter search
        dataset_name: Dataset to train on
        num_epochs: Number of training epochs
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (trained_model, training_result)
    """
    params = optimization_result.best_params
    
    config = RealisticTrainingConfig(
        dataset=dataset_name,
        embedding_dim=params['embedding_dim'],
        hidden_dim=params['hidden_dim'],
        num_epochs=num_epochs,
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        lambda_kl=params['lambda_kl'],
        lambda_sparse=params['lambda_sparse'],
        lambda_alpha=params['lambda_alpha'],
        lambda_beta=params['lambda_beta'],
        lambda_coex=params['lambda_coex']
    )
    
    return train_on_dataset(
        dataset_name=dataset_name,
        config=config,
        checkpoint_dir=checkpoint_dir,
        verbose=True
    )


# Simple grid search fallback when Optuna is not available
def grid_search(
    dataset_name: str = 'american_gut',
    param_grid: Optional[Dict[str, List[Any]]] = None,
    epochs_per_config: int = 50,
    seed: int = 42
) -> OptimizationResult:
    """Simple grid search over hyperparameters.
    
    Fallback when Optuna is not available.
    
    Args:
        dataset_name: Dataset to optimize on
        param_grid: Dict mapping param names to lists of values
        epochs_per_config: Training epochs per configuration
        seed: Random seed
        
    Returns:
        OptimizationResult with best configuration
    """
    if param_grid is None:
        param_grid = {
            'learning_rate': [1e-4, 5e-4],
            'embedding_dim': [64],
            'hidden_dim': [256],
            'lambda_kl': [0.01, 0.05],
            'lambda_sparse': [1.0, 2.0],
            'lambda_alpha': [0.5],
            'lambda_beta': [0.5],
            'lambda_coex': [5.0],
            'batch_size': [64]
        }
    
    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Grid search: {len(combinations)} configurations")
    
    # Load data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = load_dataset(dataset_name)
    
    train_data, val_data = create_train_val_split(dataset, val_fraction=0.2, seed=seed)
    num_taxa = train_data.compositions.shape[1]
    
    best_params = None
    best_value = float('inf')
    best_trial = 0
    all_trials = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"Trial {i+1}/{len(combinations)}: {params}")
        
        config = RealisticTrainingConfig(
            dataset=dataset_name,
            num_taxa=num_taxa,
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            num_epochs=epochs_per_config,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            lambda_kl=params['lambda_kl'],
            lambda_sparse=params['lambda_sparse'],
            lambda_alpha=params['lambda_alpha'],
            lambda_beta=params['lambda_beta'],
            lambda_coex=params['lambda_coex'],
            validation_frequency=5
        )
        
        try:
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
            
            trainer = RealisticMicrobiomeTrainer(model=model, config=config)
            trainer.prepare_data(train_data, val_data)
            result = trainer.train(verbose=False)
            
            objective = result.best_val_mfd
            
            all_trials.append({
                'number': i,
                'params': params,
                'value': objective
            })
            
            if objective < best_value:
                best_value = objective
                best_params = params
                best_trial = i
                print(f"  New best: {objective:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return OptimizationResult(
        best_params=best_params or {},
        best_value=best_value,
        best_trial_number=best_trial,
        n_trials=len(combinations),
        study_name=f'grid_search_{dataset_name}',
        all_trials=all_trials
    )
