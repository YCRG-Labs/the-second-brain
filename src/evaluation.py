"""Evaluation metrics for microbiome generation and prediction.

This module implements comprehensive evaluation metrics including:
- Alpha diversity (Shannon entropy)
- Beta diversity (Bray-Curtis dissimilarity)
- Microbiome Fréchet Distance (MFD)
- Prediction accuracy metrics (MAE, Top-K)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from scipy.spatial.distance import braycurtis
from scipy.stats import entropy


def shannon_entropy(composition: np.ndarray) -> float:
    """Compute Shannon entropy (alpha diversity) for a single composition.
    
    Shannon entropy measures the diversity within a single sample:
    H = -sum(p_i * log(p_i)) where p_i are relative abundances.
    
    Args:
        composition: Relative abundance vector of shape (num_taxa,)
                    Must be non-negative and sum to 1.
    
    Returns:
        Shannon entropy value (non-negative)
    
    Raises:
        ValueError: If composition is invalid (negative values or doesn't sum to 1)
    """
    if np.any(composition < 0):
        raise ValueError("Composition must be non-negative")
    
    total = np.sum(composition)
    if not np.isclose(total, 1.0, rtol=1e-5):
        raise ValueError(f"Composition must sum to 1, got {total}")
    
    # Filter out zeros to avoid log(0)
    nonzero = composition[composition > 0]
    
    # Compute Shannon entropy using scipy
    return float(entropy(nonzero, base=np.e))


def alpha_diversity(compositions: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy for multiple compositions.
    
    Args:
        compositions: Array of shape (num_samples, num_taxa)
    
    Returns:
        Array of Shannon entropy values of shape (num_samples,)
    """
    if compositions.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {compositions.shape}")
    
    num_samples = compositions.shape[0]
    diversities = np.zeros(num_samples)
    
    for i in range(num_samples):
        diversities[i] = shannon_entropy(compositions[i])
    
    return diversities


def bray_curtis_dissimilarity(comp1: np.ndarray, comp2: np.ndarray) -> float:
    """Compute Bray-Curtis dissimilarity between two compositions.
    
    Bray-Curtis dissimilarity is defined as:
    BC = sum(|x_i - y_i|) / sum(x_i + y_i)
    
    It ranges from 0 (identical) to 1 (completely dissimilar).
    
    Args:
        comp1: First composition vector of shape (num_taxa,)
        comp2: Second composition vector of shape (num_taxa,)
    
    Returns:
        Bray-Curtis dissimilarity in [0, 1]
    """
    if comp1.shape != comp2.shape:
        raise ValueError(f"Compositions must have same shape: {comp1.shape} vs {comp2.shape}")
    
    # Use scipy's implementation
    return float(braycurtis(comp1, comp2))


def beta_diversity(compositions: np.ndarray) -> np.ndarray:
    """Compute pairwise Bray-Curtis dissimilarity matrix.
    
    Args:
        compositions: Array of shape (num_samples, num_taxa)
    
    Returns:
        Dissimilarity matrix of shape (num_samples, num_samples)
        where entry (i, j) is the Bray-Curtis dissimilarity between
        samples i and j.
    """
    if compositions.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {compositions.shape}")
    
    num_samples = compositions.shape[0]
    dissimilarity_matrix = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = bray_curtis_dissimilarity(compositions[i], compositions[j])
            dissimilarity_matrix[i, j] = dist
            dissimilarity_matrix[j, i] = dist
    
    return dissimilarity_matrix



def extract_phylogenetic_features(
    compositions: np.ndarray,
    phylogenetic_kernel: np.ndarray,
    feature_extractor: Optional[nn.Module] = None
) -> np.ndarray:
    """Extract phylogenetically-weighted features from compositions.
    
    If a feature extractor is provided, it's applied to the compositions.
    Otherwise, we use the phylogenetic kernel to weight the compositions.
    
    Args:
        compositions: Array of shape (num_samples, num_taxa)
        phylogenetic_kernel: Pairwise phylogenetic distance matrix
                            of shape (num_taxa, num_taxa)
        feature_extractor: Optional neural network for feature extraction
    
    Returns:
        Feature array of shape (num_samples, feature_dim)
    
    Raises:
        ValueError: If phylogenetic kernel dimensions don't match compositions
    """
    # Validate input dimensions
    if compositions.ndim != 2:
        raise ValueError(f"Compositions must be 2D array, got shape {compositions.shape}")
    
    num_samples, num_taxa = compositions.shape
    
    # Validate phylogenetic kernel dimensions
    if phylogenetic_kernel.ndim != 2:
        raise ValueError(f"Phylogenetic kernel must be 2D array, got shape {phylogenetic_kernel.shape}")
    
    if phylogenetic_kernel.shape[0] != phylogenetic_kernel.shape[1]:
        raise ValueError(f"Phylogenetic kernel must be square, got shape {phylogenetic_kernel.shape}")
    
    if phylogenetic_kernel.shape[0] != num_taxa:
        raise ValueError(
            f"Phylogenetic kernel size ({phylogenetic_kernel.shape[0]}) must match "
            f"number of taxa ({num_taxa})"
        )
    
    # Validate that phylogenetic kernel is symmetric and non-negative
    if not np.allclose(phylogenetic_kernel, phylogenetic_kernel.T, rtol=1e-5):
        raise ValueError("Phylogenetic kernel must be symmetric")
    
    if np.any(phylogenetic_kernel < 0):
        raise ValueError("Phylogenetic kernel must be non-negative")
    
    if feature_extractor is not None:
        # Use neural feature extractor
        try:
            with torch.no_grad():
                compositions_tensor = torch.from_numpy(compositions).float()
                features = feature_extractor(compositions_tensor)
                return features.cpu().numpy()
        except Exception as e:
            raise ValueError(f"Feature extractor failed: {e}")
    else:
        # Use phylogenetic kernel weighting
        # Weight each composition by phylogenetic relationships
        # This creates a feature space that respects evolutionary distances
        try:
            weighted_features = compositions @ phylogenetic_kernel
            
            # Ensure output has expected shape
            if weighted_features.shape != (num_samples, num_taxa):
                raise ValueError(
                    f"Unexpected output shape: {weighted_features.shape}, "
                    f"expected ({num_samples}, {num_taxa})"
                )
            
            return weighted_features
        except Exception as e:
            raise ValueError(f"Phylogenetic weighting failed: {e}")


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """Compute Fréchet distance between two Gaussian distributions.
    
    The Fréchet distance between two multivariate Gaussians is:
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    
    Args:
        mu1: Mean of first distribution, shape (feature_dim,)
        sigma1: Covariance of first distribution, shape (feature_dim, feature_dim)
        mu2: Mean of second distribution, shape (feature_dim,)
        sigma2: Covariance of second distribution, shape (feature_dim, feature_dim)
        eps: Small constant for numerical stability
    
    Returns:
        Fréchet distance (non-negative)
    """
    # Compute mean difference
    diff = mu1 - mu2
    mean_dist = np.sum(diff ** 2)
    
    # Compute covariance term using matrix square root
    # Add small epsilon to diagonal for numerical stability
    sigma1_stable = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2_stable = sigma2 + eps * np.eye(sigma2.shape[0])
    
    # Compute sqrt(sigma1 @ sigma2) using eigendecomposition
    product = sigma1_stable @ sigma2_stable
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(product)
    
    # Take square root of eigenvalues (clip negative values due to numerical errors)
    eigvals = np.maximum(eigvals, 0)
    sqrt_eigvals = np.sqrt(eigvals)
    
    # Reconstruct sqrt(sigma1 @ sigma2)
    sqrt_product = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    
    # Compute trace term
    cov_dist = np.trace(sigma1_stable + sigma2_stable - 2 * sqrt_product)
    
    # Total Fréchet distance
    frechet_dist = mean_dist + cov_dist
    
    return float(np.maximum(frechet_dist, 0))  # Ensure non-negative


def microbiome_frechet_distance(
    real_compositions: np.ndarray,
    generated_compositions: np.ndarray,
    phylogenetic_kernel: np.ndarray,
    feature_extractor: Optional[nn.Module] = None
) -> float:
    """Compute Microbiome Fréchet Distance (MFD) between real and generated samples.
    
    MFD measures the distance between the distributions of real and generated
    microbiome compositions in a phylogenetically-weighted feature space.
    
    Args:
        real_compositions: Real samples of shape (num_real, num_taxa)
        generated_compositions: Generated samples of shape (num_gen, num_taxa)
        phylogenetic_kernel: Pairwise phylogenetic distances (num_taxa, num_taxa)
        feature_extractor: Optional neural network for feature extraction
    
    Returns:
        MFD value (non-negative)
    
    Raises:
        ValueError: If input arrays have incompatible shapes or insufficient samples
    """
    # Validate input shapes
    if real_compositions.ndim != 2 or generated_compositions.ndim != 2:
        raise ValueError("Compositions must be 2D arrays")
    
    if real_compositions.shape[1] != generated_compositions.shape[1]:
        raise ValueError(
            f"Number of taxa must match: real {real_compositions.shape[1]} vs "
            f"generated {generated_compositions.shape[1]}"
        )
    
    # Check minimum sample requirements for covariance estimation
    # We need at least 2 samples to compute covariance
    if real_compositions.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 real samples for covariance estimation, "
            f"got {real_compositions.shape[0]}"
        )
    
    if generated_compositions.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 generated samples for covariance estimation, "
            f"got {generated_compositions.shape[0]}"
        )
    
    # Extract features
    real_features = extract_phylogenetic_features(
        real_compositions, phylogenetic_kernel, feature_extractor
    )
    gen_features = extract_phylogenetic_features(
        generated_compositions, phylogenetic_kernel, feature_extractor
    )
    
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    
    # Handle single sample case for covariance
    if real_features.shape[0] == 1:
        sigma_real = np.zeros((real_features.shape[1], real_features.shape[1]))
    else:
        sigma_real = np.cov(real_features, rowvar=False)
    
    if gen_features.shape[0] == 1:
        sigma_gen = np.zeros((gen_features.shape[1], gen_features.shape[1]))
    else:
        sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Ensure covariance matrices are 2D
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[sigma_gen]])
    
    # Compute Fréchet distance
    mfd = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return mfd


def abundance_mae(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    per_taxon: bool = False,
    per_horizon: bool = False
) -> np.ndarray:
    """Compute Mean Absolute Error for abundance predictions.
    
    Measures the average absolute difference between predicted and true
    relative abundances. Supports multi-horizon predictions.
    
    Args:
        predictions: Predicted compositions of shape (num_samples, num_taxa)
                    or (num_samples, num_horizons, num_taxa) for multi-horizon
        ground_truth: True compositions of shape (num_samples, num_taxa)
                     or (num_samples, num_horizons, num_taxa) for multi-horizon
        per_taxon: If True, return MAE per taxon; if False, return overall MAE
        per_horizon: If True and inputs are 3D, return MAE per horizon
    
    Returns:
        If per_taxon=False and per_horizon=False: scalar MAE averaged over all
        If per_taxon=True: array of shape (num_taxa,) with MAE per taxon
        If per_horizon=True: array of shape (num_horizons,) with MAE per horizon
        If both True: array of shape (num_horizons, num_taxa)
    
    Raises:
        ValueError: If shapes don't match
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    
    # Compute absolute errors
    abs_errors = np.abs(predictions - ground_truth)
    
    # Handle multi-horizon predictions
    if predictions.ndim == 3:
        num_samples, num_horizons, num_taxa = predictions.shape
        
        if per_horizon and per_taxon:
            # Return MAE per horizon and per taxon: (num_horizons, num_taxa)
            return np.mean(abs_errors, axis=0)
        elif per_horizon:
            # Return MAE per horizon: (num_horizons,)
            return np.mean(abs_errors, axis=(0, 2))
        elif per_taxon:
            # Return MAE per taxon averaged over horizons: (num_taxa,)
            return np.mean(abs_errors, axis=(0, 1))
        else:
            # Return overall MAE
            return np.mean(abs_errors)
    else:
        # Single-horizon predictions
        if per_taxon:
            # Average over samples for each taxon
            return np.mean(abs_errors, axis=0)
        else:
            # Average over all samples and taxa
            return np.mean(abs_errors)


def top_k_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    per_horizon: bool = False
) -> np.ndarray:
    """Compute Top-K accuracy for abundance predictions.
    
    Measures what fraction of the top-K most abundant taxa in the ground truth
    are correctly identified in the top-K predictions. Supports multi-horizon predictions.
    
    Args:
        predictions: Predicted compositions of shape (num_samples, num_taxa)
                    or (num_samples, num_horizons, num_taxa) for multi-horizon
        ground_truth: True compositions of shape (num_samples, num_taxa)
                     or (num_samples, num_horizons, num_taxa) for multi-horizon
        k: Number of top taxa to consider
        per_horizon: If True and inputs are 3D, return accuracy per horizon
    
    Returns:
        If per_horizon=False: scalar accuracy averaged over all samples (and horizons)
        If per_horizon=True: array of shape (num_horizons,) with accuracy per horizon
    
    Raises:
        ValueError: If shapes don't match or k is invalid
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    
    # Handle multi-horizon predictions
    if predictions.ndim == 3:
        num_samples, num_horizons, num_taxa = predictions.shape
        
        if k <= 0 or k > num_taxa:
            raise ValueError(f"k must be in [1, {num_taxa}], got {k}")
        
        if per_horizon:
            # Compute accuracy per horizon
            horizon_accuracies = []
            
            for h in range(num_horizons):
                accuracies = []
                
                for i in range(num_samples):
                    # Get indices of top-K taxa in ground truth for this horizon
                    true_top_k = np.argsort(ground_truth[i, h])[-k:]
                    
                    # Get indices of top-K taxa in predictions for this horizon
                    pred_top_k = np.argsort(predictions[i, h])[-k:]
                    
                    # Compute overlap
                    overlap = len(set(true_top_k) & set(pred_top_k))
                    
                    # Accuracy is fraction of overlap
                    accuracy = overlap / k
                    accuracies.append(accuracy)
                
                horizon_accuracies.append(np.mean(accuracies))
            
            return np.array(horizon_accuracies)
        else:
            # Compute overall accuracy across all horizons
            all_accuracies = []
            
            for h in range(num_horizons):
                for i in range(num_samples):
                    # Get indices of top-K taxa in ground truth for this horizon
                    true_top_k = np.argsort(ground_truth[i, h])[-k:]
                    
                    # Get indices of top-K taxa in predictions for this horizon
                    pred_top_k = np.argsort(predictions[i, h])[-k:]
                    
                    # Compute overlap
                    overlap = len(set(true_top_k) & set(pred_top_k))
                    
                    # Accuracy is fraction of overlap
                    accuracy = overlap / k
                    all_accuracies.append(accuracy)
            
            return float(np.mean(all_accuracies))
    else:
        # Single-horizon predictions
        num_samples, num_taxa = predictions.shape
        
        if k <= 0 or k > num_taxa:
            raise ValueError(f"k must be in [1, {num_taxa}], got {k}")
        
        accuracies = []
        
        for i in range(num_samples):
            # Get indices of top-K taxa in ground truth
            true_top_k = np.argsort(ground_truth[i])[-k:]
            
            # Get indices of top-K taxa in predictions
            pred_top_k = np.argsort(predictions[i])[-k:]
            
            # Compute overlap
            overlap = len(set(true_top_k) & set(pred_top_k))
            
            # Accuracy is fraction of overlap
            accuracy = overlap / k
            accuracies.append(accuracy)
        
        return float(np.mean(accuracies))


def prediction_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k_values: Optional[list] = None,
    per_horizon: bool = False
) -> Dict[str, any]:
    """Compute comprehensive prediction metrics.
    
    Supports both single-horizon and multi-horizon predictions.
    
    Args:
        predictions: Predicted compositions of shape (num_samples, num_taxa)
                    or (num_samples, num_horizons, num_taxa) for multi-horizon
        ground_truth: True compositions of shape (num_samples, num_taxa)
                     or (num_samples, num_horizons, num_taxa) for multi-horizon
        k_values: List of k values for Top-K accuracy (default: [5, 10, 20])
        per_horizon: If True and inputs are 3D, return metrics per horizon
    
    Returns:
        Dictionary containing:
            - 'mae': Overall mean absolute error (or per-horizon if per_horizon=True)
            - 'top_k_acc_{k}': Top-K accuracy for each k value (or per-horizon)
            - 'num_horizons': Number of horizons (if multi-horizon)
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    metrics = {}
    
    # Determine number of taxa (last dimension)
    num_taxa = predictions.shape[-1]
    
    # Check if multi-horizon
    is_multi_horizon = predictions.ndim == 3
    
    if is_multi_horizon:
        metrics['num_horizons'] = predictions.shape[1]
    
    # Compute MAE
    mae_result = abundance_mae(predictions, ground_truth, per_horizon=per_horizon)
    
    if per_horizon and is_multi_horizon:
        # Store per-horizon MAE
        for h in range(predictions.shape[1]):
            metrics[f'mae_horizon_{h}'] = float(mae_result[h])
        metrics['mae_mean'] = float(np.mean(mae_result))
        metrics['mae_std'] = float(np.std(mae_result))
    else:
        metrics['mae'] = float(mae_result)
    
    # Compute Top-K accuracy for each k
    for k in k_values:
        if k <= num_taxa:
            topk_result = top_k_accuracy(predictions, ground_truth, k, per_horizon=per_horizon)
            
            if per_horizon and is_multi_horizon:
                # Store per-horizon Top-K accuracy
                for h in range(predictions.shape[1]):
                    metrics[f'top_{k}_acc_horizon_{h}'] = float(topk_result[h])
                metrics[f'top_{k}_acc_mean'] = float(np.mean(topk_result))
                metrics[f'top_{k}_acc_std'] = float(np.std(topk_result))
            else:
                metrics[f'top_{k}_acc'] = float(topk_result)
    
    return metrics


class MicrobiomeEvaluator:
    """Unified interface for comprehensive microbiome evaluation.
    
    This class combines all evaluation metrics into a single interface,
    supporting both generation quality assessment and prediction accuracy.
    
    Attributes:
        phylogenetic_kernel: Pairwise phylogenetic distance matrix
        feature_extractor: Optional neural network for feature extraction
        k_values: List of k values for Top-K accuracy
        biological_validator: Optional BiologicalValidator for biological constraint checking
        method_comparator: Optional MethodComparator for statistical significance testing
    """
    
    def __init__(
        self,
        phylogenetic_kernel: np.ndarray,
        feature_extractor: Optional[nn.Module] = None,
        k_values: Optional[list] = None,
        biological_validator: Optional['BiologicalValidator'] = None,
        method_comparator: Optional['MethodComparator'] = None
    ):
        """Initialize evaluator with phylogenetic information.
        
        Args:
            phylogenetic_kernel: Pairwise phylogenetic distances (num_taxa, num_taxa)
            feature_extractor: Optional neural network for feature extraction
            k_values: List of k values for Top-K accuracy (default: [5, 10, 20])
            biological_validator: Optional BiologicalValidator for biological validation
            method_comparator: Optional MethodComparator for statistical significance testing
        """
        self.phylogenetic_kernel = phylogenetic_kernel
        self.feature_extractor = feature_extractor
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        self.biological_validator = biological_validator
        self.method_comparator = method_comparator
    
    def evaluate_generation(
        self,
        real_compositions: np.ndarray,
        generated_compositions: np.ndarray,
        include_biological_validation: bool = True
    ) -> Dict[str, float]:
        """Evaluate generation quality with comprehensive metrics.
        
        Computes:
        - Microbiome Fréchet Distance (MFD)
        - Alpha diversity statistics (mean, std)
        - Beta diversity statistics (mean, std)
        - Biological validation (if validator is configured)
        
        Args:
            real_compositions: Real samples of shape (num_real, num_taxa)
            generated_compositions: Generated samples of shape (num_gen, num_taxa)
            include_biological_validation: Whether to run biological validation
        
        Returns:
            Dictionary containing all generation quality metrics
        """
        metrics = {}
        
        # Compute MFD
        metrics['mfd'] = microbiome_frechet_distance(
            real_compositions,
            generated_compositions,
            self.phylogenetic_kernel,
            self.feature_extractor
        )
        
        # Compute alpha diversity
        real_alpha = alpha_diversity(real_compositions)
        gen_alpha = alpha_diversity(generated_compositions)
        
        metrics['alpha_diversity_real_mean'] = float(np.mean(real_alpha))
        metrics['alpha_diversity_real_std'] = float(np.std(real_alpha))
        metrics['alpha_diversity_gen_mean'] = float(np.mean(gen_alpha))
        metrics['alpha_diversity_gen_std'] = float(np.std(gen_alpha))
        
        # Compute beta diversity
        real_beta = beta_diversity(real_compositions)
        gen_beta = beta_diversity(generated_compositions)
        
        # Get upper triangle (excluding diagonal) for statistics
        real_beta_values = real_beta[np.triu_indices_from(real_beta, k=1)]
        gen_beta_values = gen_beta[np.triu_indices_from(gen_beta, k=1)]
        
        metrics['beta_diversity_real_mean'] = float(np.mean(real_beta_values))
        metrics['beta_diversity_real_std'] = float(np.std(real_beta_values))
        metrics['beta_diversity_gen_mean'] = float(np.mean(gen_beta_values))
        metrics['beta_diversity_gen_std'] = float(np.std(gen_beta_values))
        
        # Run biological validation if validator is configured
        if include_biological_validation and self.biological_validator is not None:
            try:
                biological_results = self.biological_validator.validate_all(
                    generated_compositions
                )
                metrics['biological_validation'] = biological_results
            except Exception as e:
                # Log error but don't fail the entire evaluation
                metrics['biological_validation'] = {
                    'error': str(e),
                    'overall_biological_plausibility': 0.0
                }
        
        return metrics
    
    def evaluate_prediction(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate prediction accuracy with comprehensive metrics.
        
        Computes:
        - Mean Absolute Error (MAE)
        - Top-K accuracy for multiple k values
        
        Args:
            predictions: Predicted compositions of shape (num_samples, num_taxa)
            ground_truth: True compositions of shape (num_samples, num_taxa)
        
        Returns:
            Dictionary containing all prediction accuracy metrics
        """
        return prediction_metrics(predictions, ground_truth, self.k_values)
    
    def evaluate_all(
        self,
        real_compositions: np.ndarray,
        generated_compositions: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        include_biological_validation: bool = True
    ) -> Dict[str, float]:
        """Evaluate both generation and prediction (if provided).
        
        Args:
            real_compositions: Real samples for generation evaluation
            generated_compositions: Generated samples for generation evaluation
            predictions: Optional predicted compositions for prediction evaluation
            ground_truth: Optional true compositions for prediction evaluation
            include_biological_validation: Whether to run biological validation
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Generation metrics
        gen_metrics = self.evaluate_generation(
            real_compositions, 
            generated_compositions,
            include_biological_validation=include_biological_validation
        )
        metrics.update(gen_metrics)
        
        # Prediction metrics (if provided)
        if predictions is not None and ground_truth is not None:
            pred_metrics = self.evaluate_prediction(predictions, ground_truth)
            metrics.update(pred_metrics)
        
        return metrics
    
    def compare_methods(
        self,
        method_results: Dict[str, Dict[str, any]],
        metric_names: Optional[list] = None,
        include_statistical_tests: bool = True
    ) -> Dict[str, Dict[str, any]]:
        """Compare multiple methods on specified metrics with statistical testing.
        
        Args:
            method_results: Dictionary mapping method names to their metric dictionaries.
                           Values can be scalars (single run) or arrays (multiple runs).
            metric_names: List of metric names to compare (default: all common metrics)
            include_statistical_tests: Whether to perform statistical significance testing
        
        Returns:
            Dictionary with comparison statistics for each metric, including:
            - Basic statistics (mean, std, best method)
            - Statistical test results (if include_statistical_tests=True and method_comparator is set)
            - Effect sizes and confidence intervals (if available)
        """
        if not method_results:
            return {}
        
        # Get all common metrics if not specified
        if metric_names is None:
            all_metrics = set()
            for metrics in method_results.values():
                all_metrics.update(metrics.keys())
            metric_names = sorted(all_metrics)
        
        comparison = {}
        
        for metric in metric_names:
            # Collect values for this metric across methods
            values = {}
            for method_name, metrics in method_results.items():
                if metric in metrics:
                    values[method_name] = metrics[metric]
            
            if not values:
                continue
            
            # Basic comparison statistics
            metric_comparison = {
                'values': values,
                'mean': {name: float(np.mean(val)) if hasattr(val, '__iter__') and not isinstance(val, str) 
                        else float(val) for name, val in values.items()},
                'std': {name: float(np.std(val)) if hasattr(val, '__iter__') and not isinstance(val, str) and len(np.atleast_1d(val)) > 1
                       else 0.0 for name, val in values.items()}
            }
            
            # Determine best method
            means = metric_comparison['mean']
            is_error_metric = any(
                term in metric.lower()
                for term in ['mae', 'mse', 'error', 'loss', 'mfd', 'violation', 'distance']
            )
            
            if is_error_metric:
                best_method = min(means.items(), key=lambda x: x[1])[0]
            else:
                best_method = max(means.items(), key=lambda x: x[1])[0]
            
            metric_comparison['best_method'] = best_method
            
            # Add statistical testing if requested and comparator is available
            if (include_statistical_tests and 
                self.method_comparator is not None and 
                len(values) >= 2):
                
                # Check if we have multiple runs (arrays) for statistical testing
                array_values = {}
                for name, val in values.items():
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        array_values[name] = np.asarray(val)
                    else:
                        # Convert single values to arrays for consistency
                        array_values[name] = np.array([val])
                
                # Only perform statistical tests if we have arrays with multiple values
                if any(len(arr) > 1 for arr in array_values.values()):
                    try:
                        statistical_results = self.method_comparator.compare_methods(
                            array_values, metric
                        )
                        
                        # Add statistical test results to comparison
                        if 'pairwise_tests' in statistical_results:
                            metric_comparison['pairwise_tests'] = statistical_results['pairwise_tests']
                        
                        if 'effect_sizes' in statistical_results:
                            metric_comparison['effect_sizes'] = statistical_results['effect_sizes']
                        
                        if 'confidence_intervals' in statistical_results:
                            metric_comparison['confidence_intervals'] = statistical_results['confidence_intervals']
                        
                        if 'ranking' in statistical_results:
                            metric_comparison['ranking'] = statistical_results['ranking']
                        
                    except Exception as e:
                        # Log error but don't fail the comparison
                        metric_comparison['statistical_test_error'] = str(e)
            
            comparison[metric] = metric_comparison
        
        return comparison
    
    def evaluate_multiple_methods(
        self,
        method_data: Dict[str, Dict[str, any]],
        include_statistical_tests: bool = True
    ) -> Dict[str, any]:
        """Evaluate and compare multiple methods with comprehensive statistics.
        
        This method provides a complete evaluation pipeline for comparing multiple
        methods, including basic statistics, statistical significance testing,
        effect sizes, and confidence intervals.
        
        Args:
            method_data: Dictionary mapping method names to their evaluation data.
                        Each method's data should contain:
                        - 'real_compositions': Real microbiome samples
                        - 'generated_compositions': Generated samples (for generation methods)
                        - 'predictions': Predicted compositions (for prediction methods)
                        - 'ground_truth': True compositions (for prediction methods)
                        - 'results': Pre-computed results (optional, as arrays for multiple runs)
            include_statistical_tests: Whether to perform statistical significance testing
        
        Returns:
            Dictionary containing:
            - 'method_results': Individual results for each method
            - 'comparison': Statistical comparison across methods
            - 'summary': Overall summary statistics
        """
        method_results = {}
        
        # Evaluate each method
        for method_name, data in method_data.items():
            if 'results' in data:
                # Use pre-computed results
                method_results[method_name] = data['results']
            else:
                # Compute evaluation metrics
                real_comps = data.get('real_compositions')
                gen_comps = data.get('generated_compositions')
                predictions = data.get('predictions')
                ground_truth = data.get('ground_truth')
                
                if real_comps is not None and gen_comps is not None:
                    # Generation evaluation
                    results = self.evaluate_generation(
                        real_comps, gen_comps,
                        include_biological_validation=True
                    )
                    
                    # Add prediction evaluation if available
                    if predictions is not None and ground_truth is not None:
                        pred_results = self.evaluate_prediction(predictions, ground_truth)
                        results.update(pred_results)
                    
                    method_results[method_name] = results
                elif predictions is not None and ground_truth is not None:
                    # Prediction-only evaluation
                    results = self.evaluate_prediction(predictions, ground_truth)
                    method_results[method_name] = results
                else:
                    raise ValueError(f"Insufficient data for method {method_name}")
        
        # Compare methods
        comparison = self.compare_methods(
            method_results,
            include_statistical_tests=include_statistical_tests
        )
        
        # Generate summary statistics
        summary = self._generate_evaluation_summary(method_results, comparison)
        
        return {
            'method_results': method_results,
            'comparison': comparison,
            'summary': summary
        }
    
    def _generate_evaluation_summary(
        self,
        method_results: Dict[str, Dict[str, any]],
        comparison: Dict[str, Dict[str, any]]
    ) -> Dict[str, any]:
        """Generate summary statistics for method evaluation.
        
        Args:
            method_results: Individual results for each method
            comparison: Statistical comparison results
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_methods': len(method_results),
            'metrics_evaluated': list(comparison.keys()),
            'best_methods': {},
            'significant_differences': {}
        }
        
        # Identify best method for each metric
        for metric_name, metric_comparison in comparison.items():
            if 'best_method' in metric_comparison:
                summary['best_methods'][metric_name] = metric_comparison['best_method']
        
        # Count significant differences
        for metric_name, metric_comparison in comparison.items():
            if 'pairwise_tests' in metric_comparison:
                significant_pairs = [
                    pair for pair, test_result in metric_comparison['pairwise_tests'].items()
                    if test_result.get('significant', False)
                ]
                summary['significant_differences'][metric_name] = len(significant_pairs)
        
        return summary



class ComprehensiveEvaluator:
    """Complete evaluation suite for publication with statistical tests.
    
    This class extends MicrobiomeEvaluator with statistical significance testing,
    co-occurrence pattern validation, and comprehensive comparison capabilities.
    
    Attributes:
        real_data: Reference real microbiome compositions
        phylogenetic_kernel: Pairwise phylogenetic distance matrix
        feature_extractor: Optional neural network for feature extraction
        k_values: List of k values for Top-K accuracy
        base_evaluator: Underlying MicrobiomeEvaluator instance
    """
    
    def __init__(
        self,
        real_data: np.ndarray,
        phylogenetic_kernel: np.ndarray,
        feature_extractor: Optional[nn.Module] = None,
        k_values: Optional[list] = None
    ):
        """Initialize comprehensive evaluator with reference data.
        
        Args:
            real_data: Reference real compositions of shape (num_samples, num_taxa)
            phylogenetic_kernel: Pairwise phylogenetic distances (num_taxa, num_taxa)
            feature_extractor: Optional neural network for feature extraction
            k_values: List of k values for Top-K accuracy (default: [5, 10, 20])
        """
        self.real_data = real_data
        self.phylogenetic_kernel = phylogenetic_kernel
        self.feature_extractor = feature_extractor
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        
        # Create base evaluator
        self.base_evaluator = MicrobiomeEvaluator(
            phylogenetic_kernel=phylogenetic_kernel,
            feature_extractor=feature_extractor,
            k_values=k_values
        )
        
        # Precompute reference statistics
        self._real_alpha = alpha_diversity(real_data)
        self._real_beta = beta_diversity(real_data)
        self._real_beta_values = self._real_beta[np.triu_indices_from(self._real_beta, k=1)]
    
    def evaluate_generation(
        self,
        generated_data: np.ndarray,
        compute_mfd: bool = True,
        compute_statistical_tests: bool = True
    ) -> Dict[str, any]:
        """Comprehensive generation quality metrics with statistical tests.
        
        Args:
            generated_data: Generated samples of shape (num_gen, num_taxa)
            compute_mfd: Whether to compute Microbiome Fréchet Distance
            compute_statistical_tests: Whether to run statistical significance tests
        
        Returns:
            Dictionary containing:
                - 'mfd': Microbiome Fréchet Distance (if compute_mfd=True)
                - 'alpha_diversity': Dict with mean, std, and KS test p-value
                - 'beta_diversity': Dict with mean, std, and KS test p-value
                - 'phylogenetic_coherence': Phylogenetic diversity score
                - 'co_occurrence_violations': Number of violated co-occurrence patterns
        """
        from scipy.stats import ks_2samp
        
        metrics = {}
        
        # Compute MFD if requested
        if compute_mfd:
            metrics['mfd'] = microbiome_frechet_distance(
                self.real_data,
                generated_data,
                self.phylogenetic_kernel,
                self.feature_extractor
            )
        
        # Compute alpha diversity
        gen_alpha = alpha_diversity(generated_data)
        
        alpha_metrics = {
            'mean': float(np.mean(gen_alpha)),
            'std': float(np.std(gen_alpha)),
            'real_mean': float(np.mean(self._real_alpha)),
            'real_std': float(np.std(self._real_alpha))
        }
        
        # KS test for alpha diversity
        if compute_statistical_tests:
            ks_stat, ks_pvalue = ks_2samp(self._real_alpha, gen_alpha)
            alpha_metrics['ks_statistic'] = float(ks_stat)
            alpha_metrics['ks_pvalue'] = float(ks_pvalue)
        
        metrics['alpha_diversity'] = alpha_metrics
        
        # Compute beta diversity
        gen_beta = beta_diversity(generated_data)
        gen_beta_values = gen_beta[np.triu_indices_from(gen_beta, k=1)]
        
        beta_metrics = {
            'mean': float(np.mean(gen_beta_values)),
            'std': float(np.std(gen_beta_values)),
            'real_mean': float(np.mean(self._real_beta_values)),
            'real_std': float(np.std(self._real_beta_values))
        }
        
        # KS test for beta diversity
        if compute_statistical_tests:
            ks_stat, ks_pvalue = ks_2samp(self._real_beta_values, gen_beta_values)
            beta_metrics['ks_statistic'] = float(ks_stat)
            beta_metrics['ks_pvalue'] = float(ks_pvalue)
        
        metrics['beta_diversity'] = beta_metrics
        
        # Compute phylogenetic coherence
        metrics['phylogenetic_coherence'] = self._compute_phylogenetic_coherence(
            generated_data
        )
        
        # Check co-occurrence violations
        metrics['co_occurrence_violations'] = self._check_co_occurrence_violations(
            generated_data
        )
        
        return metrics
    
    def _compute_phylogenetic_coherence(self, compositions: np.ndarray) -> float:
        """Compute phylogenetic coherence score.
        
        Measures how well compositions respect phylogenetic relationships.
        Higher scores indicate better coherence with the phylogenetic tree.
        
        Args:
            compositions: Array of shape (num_samples, num_taxa)
        
        Returns:
            Phylogenetic coherence score in [0, 1]
        """
        # Compute phylogenetic diversity for each sample
        # This measures the total branch length covered by present taxa
        phylo_divs = []
        
        for comp in compositions:
            # Taxa with non-zero abundance
            present_taxa = comp > 0
            
            if np.sum(present_taxa) == 0:
                phylo_divs.append(0.0)
                continue
            
            # Compute pairwise distances among present taxa
            present_kernel = self.phylogenetic_kernel[present_taxa][:, present_taxa]
            
            # Phylogenetic diversity is the sum of unique branch lengths
            # Approximated by mean pairwise distance
            if present_kernel.size > 0:
                phylo_div = np.mean(present_kernel)
            else:
                phylo_div = 0.0
            
            phylo_divs.append(phylo_div)
        
        # Normalize by comparing to real data
        real_phylo_divs = []
        for comp in self.real_data:
            present_taxa = comp > 0
            if np.sum(present_taxa) > 0:
                present_kernel = self.phylogenetic_kernel[present_taxa][:, present_taxa]
                if present_kernel.size > 0:
                    real_phylo_divs.append(np.mean(present_kernel))
        
        if len(real_phylo_divs) == 0:
            return 0.0
        
        # Coherence is 1 - normalized difference from real data
        mean_gen = np.mean(phylo_divs)
        mean_real = np.mean(real_phylo_divs)
        
        if mean_real == 0:
            return 1.0 if mean_gen == 0 else 0.0
        
        coherence = 1.0 - min(abs(mean_gen - mean_real) / mean_real, 1.0)
        
        return float(coherence)
    
    def _check_co_occurrence_violations(
        self,
        compositions: np.ndarray,
        threshold: float = 0.01
    ) -> int:
        """Check for violations of co-occurrence patterns.
        
        Identifies samples where taxa that rarely co-occur in real data
        appear together in generated samples.
        
        Args:
            compositions: Array of shape (num_samples, num_taxa)
            threshold: Abundance threshold for considering a taxon "present"
        
        Returns:
            Number of co-occurrence violations detected
        """
        # Compute co-occurrence matrix from real data
        real_presence = (self.real_data > threshold).astype(float)
        real_cooccurrence = real_presence.T @ real_presence / len(self.real_data)
        
        # Identify rare co-occurrences (< 5% of samples)
        rare_cooccurrence_threshold = 0.05
        rare_pairs = np.where(
            (real_cooccurrence < rare_cooccurrence_threshold) &
            (real_cooccurrence > 0)
        )
        
        # Check generated samples for these rare pairs
        gen_presence = (compositions > threshold).astype(float)
        violations = 0
        
        for i, j in zip(rare_pairs[0], rare_pairs[1]):
            if i < j:  # Only check upper triangle
                # Count samples where both taxa are present
                both_present = np.sum((gen_presence[:, i] > 0) & (gen_presence[:, j] > 0))
                
                # If they co-occur more than expected, count as violation
                expected_cooccurrence = real_cooccurrence[i, j] * len(compositions)
                if both_present > expected_cooccurrence * 2:  # 2x threshold
                    violations += 1
        
        return violations
    
    def evaluate_prediction(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        horizons: Optional[list] = None
    ) -> Dict[str, any]:
        """Comprehensive prediction accuracy metrics.
        
        Args:
            predictions: Predicted compositions of shape (num_samples, num_taxa)
                        or (num_samples, num_horizons, num_taxa) for multi-horizon
            ground_truth: True compositions of shape (num_samples, num_taxa)
                         or (num_samples, num_horizons, num_taxa)
            horizons: List of horizon indices to evaluate separately
        
        Returns:
            Dictionary containing prediction metrics, optionally per horizon
        """
        # Handle multi-horizon predictions
        if predictions.ndim == 3 and ground_truth.ndim == 3:
            if horizons is None:
                horizons = list(range(predictions.shape[1]))
            
            metrics = {}
            for h in horizons:
                horizon_metrics = self.base_evaluator.evaluate_prediction(
                    predictions[:, h, :],
                    ground_truth[:, h, :]
                )
                metrics[f'horizon_{h}'] = horizon_metrics
            
            # Compute overall metrics
            overall_metrics = self.base_evaluator.evaluate_prediction(
                predictions.reshape(-1, predictions.shape[-1]),
                ground_truth.reshape(-1, ground_truth.shape[-1])
            )
            metrics['overall'] = overall_metrics
            
            return metrics
        else:
            # Single-horizon prediction
            return self.base_evaluator.evaluate_prediction(predictions, ground_truth)
    
    def compare_methods(
        self,
        results: Dict[str, Dict[str, any]],
        metrics: Optional[list] = None
    ) -> Dict[str, any]:
        """Statistical comparison of multiple methods.
        
        Performs pairwise statistical tests to determine if differences
        between methods are significant.
        
        Args:
            results: Dictionary mapping method names to their metric dictionaries
            metrics: List of metric names to compare (default: all common metrics)
        
        Returns:
            Dictionary with comparison statistics and significance tests
        """
        from scipy.stats import ttest_ind, mannwhitneyu
        
        if not results:
            return {}
        
        # Get all common metrics if not specified
        if metrics is None:
            all_metrics = set()
            for method_metrics in results.values():
                all_metrics.update(self._flatten_metrics(method_metrics).keys())
            metrics = sorted(all_metrics)
        
        comparison = {}
        method_names = list(results.keys())
        
        for metric in metrics:
            # Extract values for this metric from all methods
            values = {}
            for method_name in method_names:
                flat_metrics = self._flatten_metrics(results[method_name])
                if metric in flat_metrics:
                    val = flat_metrics[metric]
                    # Handle both scalar and array values
                    if isinstance(val, (list, np.ndarray)):
                        values[method_name] = np.array(val)
                    else:
                        values[method_name] = val
            
            if len(values) == 0:
                continue
            
            metric_comparison = {
                'values': {k: float(v) if np.isscalar(v) else v.tolist()
                          for k, v in values.items()},
                'mean': {k: float(np.mean(v)) if not np.isscalar(v) else float(v)
                        for k, v in values.items()},
                'std': {k: float(np.std(v)) if not np.isscalar(v) else 0.0
                       for k, v in values.items()}
            }
            
            # Determine best method (lower is better for error metrics)
            is_error_metric = any(
                err in metric.lower()
                for err in ['mae', 'mse', 'error', 'loss', 'mfd', 'violation']
            )
            
            if is_error_metric:
                best_method = min(metric_comparison['mean'].items(), key=lambda x: x[1])[0]
            else:
                best_method = max(metric_comparison['mean'].items(), key=lambda x: x[1])[0]
            
            metric_comparison['best_method'] = best_method
            
            # Perform pairwise statistical tests
            pairwise_tests = {}
            method_list = list(values.keys())
            
            for i, method1 in enumerate(method_list):
                for method2 in method_list[i+1:]:
                    val1 = values[method1]
                    val2 = values[method2]
                    
                    # Only perform tests if we have arrays (multiple samples)
                    if not np.isscalar(val1) and not np.isscalar(val2):
                        # Use Mann-Whitney U test (non-parametric)
                        try:
                            statistic, pvalue = mannwhitneyu(val1, val2, alternative='two-sided')
                            pairwise_tests[f'{method1}_vs_{method2}'] = {
                                'statistic': float(statistic),
                                'pvalue': float(pvalue),
                                'significant': pvalue < 0.05
                            }
                        except Exception:
                            # Fall back to t-test if Mann-Whitney fails
                            try:
                                statistic, pvalue = ttest_ind(val1, val2)
                                pairwise_tests[f'{method1}_vs_{method2}'] = {
                                    'statistic': float(statistic),
                                    'pvalue': float(pvalue),
                                    'significant': pvalue < 0.05
                                }
                            except Exception:
                                pass
            
            if pairwise_tests:
                metric_comparison['pairwise_tests'] = pairwise_tests
            
            comparison[metric] = metric_comparison
        
        return comparison
    
    def _flatten_metrics(self, metrics: Dict[str, any], prefix: str = '') -> Dict[str, any]:
        """Flatten nested metric dictionaries.
        
        Args:
            metrics: Nested dictionary of metrics
            prefix: Prefix for flattened keys
        
        Returns:
            Flattened dictionary with dot-separated keys
        """
        flat = {}
        
        for key, value in metrics.items():
            new_key = f'{prefix}.{key}' if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flat.update(self._flatten_metrics(value, new_key))
            else:
                flat[new_key] = value
        
        return flat
    
    def generate_comparison_plots(
        self,
        results: Dict[str, Dict[str, any]],
        output_dir: str,
        metrics_to_plot: Optional[list] = None
    ) -> None:
        """Generate publication-quality comparison figures.
        
        Creates bar charts with error bars and significance annotations
        comparing multiple methods.
        
        Args:
            results: Dictionary mapping method names to their metric dictionaries
            output_dir: Directory to save plots
            metrics_to_plot: List of metrics to plot (default: all common metrics)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get comparison statistics
        comparison = self.compare_methods(results, metrics_to_plot)
        
        for metric_name, metric_data in comparison.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = list(metric_data['mean'].keys())
            means = [metric_data['mean'][m] for m in methods]
            stds = [metric_data['std'][m] for m in methods]
            
            # Create bar chart
            x_pos = np.arange(len(methods))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            
            # Highlight best method
            best_idx = methods.index(metric_data['best_method'])
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.9)
            
            # Add significance annotations
            if 'pairwise_tests' in metric_data:
                y_max = max(means) + max(stds)
                y_offset = y_max * 0.05
                
                for comparison_key, test_result in metric_data['pairwise_tests'].items():
                    if test_result['significant']:
                        # Parse method names from comparison key
                        m1, m2 = comparison_key.replace('_vs_', ' ').split()
                        if m1 in methods and m2 in methods:
                            idx1 = methods.index(m1)
                            idx2 = methods.index(m2)
                            
                            # Draw significance line
                            y = y_max + y_offset
                            ax.plot([idx1, idx2], [y, y], 'k-', linewidth=1)
                            ax.text((idx1 + idx2) / 2, y, '*', ha='center', va='bottom')
                            y_offset += y_max * 0.05
            
            ax.set_xlabel('Method')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f'{metric_name}_comparison.png', dpi=300)
            plt.close()


class BiologicalValidator:
    """Biological validation metrics for microbiome compositions.
    
    This class implements validation metrics that check whether generated
    microbiome compositions respect known biological constraints and patterns:
    - Co-exclusion patterns (competitive exclusion)
    - Metabolic consistency (pathway completeness)
    - Disease biomarker validation
    
    Attributes:
        co_exclusion_pairs: List of (taxon_i, taxon_j) pairs that rarely co-occur
        metabolic_pathways: Dict mapping pathway names to required taxa
        disease_biomarkers: Dict mapping disease names to biomarker taxa
        taxon_names: Optional list of taxon names for interpretability
    """
    
    def __init__(
        self,
        co_exclusion_pairs: Optional[list] = None,
        metabolic_pathways: Optional[Dict[str, list]] = None,
        disease_biomarkers: Optional[Dict[str, Dict[str, any]]] = None,
        taxon_names: Optional[list] = None
    ):
        """Initialize biological validator with known biological constraints.
        
        Args:
            co_exclusion_pairs: List of (taxon_i, taxon_j) index pairs that
                               exhibit competitive exclusion (rarely co-occur)
            metabolic_pathways: Dict mapping pathway names to lists of taxon
                               indices required for that pathway
            disease_biomarkers: Dict mapping disease names to dicts with:
                               - 'elevated': list of taxa elevated in disease
                               - 'depleted': list of taxa depleted in disease
            taxon_names: Optional list of taxon names for reporting
        """
        self.co_exclusion_pairs = co_exclusion_pairs or []
        self.metabolic_pathways = metabolic_pathways or {}
        self.disease_biomarkers = disease_biomarkers or {}
        self.taxon_names = taxon_names
    
    @classmethod
    def from_reference_data(
        cls,
        reference_compositions: np.ndarray,
        co_exclusion_threshold: float = 0.02,
        taxon_names: Optional[list] = None
    ) -> 'BiologicalValidator':
        """Create validator by learning patterns from reference data.
        
        Automatically identifies co-exclusion pairs from the reference data
        based on co-occurrence frequencies.
        
        Args:
            reference_compositions: Reference compositions (num_samples, num_taxa)
            co_exclusion_threshold: Maximum co-occurrence frequency to consider
                                   as co-exclusion (default: 2%)
            taxon_names: Optional list of taxon names
        
        Returns:
            BiologicalValidator instance with learned patterns
        """
        # Identify co-exclusion pairs from reference data
        presence = (reference_compositions > 0.001).astype(float)
        num_samples = len(reference_compositions)
        
        co_exclusion_pairs = []
        num_taxa = reference_compositions.shape[1]
        
        for i in range(num_taxa):
            for j in range(i + 1, num_taxa):
                # Count samples where both taxa are present
                both_present = np.sum((presence[:, i] > 0) & (presence[:, j] > 0))
                co_occurrence_freq = both_present / num_samples
                
                # If they rarely co-occur, add to co-exclusion pairs
                if co_occurrence_freq < co_exclusion_threshold:
                    # Only add if both taxa are individually common enough
                    freq_i = np.mean(presence[:, i])
                    freq_j = np.mean(presence[:, j])
                    if freq_i > 0.1 and freq_j > 0.1:  # Both present in >10% of samples
                        co_exclusion_pairs.append((i, j))
        
        return cls(
            co_exclusion_pairs=co_exclusion_pairs,
            taxon_names=taxon_names
        )
    
    def validate_co_exclusion(
        self,
        compositions: np.ndarray,
        presence_threshold: float = 0.001
    ) -> Dict[str, any]:
        """Validate that compositions respect co-exclusion patterns.
        
        Checks whether taxa that exhibit competitive exclusion in nature
        are correctly kept separate in the generated compositions.
        
        Args:
            compositions: Compositions to validate (num_samples, num_taxa)
            presence_threshold: Minimum abundance to consider a taxon present
        
        Returns:
            Dictionary containing:
                - 'num_violations': Total number of co-exclusion violations
                - 'violation_rate': Fraction of samples with violations
                - 'violations_per_pair': Dict mapping pair to violation count
                - 'compliance_score': 1 - violation_rate (higher is better)
        """
        if not self.co_exclusion_pairs:
            return {
                'num_violations': 0,
                'violation_rate': 0.0,
                'violations_per_pair': {},
                'compliance_score': 1.0,
                'message': 'No co-exclusion pairs defined'
            }
        
        presence = (compositions > presence_threshold).astype(float)
        num_samples = len(compositions)
        
        violations_per_pair = {}
        total_violations = 0
        samples_with_violations = set()
        
        for i, j in self.co_exclusion_pairs:
            if i >= compositions.shape[1] or j >= compositions.shape[1]:
                continue  # Skip invalid indices
            
            # Find samples where both taxa are present (violation)
            both_present = (presence[:, i] > 0) & (presence[:, j] > 0)
            violation_count = int(np.sum(both_present))
            
            if violation_count > 0:
                pair_key = f'({i}, {j})'
                if self.taxon_names and i < len(self.taxon_names) and j < len(self.taxon_names):
                    pair_key = f'({self.taxon_names[i]}, {self.taxon_names[j]})'
                
                violations_per_pair[pair_key] = violation_count
                total_violations += violation_count
                samples_with_violations.update(np.where(both_present)[0])
        
        violation_rate = len(samples_with_violations) / num_samples if num_samples > 0 else 0.0
        
        return {
            'num_violations': total_violations,
            'violation_rate': float(violation_rate),
            'violations_per_pair': violations_per_pair,
            'compliance_score': float(1.0 - violation_rate),
            'num_pairs_checked': len(self.co_exclusion_pairs)
        }
    
    def validate_metabolic_consistency(
        self,
        compositions: np.ndarray,
        presence_threshold: float = 0.001,
        completeness_threshold: float = 0.5
    ) -> Dict[str, any]:
        """Validate metabolic pathway consistency.
        
        Checks whether compositions have complete metabolic pathways,
        i.e., if some taxa from a pathway are present, most should be present.
        
        Args:
            compositions: Compositions to validate (num_samples, num_taxa)
            presence_threshold: Minimum abundance to consider a taxon present
            completeness_threshold: Minimum fraction of pathway taxa that must
                                   be present for pathway to be considered active
        
        Returns:
            Dictionary containing:
                - 'pathway_completeness': Dict mapping pathway to completeness scores
                - 'mean_completeness': Average completeness across pathways
                - 'incomplete_pathways': List of pathways with low completeness
                - 'consistency_score': Overall metabolic consistency score
        """
        if not self.metabolic_pathways:
            return {
                'pathway_completeness': {},
                'mean_completeness': 1.0,
                'incomplete_pathways': [],
                'consistency_score': 1.0,
                'message': 'No metabolic pathways defined'
            }
        
        presence = (compositions > presence_threshold).astype(float)
        
        pathway_completeness = {}
        incomplete_pathways = []
        
        for pathway_name, pathway_taxa in self.metabolic_pathways.items():
            # Filter valid taxa indices
            valid_taxa = [t for t in pathway_taxa if t < compositions.shape[1]]
            
            if not valid_taxa:
                continue
            
            # Compute completeness for each sample
            sample_completeness = []
            
            for sample_idx in range(len(compositions)):
                taxa_present = sum(presence[sample_idx, t] > 0 for t in valid_taxa)
                
                # Only compute completeness if pathway is partially active
                if taxa_present > 0:
                    completeness = taxa_present / len(valid_taxa)
                    sample_completeness.append(completeness)
            
            if sample_completeness:
                mean_completeness = float(np.mean(sample_completeness))
                pathway_completeness[pathway_name] = {
                    'mean': mean_completeness,
                    'std': float(np.std(sample_completeness)),
                    'num_samples_active': len(sample_completeness)
                }
                
                if mean_completeness < completeness_threshold:
                    incomplete_pathways.append(pathway_name)
        
        # Compute overall consistency score
        if pathway_completeness:
            mean_completeness = np.mean([p['mean'] for p in pathway_completeness.values()])
            consistency_score = float(mean_completeness)
        else:
            mean_completeness = 1.0
            consistency_score = 1.0
        
        return {
            'pathway_completeness': pathway_completeness,
            'mean_completeness': float(mean_completeness),
            'incomplete_pathways': incomplete_pathways,
            'consistency_score': consistency_score,
            'num_pathways_checked': len(self.metabolic_pathways)
        }
    
    def validate_disease_biomarkers(
        self,
        compositions: np.ndarray,
        disease_labels: Optional[np.ndarray] = None,
        presence_threshold: float = 0.001
    ) -> Dict[str, any]:
        """Validate disease-associated biomarker patterns.
        
        Checks whether compositions labeled with a disease show the expected
        biomarker patterns (elevated or depleted taxa).
        
        Args:
            compositions: Compositions to validate (num_samples, num_taxa)
            disease_labels: Optional array of disease labels for each sample.
                           If None, validates that biomarker patterns are
                           internally consistent.
            presence_threshold: Minimum abundance to consider a taxon present
        
        Returns:
            Dictionary containing:
                - 'biomarker_validation': Dict mapping disease to validation results
                - 'overall_accuracy': Fraction of correctly identified biomarkers
                - 'validation_score': Overall biomarker validation score
        """
        if not self.disease_biomarkers:
            return {
                'biomarker_validation': {},
                'overall_accuracy': 1.0,
                'validation_score': 1.0,
                'message': 'No disease biomarkers defined'
            }
        
        biomarker_validation = {}
        all_accuracies = []
        
        for disease_name, biomarkers in self.disease_biomarkers.items():
            elevated_taxa = biomarkers.get('elevated', [])
            depleted_taxa = biomarkers.get('depleted', [])
            
            # Filter valid taxa indices
            elevated_taxa = [t for t in elevated_taxa if t < compositions.shape[1]]
            depleted_taxa = [t for t in depleted_taxa if t < compositions.shape[1]]
            
            if not elevated_taxa and not depleted_taxa:
                continue
            
            if disease_labels is not None:
                # Validate against labeled samples
                disease_mask = disease_labels == disease_name
                healthy_mask = ~disease_mask
                
                if np.sum(disease_mask) == 0 or np.sum(healthy_mask) == 0:
                    continue
                
                disease_samples = compositions[disease_mask]
                healthy_samples = compositions[healthy_mask]
                
                # Check elevated taxa
                elevated_correct = 0
                for taxon in elevated_taxa:
                    disease_mean = np.mean(disease_samples[:, taxon])
                    healthy_mean = np.mean(healthy_samples[:, taxon])
                    if disease_mean > healthy_mean:
                        elevated_correct += 1
                
                # Check depleted taxa
                depleted_correct = 0
                for taxon in depleted_taxa:
                    disease_mean = np.mean(disease_samples[:, taxon])
                    healthy_mean = np.mean(healthy_samples[:, taxon])
                    if disease_mean < healthy_mean:
                        depleted_correct += 1
                
                total_biomarkers = len(elevated_taxa) + len(depleted_taxa)
                correct = elevated_correct + depleted_correct
                accuracy = correct / total_biomarkers if total_biomarkers > 0 else 1.0
                
                biomarker_validation[disease_name] = {
                    'elevated_accuracy': elevated_correct / len(elevated_taxa) if elevated_taxa else 1.0,
                    'depleted_accuracy': depleted_correct / len(depleted_taxa) if depleted_taxa else 1.0,
                    'overall_accuracy': float(accuracy),
                    'num_disease_samples': int(np.sum(disease_mask)),
                    'num_healthy_samples': int(np.sum(healthy_mask))
                }
                all_accuracies.append(accuracy)
            else:
                # Without labels, check internal consistency
                # Elevated taxa should have higher variance (present in some, absent in others)
                elevated_variances = [np.var(compositions[:, t]) for t in elevated_taxa]
                depleted_variances = [np.var(compositions[:, t]) for t in depleted_taxa]
                
                biomarker_validation[disease_name] = {
                    'elevated_mean_variance': float(np.mean(elevated_variances)) if elevated_variances else 0.0,
                    'depleted_mean_variance': float(np.mean(depleted_variances)) if depleted_variances else 0.0,
                    'num_elevated_taxa': len(elevated_taxa),
                    'num_depleted_taxa': len(depleted_taxa)
                }
        
        overall_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 1.0
        
        return {
            'biomarker_validation': biomarker_validation,
            'overall_accuracy': overall_accuracy,
            'validation_score': overall_accuracy,
            'num_diseases_checked': len(self.disease_biomarkers)
        }
    
    def validate_all(
        self,
        compositions: np.ndarray,
        disease_labels: Optional[np.ndarray] = None,
        presence_threshold: float = 0.001
    ) -> Dict[str, any]:
        """Run all biological validation checks.
        
        Args:
            compositions: Compositions to validate (num_samples, num_taxa)
            disease_labels: Optional disease labels for biomarker validation
            presence_threshold: Minimum abundance to consider a taxon present
        
        Returns:
            Dictionary containing all validation results and overall score
        """
        co_exclusion_results = self.validate_co_exclusion(
            compositions, presence_threshold
        )
        
        metabolic_results = self.validate_metabolic_consistency(
            compositions, presence_threshold
        )
        
        biomarker_results = self.validate_disease_biomarkers(
            compositions, disease_labels, presence_threshold
        )
        
        # Compute overall biological plausibility score
        scores = [
            co_exclusion_results['compliance_score'],
            metabolic_results['consistency_score'],
            biomarker_results['validation_score']
        ]
        overall_score = float(np.mean(scores))
        
        return {
            'co_exclusion': co_exclusion_results,
            'metabolic_consistency': metabolic_results,
            'disease_biomarkers': biomarker_results,
            'overall_biological_plausibility': overall_score
        }


class MethodComparator:
    """Statistical comparison of multiple methods with significance testing.
    
    This class provides comprehensive statistical comparison capabilities:
    - Pairwise statistical significance tests (t-test, Mann-Whitney U, Wilcoxon)
    - Effect size computation (Cohen's d, Cliff's delta)
    - Confidence interval estimation (bootstrap)
    - Multiple comparison correction (Bonferroni, Holm-Bonferroni)
    
    Attributes:
        alpha: Significance level for hypothesis tests (default: 0.05)
        correction_method: Method for multiple comparison correction
        bootstrap_samples: Number of bootstrap samples for CI estimation
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = 'holm',
        bootstrap_samples: int = 1000
    ):
        """Initialize method comparator.
        
        Args:
            alpha: Significance level for hypothesis tests
            correction_method: Multiple comparison correction method
                              ('bonferroni', 'holm', 'none')
            bootstrap_samples: Number of bootstrap samples for CI estimation
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.bootstrap_samples = bootstrap_samples
    
    def compare_methods(
        self,
        method_results: Dict[str, np.ndarray],
        metric_name: str = 'metric'
    ) -> Dict[str, any]:
        """Compare multiple methods on a single metric.
        
        Args:
            method_results: Dict mapping method names to arrays of metric values
                           (one value per sample/run)
            metric_name: Name of the metric being compared
        
        Returns:
            Dictionary containing:
                - 'summary': Summary statistics for each method
                - 'pairwise_tests': Pairwise significance tests
                - 'effect_sizes': Pairwise effect sizes
                - 'confidence_intervals': Bootstrap confidence intervals
                - 'ranking': Methods ranked by performance
        """
        if len(method_results) < 2:
            return {'error': 'Need at least 2 methods to compare'}
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(method_results)
        
        # Perform pairwise significance tests
        pairwise_tests = self._pairwise_significance_tests(method_results)
        
        # Compute effect sizes
        effect_sizes = self._compute_effect_sizes(method_results)
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(method_results)
        
        # Rank methods
        ranking = self._rank_methods(summary, metric_name)
        
        return {
            'metric_name': metric_name,
            'summary': summary,
            'pairwise_tests': pairwise_tests,
            'effect_sizes': effect_sizes,
            'confidence_intervals': confidence_intervals,
            'ranking': ranking
        }
    
    def _compute_summary_statistics(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for each method.
        
        Args:
            method_results: Dict mapping method names to metric arrays
        
        Returns:
            Dict mapping method names to summary statistics
        """
        summary = {}
        
        for method_name, values in method_results.items():
            values = np.asarray(values)
            
            summary[method_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values),
                'sem': float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            }
        
        return summary
    
    def _pairwise_significance_tests(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, any]]:
        """Perform pairwise statistical significance tests.
        
        Uses appropriate tests based on data characteristics:
        - Paired t-test for normally distributed paired data
        - Wilcoxon signed-rank for non-normal paired data
        - Mann-Whitney U for independent samples
        
        Args:
            method_results: Dict mapping method names to metric arrays
        
        Returns:
            Dict of pairwise test results with corrected p-values
        """
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro, wilcoxon
        
        methods = list(method_results.keys())
        pairwise_tests = {}
        all_pvalues = []
        comparison_keys = []
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                values1 = np.asarray(method_results[method1])
                values2 = np.asarray(method_results[method2])
                
                comparison_key = f'{method1}_vs_{method2}'
                comparison_keys.append(comparison_key)
                
                # Check if data is normally distributed (for small samples)
                try:
                    if len(values1) >= 3 and len(values2) >= 3:
                        _, p_normal1 = shapiro(values1)
                        _, p_normal2 = shapiro(values2)
                        is_normal = p_normal1 > 0.05 and p_normal2 > 0.05
                    else:
                        is_normal = True  # Assume normal for very small samples
                except Exception:
                    is_normal = True
                
                # Perform appropriate test
                try:
                    if len(values1) == len(values2):
                        # Paired samples - use Wilcoxon or paired t-test
                        if is_normal:
                            from scipy.stats import ttest_rel
                            statistic, pvalue = ttest_rel(values1, values2)
                            test_name = 'paired_t_test'
                        else:
                            statistic, pvalue = wilcoxon(values1, values2)
                            test_name = 'wilcoxon'
                    else:
                        # Independent samples
                        if is_normal:
                            statistic, pvalue = ttest_ind(values1, values2)
                            test_name = 't_test'
                        else:
                            statistic, pvalue = mannwhitneyu(values1, values2, alternative='two-sided')
                            test_name = 'mann_whitney_u'
                    
                    pairwise_tests[comparison_key] = {
                        'test': test_name,
                        'statistic': float(statistic),
                        'pvalue': float(pvalue),
                        'pvalue_corrected': None,  # Will be filled after correction
                        'significant': None  # Will be filled after correction
                    }
                    all_pvalues.append(pvalue)
                    
                except Exception as e:
                    pairwise_tests[comparison_key] = {
                        'test': 'failed',
                        'error': str(e)
                    }
                    all_pvalues.append(1.0)
        
        # Apply multiple comparison correction
        corrected_pvalues = self._correct_pvalues(all_pvalues)
        
        for idx, comparison_key in enumerate(comparison_keys):
            if 'pvalue' in pairwise_tests[comparison_key]:
                pairwise_tests[comparison_key]['pvalue_corrected'] = float(corrected_pvalues[idx])
                pairwise_tests[comparison_key]['significant'] = corrected_pvalues[idx] < self.alpha
        
        return pairwise_tests
    
    def _correct_pvalues(self, pvalues: list) -> np.ndarray:
        """Apply multiple comparison correction to p-values.
        
        Args:
            pvalues: List of raw p-values
        
        Returns:
            Array of corrected p-values
        """
        pvalues = np.asarray(pvalues)
        n = len(pvalues)
        
        if n == 0:
            return pvalues
        
        if self.correction_method == 'bonferroni':
            return np.minimum(pvalues * n, 1.0)
        
        elif self.correction_method == 'holm':
            # Holm-Bonferroni step-down procedure
            sorted_indices = np.argsort(pvalues)
            sorted_pvalues = pvalues[sorted_indices]
            
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = sorted_pvalues[i] * (n - i)
            
            # Ensure monotonicity
            corrected = np.minimum.accumulate(corrected[np.argsort(sorted_indices)][::-1])[::-1]
            return np.minimum(corrected, 1.0)
        
        else:  # 'none'
            return pvalues
    
    def _compute_effect_sizes(
        self,
        method_results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute effect sizes for pairwise comparisons.
        
        Computes:
        - Cohen's d: Standardized mean difference
        - Cliff's delta: Non-parametric effect size
        
        Args:
            method_results: Dict mapping method names to metric arrays
        
        Returns:
            Dict of pairwise effect sizes
        """
        methods = list(method_results.keys())
        effect_sizes = {}
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                values1 = np.asarray(method_results[method1])
                values2 = np.asarray(method_results[method2])
                
                comparison_key = f'{method1}_vs_{method2}'
                
                # Cohen's d
                cohens_d = self._cohens_d(values1, values2)
                
                # Cliff's delta
                cliffs_delta = self._cliffs_delta(values1, values2)
                
                # Interpret effect size
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    interpretation = 'negligible'
                elif abs_d < 0.5:
                    interpretation = 'small'
                elif abs_d < 0.8:
                    interpretation = 'medium'
                else:
                    interpretation = 'large'
                
                effect_sizes[comparison_key] = {
                    'cohens_d': float(cohens_d),
                    'cliffs_delta': float(cliffs_delta),
                    'interpretation': interpretation
                }
        
        return effect_sizes
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
        
        Returns:
            Cohen's d value (positive if group1 > group2)
        """
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cliff's delta effect size (non-parametric).
        
        Args:
            group1: First group of values
            group2: Second group of values
        
        Returns:
            Cliff's delta in [-1, 1]
        """
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Count dominance
        greater = 0
        less = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    greater += 1
                elif x < y:
                    less += 1
        
        return (greater - less) / (n1 * n2)
    
    def _compute_confidence_intervals(
        self,
        method_results: Dict[str, np.ndarray],
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for each method.
        
        Args:
            method_results: Dict mapping method names to metric arrays
            confidence_level: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Dict mapping method names to confidence interval bounds
        """
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for method_name, values in method_results.items():
            values = np.asarray(values)
            n = len(values)
            
            if n < 2:
                confidence_intervals[method_name] = {
                    'mean': float(np.mean(values)),
                    'ci_lower': float(np.mean(values)),
                    'ci_upper': float(np.mean(values)),
                    'ci_width': 0.0
                }
                continue
            
            # Bootstrap resampling
            bootstrap_means = []
            
            for _ in range(self.bootstrap_samples):
                resample = np.random.choice(values, size=n, replace=True)
                bootstrap_means.append(np.mean(resample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Compute percentile confidence interval
            ci_lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
            ci_upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
            
            confidence_intervals[method_name] = {
                'mean': float(np.mean(values)),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'confidence_level': confidence_level
            }
        
        return confidence_intervals
    
    def _rank_methods(
        self,
        summary: Dict[str, Dict[str, float]],
        metric_name: str
    ) -> list:
        """Rank methods by performance.
        
        Args:
            summary: Summary statistics for each method
            metric_name: Name of the metric (used to determine if lower is better)
        
        Returns:
            List of (method_name, mean_value) tuples sorted by performance
        """
        # Determine if lower is better based on metric name
        lower_is_better = any(
            term in metric_name.lower()
            for term in ['error', 'loss', 'mae', 'mse', 'mfd', 'violation', 'distance']
        )
        
        # Sort methods by mean value
        ranked = sorted(
            [(name, stats['mean']) for name, stats in summary.items()],
            key=lambda x: x[1],
            reverse=not lower_is_better
        )
        
        return [{'rank': i + 1, 'method': name, 'mean': value} for i, (name, value) in enumerate(ranked)]
    
    def compare_multiple_metrics(
        self,
        all_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, any]]:
        """Compare methods across multiple metrics.
        
        Args:
            all_results: Dict mapping method names to dicts of metric arrays
                        {method_name: {metric_name: values_array}}
        
        Returns:
            Dict mapping metric names to comparison results
        """
        # Reorganize data by metric
        metrics_data = {}
        
        for method_name, method_metrics in all_results.items():
            for metric_name, values in method_metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {}
                metrics_data[metric_name][method_name] = values
        
        # Compare each metric
        comparisons = {}
        
        for metric_name, method_results in metrics_data.items():
            comparisons[metric_name] = self.compare_methods(method_results, metric_name)
        
        return comparisons


class ComparisonVisualizer:
    """Publication-quality visualization for method comparisons.
    
    This class generates various visualization types for comparing methods:
    - Bar charts with error bars
    - Box plots with individual data points
    - Significance annotations
    - Multi-metric comparison plots
    
    Attributes:
        figsize: Default figure size (width, height)
        dpi: Resolution for saved figures
        style: Matplotlib style to use
        color_palette: Colors for different methods
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 300,
        style: str = 'seaborn-v0_8-whitegrid',
        color_palette: Optional[list] = None
    ):
        """Initialize comparison visualizer.
        
        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Resolution for saved figures
            style: Matplotlib style to use
            color_palette: List of colors for methods (default: tab10)
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def plot_bar_chart(
        self,
        method_results: Dict[str, np.ndarray],
        metric_name: str,
        comparison_results: Optional[Dict[str, any]] = None,
        output_path: Optional[str] = None,
        show_individual_points: bool = True,
        highlight_best: bool = True
    ):
        """Create bar chart with error bars comparing methods.
        
        Args:
            method_results: Dict mapping method names to metric arrays
            metric_name: Name of the metric being compared
            comparison_results: Optional pre-computed comparison results
            output_path: Path to save figure (if None, displays figure)
            show_individual_points: Whether to overlay individual data points
            highlight_best: Whether to highlight the best performing method
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        try:
            plt.style.use(self.style)
        except Exception:
            pass  # Use default style if specified style not available
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(method_results.keys())
        n_methods = len(methods)
        x_pos = np.arange(n_methods)
        
        # Compute statistics
        means = [np.mean(method_results[m]) for m in methods]
        stds = [np.std(method_results[m], ddof=1) if len(method_results[m]) > 1 else 0 for m in methods]
        sems = [std / np.sqrt(len(method_results[m])) for std, m in zip(stds, methods)]
        
        # Determine best method
        lower_is_better = any(
            term in metric_name.lower()
            for term in ['error', 'loss', 'mae', 'mse', 'mfd', 'violation', 'distance']
        )
        best_idx = np.argmin(means) if lower_is_better else np.argmax(means)
        
        # Create bar colors
        colors = [self.color_palette[i % len(self.color_palette)] for i in range(n_methods)]
        if highlight_best:
            colors[best_idx] = '#2ca02c'  # Green for best
        
        # Create bars with error bars
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        
        # Overlay individual data points
        if show_individual_points:
            for i, method in enumerate(methods):
                values = method_results[method]
                jitter = np.random.uniform(-0.15, 0.15, len(values))
                ax.scatter(x_pos[i] + jitter, values, color='black', alpha=0.5, s=20, zorder=3)
        
        # Add significance annotations if comparison results provided
        if comparison_results and 'pairwise_tests' in comparison_results:
            self._add_significance_annotations(
                ax, methods, means, stds, comparison_results['pairwise_tests']
            )
        
        # Formatting
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel(self._format_metric_name(metric_name), fontsize=12)
        ax.set_title(f'Comparison: {self._format_metric_name(metric_name)}', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend for best method
        if highlight_best:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2ca02c', alpha=0.7, label='Best')]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_box_plot(
        self,
        method_results: Dict[str, np.ndarray],
        metric_name: str,
        comparison_results: Optional[Dict[str, any]] = None,
        output_path: Optional[str] = None,
        show_points: bool = True,
        notch: bool = True
    ):
        """Create box plot comparing methods.
        
        Args:
            method_results: Dict mapping method names to metric arrays
            metric_name: Name of the metric being compared
            comparison_results: Optional pre-computed comparison results
            output_path: Path to save figure (if None, displays figure)
            show_points: Whether to overlay individual data points
            notch: Whether to show notched boxes (confidence interval for median)
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        try:
            plt.style.use(self.style)
        except Exception:
            pass
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(method_results.keys())
        data = [method_results[m] for m in methods]
        
        # Create box plot
        bp = ax.boxplot(
            data,
            labels=methods,
            notch=notch,
            patch_artist=True,
            showmeans=True,
            meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red'}
        )
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.color_palette[i % len(self.color_palette)])
            patch.set_alpha(0.7)
        
        # Overlay individual points
        if show_points:
            for i, (method, values) in enumerate(zip(methods, data)):
                jitter = np.random.uniform(-0.15, 0.15, len(values))
                ax.scatter(np.ones(len(values)) * (i + 1) + jitter, values, 
                          color='black', alpha=0.4, s=15, zorder=3)
        
        # Add significance annotations
        if comparison_results and 'pairwise_tests' in comparison_results:
            means = [np.mean(d) for d in data]
            stds = [np.std(d) for d in data]
            self._add_significance_annotations(
                ax, methods, means, stds, comparison_results['pairwise_tests'],
                x_offset=1  # Box plots are 1-indexed
            )
        
        # Formatting
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel(self._format_metric_name(metric_name), fontsize=12)
        ax.set_title(f'Distribution: {self._format_metric_name(metric_name)}', fontsize=14)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_multi_metric_comparison(
        self,
        all_results: Dict[str, Dict[str, np.ndarray]],
        metrics_to_plot: Optional[list] = None,
        output_path: Optional[str] = None,
        normalize: bool = True
    ):
        """Create multi-metric comparison plot (radar or grouped bar).
        
        Args:
            all_results: Dict mapping method names to dicts of metric arrays
            metrics_to_plot: List of metrics to include (default: all)
            output_path: Path to save figure
            normalize: Whether to normalize metrics to [0, 1] range
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        try:
            plt.style.use(self.style)
        except Exception:
            pass
        
        # Get all metrics
        all_metrics = set()
        for method_metrics in all_results.values():
            all_metrics.update(method_metrics.keys())
        
        if metrics_to_plot:
            metrics = [m for m in metrics_to_plot if m in all_metrics]
        else:
            metrics = sorted(all_metrics)
        
        if not metrics:
            return None
        
        methods = list(all_results.keys())
        n_methods = len(methods)
        n_metrics = len(metrics)
        
        # Compute mean values for each method-metric combination
        values_matrix = np.zeros((n_methods, n_metrics))
        
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                if metric in all_results[method]:
                    values_matrix[i, j] = np.mean(all_results[method][metric])
        
        # Normalize if requested
        if normalize:
            for j in range(n_metrics):
                col = values_matrix[:, j]
                min_val, max_val = col.min(), col.max()
                if max_val > min_val:
                    # Check if lower is better
                    lower_is_better = any(
                        term in metrics[j].lower()
                        for term in ['error', 'loss', 'mae', 'mse', 'mfd', 'violation']
                    )
                    if lower_is_better:
                        # Invert so higher is always better after normalization
                        values_matrix[:, j] = 1 - (col - min_val) / (max_val - min_val)
                    else:
                        values_matrix[:, j] = (col - min_val) / (max_val - min_val)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            offset = (i - n_methods / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values_matrix[i], width,
                label=method,
                color=self.color_palette[i % len(self.color_palette)],
                alpha=0.8
            )
        
        # Formatting
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Normalized Score' if normalize else 'Value', fontsize=12)
        ax.set_title('Multi-Metric Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([self._format_metric_name(m) for m in metrics], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        if normalize:
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_confidence_intervals(
        self,
        confidence_intervals: Dict[str, Dict[str, float]],
        metric_name: str,
        output_path: Optional[str] = None
    ):
        """Create forest plot showing confidence intervals.
        
        Args:
            confidence_intervals: Dict from MethodComparator.compare_methods()
            metric_name: Name of the metric
            output_path: Path to save figure
        
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        try:
            plt.style.use(self.style)
        except Exception:
            pass
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(confidence_intervals.keys())
        n_methods = len(methods)
        y_pos = np.arange(n_methods)
        
        means = [confidence_intervals[m]['mean'] for m in methods]
        ci_lowers = [confidence_intervals[m]['ci_lower'] for m in methods]
        ci_uppers = [confidence_intervals[m]['ci_upper'] for m in methods]
        
        # Compute error bars
        errors = np.array([[means[i] - ci_lowers[i], ci_uppers[i] - means[i]] for i in range(n_methods)]).T
        
        # Create horizontal bar chart with error bars
        ax.barh(y_pos, means, xerr=errors, capsize=5, 
                color=[self.color_palette[i % len(self.color_palette)] for i in range(n_methods)],
                alpha=0.7, edgecolor='black')
        
        # Add vertical line at overall mean
        overall_mean = np.mean(means)
        ax.axvline(x=overall_mean, color='red', linestyle='--', alpha=0.7, label='Overall Mean')
        
        # Formatting
        ax.set_xlabel(self._format_metric_name(metric_name), fontsize=12)
        ax.set_ylabel('Method', fontsize=12)
        ax.set_title(f'Confidence Intervals: {self._format_metric_name(metric_name)}', fontsize=14)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def _add_significance_annotations(
        self,
        ax,
        methods: list,
        means: list,
        stds: list,
        pairwise_tests: Dict[str, Dict[str, any]],
        x_offset: int = 0
    ):
        """Add significance annotations to a plot.
        
        Args:
            ax: Matplotlib axes object
            methods: List of method names
            means: List of mean values
            stds: List of standard deviations
            pairwise_tests: Dict of pairwise test results
            x_offset: Offset for x positions (0 for bar charts, 1 for box plots)
        """
        # Find significant comparisons
        significant_pairs = []
        
        for comparison_key, test_result in pairwise_tests.items():
            if test_result.get('significant', False):
                # Parse method names
                parts = comparison_key.split('_vs_')
                if len(parts) == 2:
                    m1, m2 = parts
                    if m1 in methods and m2 in methods:
                        idx1 = methods.index(m1)
                        idx2 = methods.index(m2)
                        pvalue = test_result.get('pvalue_corrected', test_result.get('pvalue', 0))
                        significant_pairs.append((idx1, idx2, pvalue))
        
        if not significant_pairs:
            return
        
        # Sort by distance between pairs (draw shorter connections first)
        significant_pairs.sort(key=lambda x: abs(x[1] - x[0]))
        
        # Get y-axis limits
        y_max = max(m + s for m, s in zip(means, stds))
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.05
        
        # Draw significance brackets
        for level, (idx1, idx2, pvalue) in enumerate(significant_pairs):
            x1 = idx1 + x_offset
            x2 = idx2 + x_offset
            y = y_max + y_offset * (level + 1)
            
            # Draw bracket
            ax.plot([x1, x1, x2, x2], [y - y_offset * 0.2, y, y, y - y_offset * 0.2], 
                   'k-', linewidth=1)
            
            # Add significance stars
            if pvalue < 0.001:
                stars = '***'
            elif pvalue < 0.01:
                stars = '**'
            else:
                stars = '*'
            
            ax.text((x1 + x2) / 2, y, stars, ha='center', va='bottom', fontsize=12)
        
        # Adjust y-axis limit to accommodate annotations
        ax.set_ylim(ax.get_ylim()[0], y_max + y_offset * (len(significant_pairs) + 1.5))
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display.
        
        Args:
            metric_name: Raw metric name
        
        Returns:
            Formatted metric name
        """
        # Replace underscores with spaces and title case
        formatted = metric_name.replace('_', ' ').title()
        
        # Handle common abbreviations
        replacements = {
            'Mae': 'MAE',
            'Mse': 'MSE',
            'Mfd': 'MFD',
            'Ci': 'CI',
            'Ks': 'KS'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def generate_all_comparison_plots(
        self,
        method_results: Dict[str, Dict[str, np.ndarray]],
        output_dir: str,
        comparator: Optional['MethodComparator'] = None
    ):
        """Generate all comparison plots for multiple metrics.
        
        Args:
            method_results: Dict mapping method names to dicts of metric arrays
            output_dir: Directory to save all plots
            comparator: Optional MethodComparator for significance testing
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if comparator is None:
            comparator = MethodComparator()
        
        # Reorganize data by metric
        metrics_data = {}
        for method_name, method_metrics in method_results.items():
            for metric_name, values in method_metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {}
                metrics_data[metric_name][method_name] = values
        
        # Generate plots for each metric
        for metric_name, metric_results in metrics_data.items():
            # Get comparison results
            comparison = comparator.compare_methods(metric_results, metric_name)
            
            # Bar chart
            self.plot_bar_chart(
                metric_results, metric_name, comparison,
                output_path=str(output_path / f'{metric_name}_bar.png')
            )
            
            # Box plot
            self.plot_box_plot(
                metric_results, metric_name, comparison,
                output_path=str(output_path / f'{metric_name}_box.png')
            )
            
            # Confidence intervals
            if 'confidence_intervals' in comparison:
                self.plot_confidence_intervals(
                    comparison['confidence_intervals'], metric_name,
                    output_path=str(output_path / f'{metric_name}_ci.png')
                )
        
        # Multi-metric comparison
        self.plot_multi_metric_comparison(
            method_results,
            output_path=str(output_path / 'multi_metric_comparison.png')
        )
