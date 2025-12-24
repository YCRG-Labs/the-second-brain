"""Publication-quality figure generation for microbiome simulation.

This module implements comprehensive figure generation for publication,
including:
- Generation quality plots (t-SNE/UMAP, diversity distributions, MFD)
- Prediction accuracy plots (time series, horizon metrics, error distributions)
- Ablation study plots (component contributions, performance degradation)
- Sensitivity analysis plots (hyperparameter sweeps, heatmaps)
- LaTeX table generation for quantitative results
"""

import numpy as np

# Use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings


# Publication-quality color schemes
PUBLICATION_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'quinary': '#3B1F2B',      # Dark purple
    'success': '#2E7D32',      # Green
    'warning': '#F57C00',      # Orange
    'error': '#C62828',        # Red
}

# Method-specific colors for consistent visualization
METHOD_COLORS = {
    'ours': '#2E86AB',
    'vae': '#A23B72',
    'gan': '#F18F01',
    'compositional_vae': '#C73E1D',
    'lstm': '#3B1F2B',
    'transformer': '#6A0572',
    'arima': '#7B7B7B',
}


class FigureGenerator:
    """Generate publication-quality figures for microbiome simulation.
    
    This class provides a unified interface for creating all figures needed
    for publication, with consistent styling, resolution, and formatting.
    
    Attributes:
        style: Matplotlib style to use
        dpi: Resolution for raster outputs
        figsize_single: Default figure size for single plots
        figsize_double: Default figure size for double-width plots
        font_size: Base font size for labels
        color_scheme: Color palette for plots
        output_format: Default output format ('pdf', 'svg', 'png')
    """
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-paper',
        dpi: int = 300,
        figsize_single: Tuple[float, float] = (4.5, 3.5),
        figsize_double: Tuple[float, float] = (9.0, 3.5),
        font_size: int = 10,
        color_scheme: Optional[Dict[str, str]] = None,
        output_format: str = 'pdf'
    ):
        """Initialize figure generator with publication settings.
        
        Args:
            style: Matplotlib style name (default: seaborn-v0_8-paper)
            dpi: Resolution for raster outputs (default: 300)
            figsize_single: Default size for single-column figures
            figsize_double: Default size for double-column figures
            font_size: Base font size for labels
            color_scheme: Custom color palette (default: PUBLICATION_COLORS)
            output_format: Default output format ('pdf', 'svg', 'png')
        """
        self.style = style
        self.dpi = dpi
        self.figsize_single = figsize_single
        self.figsize_double = figsize_double
        self.font_size = font_size
        self.color_scheme = color_scheme or PUBLICATION_COLORS
        self.output_format = output_format
        
        # Configure matplotlib defaults
        self._configure_matplotlib()
    
    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for publication-quality output."""
        # Try to use the specified style, fall back to defaults if not available
        try:
            plt.style.use(self.style)
        except OSError:
            # Style not found, use default settings
            warnings.warn(f"Style '{self.style}' not found, using defaults")
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'dejavusans',
        })
    
    def _get_method_color(self, method_name: str) -> str:
        """Get consistent color for a method name."""
        method_lower = method_name.lower().replace(' ', '_').replace('-', '_')
        
        if method_lower in METHOD_COLORS:
            return METHOD_COLORS[method_lower]
        
        # Generate a consistent color based on method name hash
        colors = list(mcolors.TABLEAU_COLORS.values())
        idx = hash(method_name) % len(colors)
        return colors[idx]
    
    def _save_figure(
        self,
        fig: plt.Figure,
        output_path: Path,
        formats: Optional[List[str]] = None
    ) -> None:
        """Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure to save
            output_path: Base path for output (without extension)
            formats: List of formats to save (default: [self.output_format])
        """
        if formats is None:
            formats = [self.output_format]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            save_path = output_path.with_suffix(f'.{fmt}')
            fig.savefig(save_path, format=fmt, dpi=self.dpi, bbox_inches='tight')

    
    def plot_generation_quality(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        output_path: Union[str, Path],
        method_name: str = 'Generated',
        include_tsne: bool = True,
        include_umap: bool = False,
        random_state: int = 42
    ) -> None:
        """Generate Figure: Generation quality comparison.
        
        Creates a multi-panel figure showing:
        - t-SNE/UMAP visualization of real vs generated samples
        - Alpha diversity (Shannon entropy) distributions
        - Beta diversity (Bray-Curtis) distributions
        
        Args:
            real_data: Real compositions of shape (num_real, num_taxa)
            generated_data: Generated compositions of shape (num_gen, num_taxa)
            output_path: Path to save the figure
            method_name: Name of the generation method for labels
            include_tsne: Whether to include t-SNE visualization
            include_umap: Whether to include UMAP visualization
            random_state: Random seed for dimensionality reduction
        """
        from src.evaluation import alpha_diversity, beta_diversity
        
        # Determine number of panels
        num_panels = 2  # Always include diversity plots
        if include_tsne:
            num_panels += 1
        if include_umap:
            num_panels += 1
        
        fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 3.5))
        if num_panels == 1:
            axes = [axes]
        
        panel_idx = 0
        
        # t-SNE visualization
        if include_tsne:
            ax = axes[panel_idx]
            self._plot_embedding(
                ax, real_data, generated_data, method_name,
                method='tsne', random_state=random_state
            )
            panel_idx += 1
        
        # UMAP visualization
        if include_umap:
            ax = axes[panel_idx]
            self._plot_embedding(
                ax, real_data, generated_data, method_name,
                method='umap', random_state=random_state
            )
            panel_idx += 1
        
        # Alpha diversity distribution
        ax = axes[panel_idx]
        self._plot_alpha_diversity_comparison(ax, real_data, generated_data, method_name)
        panel_idx += 1
        
        # Beta diversity distribution
        ax = axes[panel_idx]
        self._plot_beta_diversity_comparison(ax, real_data, generated_data, method_name)
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def _plot_embedding(
        self,
        ax: plt.Axes,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        method_name: str,
        method: str = 'tsne',
        random_state: int = 42
    ) -> None:
        """Plot dimensionality reduction embedding."""
        # Combine data for joint embedding
        combined = np.vstack([real_data, generated_data])
        labels = np.array(['Real'] * len(real_data) + [method_name] * len(generated_data))
        
        # Compute embedding
        if method == 'tsne':
            try:
                from sklearn.manifold import TSNE
                # Ensure perplexity is less than n_samples
                n_samples = combined.shape[0]
                perplexity = min(30, max(5, n_samples - 1))
                
                # Check for degenerate cases (all identical points)
                if np.allclose(combined, combined[0], rtol=1e-10):
                    # Create a simple scatter plot for identical points
                    embedding = np.random.RandomState(random_state).normal(0, 0.1, (n_samples, 2))
                else:
                    reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
                    embedding = reducer.fit_transform(combined)
                title = 't-SNE Visualization'
            except ImportError:
                warnings.warn("sklearn not available for t-SNE")
                return
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=random_state)
                embedding = reducer.fit_transform(combined)
                title = 'UMAP Visualization'
            except ImportError:
                warnings.warn("umap-learn not available for UMAP")
                return
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        # Plot
        real_mask = labels == 'Real'
        ax.scatter(
            embedding[real_mask, 0], embedding[real_mask, 1],
            c=self.color_scheme['primary'], alpha=0.6, s=20, label='Real'
        )
        ax.scatter(
            embedding[~real_mask, 0], embedding[~real_mask, 1],
            c=self.color_scheme['secondary'], alpha=0.6, s=20, label=method_name
        )
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(title)
        ax.legend(loc='best', framealpha=0.9)
    
    def _plot_alpha_diversity_comparison(
        self,
        ax: plt.Axes,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        method_name: str
    ) -> None:
        """Plot alpha diversity distribution comparison."""
        from src.evaluation import alpha_diversity
        
        real_alpha = alpha_diversity(real_data)
        gen_alpha = alpha_diversity(generated_data)
        
        # Histogram comparison
        bins = np.linspace(
            min(real_alpha.min(), gen_alpha.min()),
            max(real_alpha.max(), gen_alpha.max()),
            30
        )
        
        ax.hist(real_alpha, bins=bins, alpha=0.6, density=True,
                color=self.color_scheme['primary'], label='Real')
        ax.hist(gen_alpha, bins=bins, alpha=0.6, density=True,
                color=self.color_scheme['secondary'], label=method_name)
        
        # Add KS test p-value
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(real_alpha, gen_alpha)
        ax.text(0.95, 0.95, f'KS p={p_value:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Shannon Entropy')
        ax.set_ylabel('Density')
        ax.set_title('Alpha Diversity Distribution')
        ax.legend(loc='upper left', framealpha=0.9)
    
    def _plot_beta_diversity_comparison(
        self,
        ax: plt.Axes,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        method_name: str
    ) -> None:
        """Plot beta diversity distribution comparison."""
        from src.evaluation import beta_diversity
        
        # Compute pairwise distances
        real_beta = beta_diversity(real_data)
        gen_beta = beta_diversity(generated_data)
        
        # Extract upper triangle values
        real_values = real_beta[np.triu_indices_from(real_beta, k=1)]
        gen_values = gen_beta[np.triu_indices_from(gen_beta, k=1)]
        
        # Histogram comparison
        bins = np.linspace(0, 1, 30)
        
        ax.hist(real_values, bins=bins, alpha=0.6, density=True,
                color=self.color_scheme['primary'], label='Real')
        ax.hist(gen_values, bins=bins, alpha=0.6, density=True,
                color=self.color_scheme['secondary'], label=method_name)
        
        # Add KS test p-value
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(real_values, gen_values)
        ax.text(0.95, 0.95, f'KS p={p_value:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=self.font_size - 2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Bray-Curtis Dissimilarity')
        ax.set_ylabel('Density')
        ax.set_title('Beta Diversity Distribution')
        ax.legend(loc='upper left', framealpha=0.9)

    
    def plot_mfd_comparison(
        self,
        method_results: Dict[str, float],
        output_path: Union[str, Path],
        title: str = 'Microbiome Fréchet Distance Comparison',
        error_bars: Optional[Dict[str, float]] = None
    ) -> None:
        """Plot MFD comparison bar chart across methods.
        
        Args:
            method_results: Dict mapping method names to MFD values
            output_path: Path to save the figure
            title: Plot title
            error_bars: Optional dict mapping method names to std errors
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        methods = list(method_results.keys())
        values = [method_results[m] for m in methods]
        colors = [self._get_method_color(m) for m in methods]
        
        x_pos = np.arange(len(methods))
        
        if error_bars:
            errors = [error_bars.get(m, 0) for m in methods]
            bars = ax.bar(x_pos, values, yerr=errors, capsize=5,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        else:
            bars = ax.bar(x_pos, values, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
        
        # Highlight best (lowest) MFD
        best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor(self.color_scheme['success'])
        bars[best_idx].set_linewidth(2)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('MFD (↓ better)')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def plot_prediction_accuracy(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray,
        time_points: np.ndarray,
        output_path: Union[str, Path],
        confidence_level: float = 0.95,
        sample_idx: int = 0,
        taxon_idx: int = 0
    ) -> None:
        """Generate Figure: Prediction accuracy with confidence intervals.
        
        Creates a multi-panel figure showing:
        - Time series predictions with confidence intervals
        - Horizon-specific MAE metrics
        - Error distributions
        
        Args:
            predictions: Dict mapping method names to predictions
                        Shape: (num_samples, num_horizons, num_taxa)
            ground_truth: True values, shape (num_samples, num_horizons, num_taxa)
            time_points: Time points for x-axis
            output_path: Path to save the figure
            confidence_level: Confidence level for intervals (default: 0.95)
            sample_idx: Sample index to visualize for time series
            taxon_idx: Taxon index to visualize for time series
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        
        # Panel 1: Time series with confidence intervals
        self._plot_time_series_predictions(
            axes[0], predictions, ground_truth, time_points,
            sample_idx, taxon_idx, confidence_level
        )
        
        # Panel 2: Horizon-specific MAE
        self._plot_horizon_metrics(axes[1], predictions, ground_truth)
        
        # Panel 3: Error distributions
        self._plot_error_distributions(axes[2], predictions, ground_truth)
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def _plot_time_series_predictions(
        self,
        ax: plt.Axes,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray,
        time_points: np.ndarray,
        sample_idx: int,
        taxon_idx: int,
        confidence_level: float
    ) -> None:
        """Plot time series predictions with confidence intervals."""
        # Plot ground truth
        gt_values = ground_truth[sample_idx, :, taxon_idx]
        ax.plot(time_points, gt_values, 'ko-', label='Ground Truth',
                linewidth=2, markersize=6)
        
        # Plot each method's predictions
        for method_name, preds in predictions.items():
            pred_values = preds[:, :, taxon_idx]  # All samples, all horizons, one taxon
            
            # Compute mean and confidence interval across samples
            mean_pred = np.mean(pred_values, axis=0)
            std_pred = np.std(pred_values, axis=0)
            
            # Confidence interval
            from scipy.stats import t
            n = pred_values.shape[0]
            t_val = t.ppf((1 + confidence_level) / 2, n - 1)
            ci = t_val * std_pred / np.sqrt(n)
            
            color = self._get_method_color(method_name)
            ax.plot(time_points, mean_pred, '-', color=color, label=method_name,
                   linewidth=1.5)
            ax.fill_between(time_points, mean_pred - ci, mean_pred + ci,
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Abundance')
        ax.set_title('Temporal Predictions')
        ax.legend(loc='best', fontsize=self.font_size - 2)
    
    def _plot_horizon_metrics(
        self,
        ax: plt.Axes,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray
    ) -> None:
        """Plot MAE metrics per prediction horizon."""
        from src.evaluation import abundance_mae
        
        num_horizons = ground_truth.shape[1]
        horizons = np.arange(1, num_horizons + 1)
        
        for method_name, preds in predictions.items():
            maes = []
            for h in range(num_horizons):
                mae = abundance_mae(preds[:, h, :], ground_truth[:, h, :])
                maes.append(mae)
            
            color = self._get_method_color(method_name)
            ax.plot(horizons, maes, 'o-', color=color, label=method_name,
                   linewidth=1.5, markersize=5)
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('MAE')
        ax.set_title('Horizon-Specific Error')
        ax.legend(loc='best', fontsize=self.font_size - 2)
        ax.set_xticks(horizons)
    
    def _plot_error_distributions(
        self,
        ax: plt.Axes,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray
    ) -> None:
        """Plot error distributions as violin plots."""
        errors_data = []
        labels = []
        colors = []
        
        for method_name, preds in predictions.items():
            errors = np.abs(preds - ground_truth).flatten()
            errors_data.append(errors)
            labels.append(method_name)
            colors.append(self._get_method_color(method_name))
        
        parts = ax.violinplot(errors_data, positions=range(len(labels)),
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    
    def plot_ablation_study(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
        baseline_name: str = 'Full Model',
        metrics_to_plot: Optional[List[str]] = None
    ) -> None:
        """Generate Figure: Ablation study results.
        
        Creates a multi-panel figure showing:
        - Component contribution bars
        - Performance degradation from removing each component
        
        Args:
            ablation_results: Dict mapping ablation names to metric dicts
                             e.g., {'Full Model': {'mfd': 10.5, 'mae': 0.02},
                                    'No Hyperbolic': {'mfd': 15.2, 'mae': 0.03}}
            output_path: Path to save the figure
            baseline_name: Name of the full model for comparison
            metrics_to_plot: List of metrics to include (default: all)
        """
        if baseline_name not in ablation_results:
            raise ValueError(f"Baseline '{baseline_name}' not in results")
        
        baseline_metrics = ablation_results[baseline_name]
        
        if metrics_to_plot is None:
            metrics_to_plot = list(baseline_metrics.keys())
        
        # Filter to only metrics that exist
        metrics_to_plot = [m for m in metrics_to_plot if m in baseline_metrics]
        
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]
        
        ablation_names = [k for k in ablation_results.keys() if k != baseline_name]
        
        for ax, metric in zip(axes, metrics_to_plot):
            self._plot_ablation_metric(
                ax, ablation_results, baseline_name, ablation_names, metric
            )
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def _plot_ablation_metric(
        self,
        ax: plt.Axes,
        ablation_results: Dict[str, Dict[str, float]],
        baseline_name: str,
        ablation_names: List[str],
        metric: str
    ) -> None:
        """Plot ablation results for a single metric."""
        baseline_value = ablation_results[baseline_name].get(metric, 0)
        
        # Compute relative change for each ablation
        names = [baseline_name] + ablation_names
        values = [baseline_value]
        relative_changes = [0.0]  # Baseline has 0% change
        
        for name in ablation_names:
            value = ablation_results[name].get(metric, baseline_value)
            values.append(value)
            
            if baseline_value != 0:
                rel_change = (value - baseline_value) / abs(baseline_value) * 100
            else:
                rel_change = 0.0
            relative_changes.append(rel_change)
        
        x_pos = np.arange(len(names))
        
        # Determine if higher or lower is better
        is_error_metric = any(
            err in metric.lower()
            for err in ['mae', 'mse', 'error', 'loss', 'mfd', 'violation']
        )
        
        # Color bars based on performance change
        colors = []
        for i, change in enumerate(relative_changes):
            if i == 0:  # Baseline
                colors.append(self.color_scheme['primary'])
            elif (is_error_metric and change > 0) or (not is_error_metric and change < 0):
                # Performance degraded
                colors.append(self.color_scheme['error'])
            else:
                # Performance improved (unexpected for ablation)
                colors.append(self.color_scheme['success'])
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        
        # Add percentage change annotations
        for i, (bar, change) in enumerate(zip(bars, relative_changes)):
            if i > 0:  # Skip baseline
                sign = '+' if change > 0 else ''
                ax.annotate(f'{sign}{change:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom',
                           fontsize=self.font_size - 2,
                           color='red' if (is_error_metric and change > 0) else 'green')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Ablation: {metric.replace("_", " ").title()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add baseline reference line
        ax.axhline(y=baseline_value, color='gray', linestyle='--', alpha=0.5)
    
    def plot_component_contributions(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
        baseline_name: str = 'Full Model',
        metric: str = 'mfd'
    ) -> None:
        """Plot component contribution as stacked bar chart.
        
        Shows how much each component contributes to the overall performance.
        
        Args:
            ablation_results: Dict mapping ablation names to metric dicts
            output_path: Path to save the figure
            baseline_name: Name of the full model
            metric: Metric to analyze
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        baseline_value = ablation_results[baseline_name].get(metric, 0)
        
        # Compute contribution of each component
        contributions = {}
        for name, metrics in ablation_results.items():
            if name == baseline_name:
                continue
            
            value = metrics.get(metric, baseline_value)
            contribution = value - baseline_value
            
            # Extract component name from ablation name
            component = name.replace('No ', '').replace('Without ', '')
            contributions[component] = contribution
        
        # Sort by contribution magnitude
        sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        components = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(components))
        colors = [self.color_scheme['error'] if v > 0 else self.color_scheme['success']
                 for v in values]
        
        ax.barh(y_pos, values, color=colors, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(f'Impact on {metric.upper()} (Δ from baseline)')
        ax.set_ylabel('Component')
        ax.set_title('Component Contributions')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)

    
    def plot_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Dict[str, List[float]]],
        output_path: Union[str, Path],
        param_name: str = 'Parameter',
        metrics_to_plot: Optional[List[str]] = None
    ) -> None:
        """Generate Figure: Sensitivity analysis plots.
        
        Creates line plots showing how metrics change with hyperparameter values.
        
        Args:
            sensitivity_results: Dict with structure:
                {'param_values': [0.001, 0.01, 0.1],
                 'metric1': {'mean': [...], 'std': [...]},
                 'metric2': {'mean': [...], 'std': [...]}}
            output_path: Path to save the figure
            param_name: Name of the parameter being swept
            metrics_to_plot: List of metrics to include (default: all)
        """
        if 'param_values' not in sensitivity_results:
            raise ValueError("sensitivity_results must contain 'param_values'")
        
        param_values = sensitivity_results['param_values']
        
        # Get metrics to plot
        available_metrics = [k for k in sensitivity_results.keys() if k != 'param_values']
        if metrics_to_plot is None:
            metrics_to_plot = available_metrics
        else:
            metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
        
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 3.5))
        if num_metrics == 1:
            axes = [axes]
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for ax, metric in zip(axes, metrics_to_plot):
            metric_data = sensitivity_results[metric]
            
            if isinstance(metric_data, dict):
                means = metric_data.get('mean', metric_data.get('values', []))
                stds = metric_data.get('std', [0] * len(means))
            else:
                means = metric_data
                stds = [0] * len(means)
            
            color = colors[metrics_to_plot.index(metric) % len(colors)]
            
            ax.plot(param_values, means, 'o-', color=color, linewidth=1.5, markersize=6)
            
            if any(s > 0 for s in stds):
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.fill_between(param_values, means_arr - stds_arr, means_arr + stds_arr,
                               color=color, alpha=0.2)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Sensitivity: {metric.replace("_", " ").title()}')
            
            # Use log scale if values span multiple orders of magnitude
            if len(param_values) > 1:
                ratio = max(param_values) / min(param_values)
                if ratio > 100:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def plot_sensitivity_heatmap(
        self,
        param1_values: List[float],
        param2_values: List[float],
        metric_values: np.ndarray,
        output_path: Union[str, Path],
        param1_name: str = 'Parameter 1',
        param2_name: str = 'Parameter 2',
        metric_name: str = 'Metric',
        cmap: str = 'viridis'
    ) -> None:
        """Generate heatmap for 2D sensitivity analysis.
        
        Args:
            param1_values: Values for first parameter (x-axis)
            param2_values: Values for second parameter (y-axis)
            metric_values: 2D array of metric values, shape (len(param2), len(param1))
            output_path: Path to save the figure
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            metric_name: Name of the metric
            cmap: Colormap name
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        # Create heatmap
        im = ax.imshow(metric_values, cmap=cmap, aspect='auto',
                      origin='lower', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name.replace('_', ' ').title())
        
        # Set tick labels
        ax.set_xticks(np.arange(len(param1_values)))
        ax.set_yticks(np.arange(len(param2_values)))
        ax.set_xticklabels([f'{v:.2g}' for v in param1_values], rotation=45, ha='right')
        ax.set_yticklabels([f'{v:.2g}' for v in param2_values])
        
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Sensitivity')
        
        # Add value annotations
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                value = metric_values[i, j]
                text_color = 'white' if value > np.median(metric_values) else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=self.font_size - 2)
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)
    
    def plot_hyperparameter_sweep(
        self,
        sweep_results: Dict[str, Dict[str, Any]],
        output_path: Union[str, Path],
        metric: str = 'mfd'
    ) -> None:
        """Plot hyperparameter sweep results with confidence intervals.
        
        Args:
            sweep_results: Dict mapping param names to sweep data:
                {'learning_rate': {'values': [...], 'metric_mean': [...], 'metric_std': [...]},
                 'embedding_dim': {...}}
            output_path: Path to save the figure
            metric: Metric name for labeling
        """
        num_params = len(sweep_results)
        fig, axes = plt.subplots(1, num_params, figsize=(4 * num_params, 3.5))
        if num_params == 1:
            axes = [axes]
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for ax, (param_name, data) in zip(axes, sweep_results.items()):
            values = data['values']
            means = np.array(data.get('metric_mean', data.get('mean', [])))
            stds = np.array(data.get('metric_std', data.get('std', [0] * len(means))))
            
            color = colors[list(sweep_results.keys()).index(param_name) % len(colors)]
            
            ax.plot(values, means, 'o-', color=color, linewidth=1.5, markersize=6)
            ax.fill_between(values, means - stds, means + stds, color=color, alpha=0.2)
            
            # Mark best value
            best_idx = np.argmin(means)  # Assuming lower is better
            ax.scatter([values[best_idx]], [means[best_idx]], 
                      color=self.color_scheme['success'], s=100, zorder=5,
                      marker='*', edgecolors='black', linewidths=0.5)
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{param_name.replace("_", " ").title()} Sweep')
            
            # Use log scale if appropriate
            if len(values) > 1 and max(values) / min(values) > 100:
                ax.set_xscale('log')
        
        plt.tight_layout()
        self._save_figure(fig, Path(output_path), formats=['pdf', 'png'])
        plt.close(fig)



class LaTeXTableGenerator:
    """Generate publication-quality LaTeX tables.
    
    This class provides methods for generating LaTeX-formatted tables
    for quantitative results, statistical significance, and comparisons.
    
    Attributes:
        precision: Number of decimal places for numeric values
        bold_best: Whether to bold the best value in each column
        include_std: Whether to include standard deviations
    """
    
    def __init__(
        self,
        precision: int = 3,
        bold_best: bool = True,
        include_std: bool = True
    ):
        """Initialize LaTeX table generator.
        
        Args:
            precision: Number of decimal places for numeric values
            bold_best: Whether to bold the best value in each column
            include_std: Whether to include standard deviations
        """
        self.precision = precision
        self.bold_best = bold_best
        self.include_std = include_std
    
    def generate_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        output_path: Union[str, Path],
        caption: str = 'Method Comparison',
        label: str = 'tab:comparison',
        std_results: Optional[Dict[str, Dict[str, float]]] = None,
        lower_is_better: Optional[Dict[str, bool]] = None
    ) -> str:
        """Generate LaTeX table comparing methods across metrics.
        
        Args:
            results: Dict mapping method names to metric dicts
            metrics: List of metric names to include
            output_path: Path to save the .tex file
            caption: Table caption
            label: LaTeX label for referencing
            std_results: Optional dict with standard deviations
            lower_is_better: Dict mapping metric names to whether lower is better
        
        Returns:
            LaTeX table string
        """
        if lower_is_better is None:
            lower_is_better = {}
            for m in metrics:
                # Default: error metrics are lower-is-better
                lower_is_better[m] = any(
                    err in m.lower()
                    for err in ['mae', 'mse', 'error', 'loss', 'mfd', 'violation']
                )
        
        methods = list(results.keys())
        
        # Build table
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        # Column specification
        col_spec = 'l' + 'c' * len(metrics)
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append('\\toprule')
        
        # Header row
        header = 'Method & ' + ' & '.join([m.replace('_', ' ').title() for m in metrics])
        lines.append(header + ' \\\\')
        lines.append('\\midrule')
        
        # Find best values for each metric
        best_values = {}
        for metric in metrics:
            values = [results[m].get(metric, float('inf')) for m in methods]
            if lower_is_better.get(metric, True):
                best_values[metric] = min(values)
            else:
                best_values[metric] = max(values)
        
        # Data rows
        for method in methods:
            row_values = [method]
            
            for metric in metrics:
                value = results[method].get(metric, float('nan'))
                
                # Format value
                if np.isnan(value):
                    formatted = '-'
                else:
                    formatted = f'{value:.{self.precision}f}'
                    
                    # Add std if available
                    if self.include_std and std_results and method in std_results:
                        std = std_results[method].get(metric, 0)
                        if std > 0:
                            formatted = f'{value:.{self.precision}f} $\\pm$ {std:.{self.precision}f}'
                    
                    # Bold best value
                    if self.bold_best and np.isclose(value, best_values[metric], rtol=1e-6):
                        formatted = f'\\textbf{{{formatted}}}'
                
                row_values.append(formatted)
            
            lines.append(' & '.join(row_values) + ' \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    
    def generate_ablation_table(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        metrics: List[str],
        output_path: Union[str, Path],
        baseline_name: str = 'Full Model',
        caption: str = 'Ablation Study Results',
        label: str = 'tab:ablation'
    ) -> str:
        """Generate LaTeX table for ablation study results.
        
        Includes relative change from baseline for each ablation.
        
        Args:
            ablation_results: Dict mapping ablation names to metric dicts
            metrics: List of metric names to include
            output_path: Path to save the .tex file
            baseline_name: Name of the baseline configuration
            caption: Table caption
            label: LaTeX label
        
        Returns:
            LaTeX table string
        """
        configurations = list(ablation_results.keys())
        baseline_metrics = ablation_results.get(baseline_name, {})
        
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        # Column specification: config name + metric value + relative change for each metric
        num_cols = 1 + len(metrics) * 2
        col_spec = 'l' + 'cc' * len(metrics)
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append('\\toprule')
        
        # Header row
        header_parts = ['Configuration']
        for metric in metrics:
            metric_name = metric.replace('_', ' ').title()
            header_parts.extend([metric_name, '$\\Delta$\\%'])
        lines.append(' & '.join(header_parts) + ' \\\\')
        lines.append('\\midrule')
        
        # Data rows
        for config in configurations:
            row_values = [config]
            
            for metric in metrics:
                value = ablation_results[config].get(metric, float('nan'))
                baseline_value = baseline_metrics.get(metric, value)
                
                # Format value
                if np.isnan(value):
                    formatted_value = '-'
                    formatted_change = '-'
                else:
                    formatted_value = f'{value:.{self.precision}f}'
                    
                    # Compute relative change
                    if baseline_value != 0 and config != baseline_name:
                        rel_change = (value - baseline_value) / abs(baseline_value) * 100
                        sign = '+' if rel_change > 0 else ''
                        formatted_change = f'{sign}{rel_change:.1f}'
                        
                        # Color code: red for degradation, green for improvement
                        is_error_metric = any(
                            err in metric.lower()
                            for err in ['mae', 'mse', 'error', 'loss', 'mfd']
                        )
                        if (is_error_metric and rel_change > 0) or (not is_error_metric and rel_change < 0):
                            formatted_change = f'\\textcolor{{red}}{{{formatted_change}}}'
                        elif rel_change != 0:
                            formatted_change = f'\\textcolor{{green}}{{{formatted_change}}}'
                    else:
                        formatted_change = '-'
                
                row_values.extend([formatted_value, formatted_change])
            
            lines.append(' & '.join(row_values) + ' \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    
    def generate_significance_table(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        output_path: Union[str, Path],
        caption: str = 'Statistical Significance Tests',
        label: str = 'tab:significance',
        include_effect_sizes: bool = True
    ) -> str:
        """Generate LaTeX table with statistical significance results.
        
        Args:
            comparison_results: Output from MethodComparator.compare_methods()
            output_path: Path to save the .tex file
            caption: Table caption
            label: LaTeX label
            include_effect_sizes: Whether to include effect size columns
        
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        # Determine column specification based on whether effect sizes are included
        if include_effect_sizes:
            lines.append('\\begin{tabular}{lcccc}')
            lines.append('\\toprule')
            lines.append(r'Comparison & p-value & Significant & Cohen\'s d & Effect Size \\\\')
        else:
            lines.append('\\begin{tabular}{lccc}')
            lines.append('\\toprule')
            lines.append('Comparison & Metric & p-value & Significant \\\\')
        
        lines.append('\\midrule')
        
        # Handle both single metric and multi-metric results
        if 'pairwise_tests' in comparison_results:
            # Single metric result from MethodComparator
            self._add_significance_rows(lines, comparison_results, include_effect_sizes)
        else:
            # Multi-metric results
            for metric_name, metric_data in comparison_results.items():
                if 'pairwise_tests' not in metric_data:
                    continue
                self._add_significance_rows(lines, metric_data, include_effect_sizes, metric_name)
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    
    def _add_significance_rows(
        self,
        lines: List[str],
        metric_data: Dict[str, Any],
        include_effect_sizes: bool,
        metric_name: Optional[str] = None
    ) -> None:
        """Add significance test rows to LaTeX table."""
        pairwise_tests = metric_data.get('pairwise_tests', {})
        effect_sizes = metric_data.get('effect_sizes', {})
        
        for comparison_key, test_result in pairwise_tests.items():
            if 'pvalue' not in test_result:
                continue
                
            p_value = test_result.get('pvalue_corrected', test_result['pvalue'])
            significant = test_result.get('significant', p_value < 0.05)
            
            # Format p-value
            if p_value < 0.001:
                p_str = '$<$0.001'
            elif p_value < 0.01:
                p_str = f'{p_value:.3f}'
            else:
                p_str = f'{p_value:.3f}'
            
            # Format significance
            sig_str = '\\checkmark' if significant else '-'
            if significant:
                sig_str = f'\\textbf{{{sig_str}}}'
            
            # Clean up comparison key
            comparison_clean = comparison_key.replace('_vs_', ' vs ')
            
            if include_effect_sizes and comparison_key in effect_sizes:
                # Include effect size information
                effect_data = effect_sizes[comparison_key]
                cohens_d = effect_data.get('cohens_d', 0.0)
                interpretation = effect_data.get('interpretation', 'unknown')
                
                cohens_d_str = f'{cohens_d:.3f}'
                if abs(cohens_d) > 0.8:
                    cohens_d_str = f'\\textbf{{{cohens_d_str}}}'  # Bold for large effects
                
                lines.append(f'{comparison_clean} & {p_str} & {sig_str} & {cohens_d_str} & {interpretation.title()} \\\\')
            else:
                # Original format with metric name
                if metric_name:
                    metric_clean = metric_name.replace('_', ' ').title()
                    lines.append(f'{comparison_clean} & {metric_clean} & {p_str} & {sig_str} \\\\')
                else:
                    lines.append(f'{comparison_clean} & {p_str} & {sig_str} \\\\')
    
    def generate_statistical_summary_table(
        self,
        method_results: Dict[str, Dict[str, float]],
        output_path: Union[str, Path],
        metrics: List[str],
        caption: str = 'Statistical Summary',
        label: str = 'tab:summary',
        include_confidence_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> str:
        """Generate LaTeX table with statistical summary of results.
        
        Args:
            method_results: Dict mapping method names to metric dicts with 'mean', 'std', etc.
            output_path: Path to save the .tex file
            metrics: List of metric names to include
            caption: Table caption
            label: LaTeX label
            include_confidence_intervals: Whether to include confidence intervals
            confidence_level: Confidence level for CIs (default: 0.95)
        
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        # Determine column specification
        if include_confidence_intervals:
            col_spec = 'l' + 'cc' * len(metrics)  # Method + (Mean±CI, Std) for each metric
            lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
            lines.append('\\toprule')
            
            # Header row with confidence intervals
            header_parts = ['Method']
            for metric in metrics:
                metric_name = metric.replace('_', ' ').title()
                ci_percent = int(confidence_level * 100)
                header_parts.extend([f'{metric_name} ({ci_percent}\\% CI)', 'Std Dev'])
            lines.append(' & '.join(header_parts) + ' \\\\')
        else:
            col_spec = 'l' + 'cc' * len(metrics)  # Method + (Mean, Std) for each metric
            lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
            lines.append('\\toprule')
            
            # Header row without confidence intervals
            header_parts = ['Method']
            for metric in metrics:
                metric_name = metric.replace('_', ' ').title()
                header_parts.extend([f'{metric_name}', 'Std Dev'])
            lines.append(' & '.join(header_parts) + ' \\\\')
        
        lines.append('\\midrule')
        
        # Data rows
        methods = list(method_results.keys())
        for method in methods:
            row_values = [method]
            
            for metric in metrics:
                method_data = method_results[method]
                
                if isinstance(method_data, dict):
                    mean = method_data.get('mean', method_data.get(metric, float('nan')))
                    std = method_data.get('std', 0.0)
                    n = method_data.get('n', 1)
                else:
                    mean = method_data
                    std = 0.0
                    n = 1
                
                # Format mean value
                if np.isnan(mean):
                    mean_str = '-'
                    std_str = '-'
                else:
                    if include_confidence_intervals and n > 1:
                        # Compute confidence interval
                        from scipy.stats import t
                        sem = std / np.sqrt(n)
                        t_val = t.ppf((1 + confidence_level) / 2, n - 1)
                        ci_margin = t_val * sem
                        
                        mean_str = f'{mean:.{self.precision}f} $\\pm$ {ci_margin:.{self.precision}f}'
                    else:
                        mean_str = f'{mean:.{self.precision}f}'
                    
                    std_str = f'{std:.{self.precision}f}'
                
                row_values.extend([mean_str, std_str])
            
            lines.append(' & '.join(row_values) + ' \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    
    def generate_method_ranking_table(
        self,
        comparison_results: Dict[str, Any],
        output_path: Union[str, Path],
        caption: str = 'Method Ranking',
        label: str = 'tab:ranking'
    ) -> str:
        """Generate LaTeX table showing method rankings.
        
        Args:
            comparison_results: Output from MethodComparator.compare_methods()
            output_path: Path to save the .tex file
            caption: Table caption
            label: LaTeX label
        
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append('\\begin{table}[htbp]')
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        lines.append('\\begin{tabular}{clcc}')
        lines.append('\\toprule')
        lines.append('Rank & Method & Mean & 95\\% CI \\\\')
        lines.append('\\midrule')
        
        # Get ranking information
        ranking = comparison_results.get('ranking', [])
        summary = comparison_results.get('summary', {})
        confidence_intervals = comparison_results.get('confidence_intervals', {})
        
        for rank, method_name in enumerate(ranking, 1):
            if method_name not in summary:
                continue
                
            method_stats = summary[method_name]
            mean = method_stats.get('mean', 0.0)
            
            # Get confidence interval
            ci_data = confidence_intervals.get(method_name, {})
            ci_lower = ci_data.get('ci_lower', mean)
            ci_upper = ci_data.get('ci_upper', mean)
            
            # Format values
            mean_str = f'{mean:.{self.precision}f}'
            ci_str = f'[{ci_lower:.{self.precision}f}, {ci_upper:.{self.precision}f}]'
            
            # Highlight best method
            if rank == 1:
                mean_str = f'\\textbf{{{mean_str}}}'
                method_name = f'\\textbf{{{method_name}}}'
            
            lines.append(f'{rank} & {method_name} & {mean_str} & {ci_str} \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    def generate_comprehensive_comparison_table(
        self,
        method_results: Dict[str, Dict[str, np.ndarray]],
        metrics: List[str],
        output_path: Union[str, Path],
        caption: str = 'Comprehensive Method Comparison',
        label: str = 'tab:comprehensive',
        alpha: float = 0.05,
        correction_method: str = 'holm'
    ) -> str:
        """Generate comprehensive comparison table with statistical testing.
        
        Combines method comparison with statistical significance testing,
        effect sizes, and confidence intervals in a single table.
        
        Args:
            method_results: Dict mapping method names to metric arrays
                           {method: {metric: [values]}}
            metrics: List of metric names to include
            output_path: Path to save the .tex file
            caption: Table caption
            label: LaTeX label
            alpha: Significance level for statistical tests
            correction_method: Multiple comparison correction method
        
        Returns:
            LaTeX table string
        """
        from src.evaluation import MethodComparator
        
        # Initialize comparator
        comparator = MethodComparator(alpha=alpha, correction_method=correction_method)
        
        lines = []
        lines.append('\\begin{table*}[htbp]')  # Use table* for wide tables
        lines.append('\\centering')
        lines.append(f'\\caption{{{caption}}}')
        lines.append(f'\\label{{{label}}}')
        
        # Column specification: Method + Mean±CI + Rank + Significance for each metric
        col_spec = 'l' + 'ccc' * len(metrics)
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append('\\toprule')
        
        # Header row
        header_parts = ['Method']
        for metric in metrics:
            metric_name = metric.replace('_', ' ').title()
            header_parts.extend([f'{metric_name}', 'Rank', 'Sig. Better'])
        lines.append(' & '.join(header_parts) + ' \\\\')
        lines.append('\\midrule')
        
        # Perform statistical comparisons for each metric
        metric_comparisons = {}
        for metric in metrics:
            if metric in method_results[list(method_results.keys())[0]]:
                metric_data = {method: results[metric] for method, results in method_results.items()}
                metric_comparisons[metric] = comparator.compare_methods(metric_data, metric)
        
        # Generate rows for each method
        methods = list(method_results.keys())
        for method in methods:
            row_values = [method]
            
            for metric in metrics:
                if metric not in metric_comparisons:
                    row_values.extend(['-', '-', '-'])
                    continue
                
                comparison = metric_comparisons[metric]
                summary = comparison.get('summary', {})
                ranking = comparison.get('ranking', [])
                pairwise_tests = comparison.get('pairwise_tests', {})
                confidence_intervals = comparison.get('confidence_intervals', {})
                
                # Get method statistics
                method_stats = summary.get(method, {})
                mean = method_stats.get('mean', 0.0)
                
                # Get confidence interval
                ci_data = confidence_intervals.get(method, {})
                ci_lower = ci_data.get('ci_lower', mean)
                ci_upper = ci_data.get('ci_upper', mean)
                
                # Format mean with CI
                mean_ci_str = f'{mean:.{self.precision}f} [{ci_lower:.{self.precision}f}, {ci_upper:.{self.precision}f}]'
                
                # Get rank
                try:
                    rank = ranking.index(method) + 1
                    rank_str = str(rank)
                    if rank == 1:
                        mean_ci_str = f'\\textbf{{{mean_ci_str}}}'
                        rank_str = f'\\textbf{{{rank_str}}}'
                except ValueError:
                    rank_str = '-'
                
                # Count significant wins
                sig_wins = 0
                for comp_key, test_result in pairwise_tests.items():
                    if method in comp_key and test_result.get('significant', False):
                        # Check if this method is better
                        other_method = comp_key.replace(f'{method}_vs_', '').replace(f'_vs_{method}', '')
                        if other_method in summary:
                            other_mean = summary[other_method].get('mean', 0.0)
                            # Assume lower is better for most metrics
                            is_error_metric = any(
                                err in metric.lower()
                                for err in ['mae', 'mse', 'error', 'loss', 'mfd']
                            )
                            if (is_error_metric and mean < other_mean) or (not is_error_metric and mean > other_mean):
                                sig_wins += 1
                
                sig_str = f'{sig_wins}/{len(methods)-1}' if len(methods) > 1 else '-'
                
                row_values.extend([mean_ci_str, rank_str, sig_str])
            
            lines.append(' & '.join(row_values) + ' \\\\')
        
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        
        # Add footnote explaining significance
        lines.append(f'\\begin{{tablenotes}}')
        lines.append(f'\\small')
        lines.append(f'\\item Note: Sig. Better shows number of methods this method significantly outperforms.')
        lines.append(f'\\item Statistical significance determined using {correction_method} correction at $\\alpha = {alpha}$.')
        lines.append(f'\\end{{tablenotes}}')
        
        lines.append('\\end{table*}')
        
        latex_str = '\n'.join(lines)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        return latex_str
    
    def format_pvalue(self, p_value: float) -> str:
        """Format p-value for LaTeX display with significance indicators.
        
        Args:
            p_value: Raw p-value
        
        Returns:
            Formatted p-value string with significance indicators
        """
        if p_value < 0.001:
            return '$<$0.001***'
        elif p_value < 0.01:
            return f'{p_value:.3f}**'
        elif p_value < 0.05:
            return f'{p_value:.3f}*'
        else:
            return f'{p_value:.3f}'
    
    def format_effect_size(self, cohens_d: float) -> str:
        """Format Cohen's d effect size with interpretation.
        
        Args:
            cohens_d: Cohen's d value
        
        Returns:
            Formatted effect size string
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            symbol = ''
        elif abs_d < 0.5:
            symbol = '$^\\dagger$'  # Small effect
        elif abs_d < 0.8:
            symbol = '$^\\ddagger$'  # Medium effect
        else:
            symbol = '$^\\S$'  # Large effect
        
        return f'{cohens_d:.3f}{symbol}'
    
    def add_significance_legend(self, lines: List[str]) -> None:
        """Add significance legend to table footnotes.
        
        Args:
            lines: List of LaTeX lines to append legend to
        """
        lines.append('\\begin{tablenotes}')
        lines.append('\\small')
        lines.append('\\item * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$')
        lines.append('\\item Effect sizes: $^\\dagger$ small ($|d| \\geq 0.2$), $^\\ddagger$ medium ($|d| \\geq 0.5$), $^\\S$ large ($|d| \\geq 0.8$)')
        lines.append('\\end{tablenotes}')