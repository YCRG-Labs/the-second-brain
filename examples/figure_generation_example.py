#!/usr/bin/env python3
"""
Figure Generation Example

This example demonstrates how to use the FigureGenerator and LaTeXTableGenerator
to create publication-quality visualizations and tables.

Usage:
    python examples/figure_generation_example.py
    python examples/figure_generation_example.py --example plots
    python examples/figure_generation_example.py --example tables
    python examples/figure_generation_example.py --example advanced
"""

import argparse
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
import sys
sys.path.append('src')

from figures import FigureGenerator, LaTeXTableGenerator
from evaluation import shannon_entropy, alpha_diversity, beta_diversity


def create_sample_data(seed=42):
    """Create sample data for figure generation."""
    np.random.seed(seed)
    
    # Real and generated compositions
    real_compositions = np.random.dirichlet(np.ones(50) * 2, size=200)
    our_generated = np.random.dirichlet(np.ones(50) * 1.8, size=200)
    vae_generated = np.random.dirichlet(np.ones(50) * 1.5, size=200)
    gan_generated = np.random.dirichlet(np.ones(50) * 1.3, size=200)
    
    # Diversity metrics
    real_alpha = alpha_diversity(real_compositions)
    our_alpha = alpha_diversity(our_generated)
    vae_alpha = alpha_diversity(vae_generated)
    gan_alpha = alpha_diversity(gan_generated)
    
    real_beta = beta_diversity(real_compositions)
    our_beta = beta_diversity(our_generated)
    vae_beta = beta_diversity(vae_generated)
    gan_beta = beta_diversity(gan_generated)
    
    # Method comparison data
    method_data = {
        'Our Method': {
            'mfd': 0.123,
            'mfd_std': 0.015,
            'alpha_div': our_alpha.mean(),
            'alpha_std': our_alpha.std(),
            'beta_div': our_beta.mean(),
            'beta_std': our_beta.std(),
            'compositions': our_generated
        },
        'VAE': {
            'mfd': 0.187,
            'mfd_std': 0.023,
            'alpha_div': vae_alpha.mean(),
            'alpha_std': vae_alpha.std(),
            'beta_div': vae_beta.mean(),
            'beta_std': vae_beta.std(),
            'compositions': vae_generated
        },
        'GAN': {
            'mfd': 0.201,
            'mfd_std': 0.031,
            'alpha_div': gan_alpha.mean(),
            'alpha_std': gan_alpha.std(),
            'beta_div': gan_beta.mean(),
            'beta_std': gan_beta.std(),
            'compositions': gan_generated
        }
    }
    
    # Prediction data
    horizons = [1, 2, 3, 4, 5]
    prediction_accuracies = {
        'Our Method': [0.85, 0.78, 0.72, 0.68, 0.65],
        'LSTM': [0.82, 0.74, 0.67, 0.61, 0.58],
        'ARIMA': [0.79, 0.71, 0.63, 0.56, 0.52]
    }
    
    # Ablation study data
    ablation_results = {
        'Full Model': 0.123,
        'No Hyperbolic': 0.145,
        'No CLR': 0.167,
        'No Diversity Loss': 0.189,
        'No Co-exclusion': 0.201
    }
    
    # Sensitivity analysis data
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    batch_sizes = [16, 32, 64, 128, 256]
    performance_grid = np.random.rand(len(learning_rates), len(batch_sizes))
    
    return {
        'real_compositions': real_compositions,
        'method_data': method_data,
        'real_alpha': real_alpha,
        'real_beta': real_beta,
        'prediction_accuracies': prediction_accuracies,
        'horizons': horizons,
        'ablation_results': ablation_results,
        'learning_rates': learning_rates,
        'batch_sizes': batch_sizes,
        'performance_grid': performance_grid
    }


def basic_figure_example():
    """Demonstrate basic figure generation."""
    print("=" * 60)
    print("BASIC FIGURE GENERATION EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('examples/figure_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Initializing Figure Generator...")
    
    # Initialize figure generator with publication settings
    fig_gen = FigureGenerator(
        style='seaborn-v0_8-paper',
        dpi=300,
        figsize_single=(4.5, 3.5),
        figsize_double=(9.0, 3.5),
        font_size=10,
        output_format='pdf'
    )
    
    print("2. Generating MFD Comparison Plot...")
    
    # MFD comparison
    mfd_data = {
        method: {'mfd': info['mfd'], 'std': info['mfd_std']}
        for method, info in data['method_data'].items()
    }
    
    mfd_path = fig_gen.plot_mfd_comparison(
        method_data=mfd_data,
        output_path=output_dir / 'mfd_comparison.pdf',
        title='Microbiome Fréchet Distance Comparison'
    )
    print(f"  Saved: {mfd_path}")
    
    print("3. Generating Diversity Distribution Plots...")
    
    # Alpha diversity comparison
    for method, info in data['method_data'].items():
        gen_alpha = alpha_diversity(info['compositions'])
        
        alpha_path = fig_gen.plot_diversity_distributions(
            real_diversity=data['real_alpha'],
            generated_diversity=gen_alpha,
            output_path=output_dir / f'alpha_diversity_{method.lower().replace(" ", "_")}.pdf',
            diversity_type='alpha',
            method_name=method,
            statistical_test=True
        )
        print(f"  Saved: {alpha_path}")
    
    print("4. Generating t-SNE Comparison...")
    
    # t-SNE visualization (using first method as example)
    our_compositions = data['method_data']['Our Method']['compositions']
    
    tsne_path = fig_gen.plot_tsne_comparison(
        real_compositions=data['real_compositions'],
        generated_compositions=our_compositions,
        output_path=output_dir / 'tsne_comparison.pdf',
        title='t-SNE: Real vs Generated Compositions',
        method_name='Our Method'
    )
    print(f"  Saved: {tsne_path}")
    
    print("5. Generating Method Comparison Summary...")
    
    # Create comprehensive comparison plot
    comparison_path = fig_gen.plot_method_comparison_summary(
        method_data=data['method_data'],
        metrics=['mfd', 'alpha_div', 'beta_div'],
        output_path=output_dir / 'method_comparison_summary.pdf',
        title='Method Comparison Summary'
    )
    print(f"  Saved: {comparison_path}")
    
    print(f"\nBasic figures saved to: {output_dir}")


def plot_generation_example():
    """Demonstrate various plot types."""
    print("=" * 60)
    print("PLOT GENERATION EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('examples/figure_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize figure generator
    fig_gen = FigureGenerator(dpi=300, output_format='png')
    
    print("1. Prediction Accuracy Plots...")
    
    # Prediction accuracy vs horizon
    pred_path = fig_gen.plot_prediction_accuracy(
        horizons=data['horizons'],
        method_accuracies=data['prediction_accuracies'],
        output_path=output_dir / 'prediction_accuracy.png',
        title='Prediction Accuracy vs Time Horizon'
    )
    print(f"  Saved: {pred_path}")
    
    # Error distributions
    prediction_errors = {
        method: np.random.gamma(1 + 0.2 * i, 0.1, 1000)
        for i, method in enumerate(data['prediction_accuracies'].keys())
    }
    
    error_path = fig_gen.plot_error_distributions(
        prediction_errors=prediction_errors,
        output_path=output_dir / 'error_distributions.png',
        title='Prediction Error Distributions'
    )
    print(f"  Saved: {error_path}")
    
    print("2. Ablation Study Plot...")
    
    # Ablation study
    ablation_path = fig_gen.plot_ablation_study(
        ablation_results=data['ablation_results'],
        output_path=output_dir / 'ablation_study.png',
        title='Component Contribution Analysis',
        metric_name='MFD'
    )
    print(f"  Saved: {ablation_path}")
    
    print("3. Sensitivity Analysis Heatmap...")
    
    # Sensitivity heatmap
    sensitivity_path = fig_gen.plot_sensitivity_heatmap(
        x_values=data['batch_sizes'],
        y_values=data['learning_rates'],
        performance_grid=data['performance_grid'],
        x_label='Batch Size',
        y_label='Learning Rate',
        output_path=output_dir / 'sensitivity_heatmap.png',
        title='Hyperparameter Sensitivity Analysis'
    )
    print(f"  Saved: {sensitivity_path}")
    
    print("4. Multi-Panel Figure...")
    
    # Create multi-panel figure
    multi_panel_path = fig_gen.create_multi_panel_figure(
        panels=[
            {
                'type': 'mfd_comparison',
                'data': {method: {'mfd': info['mfd'], 'std': info['mfd_std']} 
                        for method, info in data['method_data'].items()},
                'title': 'MFD Comparison'
            },
            {
                'type': 'ablation_study',
                'data': data['ablation_results'],
                'title': 'Ablation Study'
            },
            {
                'type': 'prediction_accuracy',
                'data': {
                    'horizons': data['horizons'],
                    'accuracies': data['prediction_accuracies']
                },
                'title': 'Prediction Accuracy'
            }
        ],
        output_path=output_dir / 'multi_panel_figure.png',
        title='Comprehensive Results Overview'
    )
    print(f"  Saved: {multi_panel_path}")
    
    print("5. Custom Styling Example...")
    
    # Custom color scheme
    custom_colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'quaternary': '#d62728'
    }
    
    custom_fig_gen = FigureGenerator(
        color_scheme=custom_colors,
        font_size=12,
        figsize_single=(6, 4)
    )
    
    custom_path = custom_fig_gen.plot_mfd_comparison(
        method_data={method: {'mfd': info['mfd'], 'std': info['mfd_std']} 
                    for method, info in data['method_data'].items()},
        output_path=output_dir / 'custom_styled_mfd.png',
        title='Custom Styled MFD Comparison'
    )
    print(f"  Saved: {custom_path}")
    
    print(f"\nPlot examples saved to: {output_dir}")


def table_generation_example():
    """Demonstrate LaTeX table generation."""
    print("=" * 60)
    print("LATEX TABLE GENERATION EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('examples/table_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Initializing Table Generator...")
    
    # Initialize table generator
    table_gen = LaTeXTableGenerator(
        precision=3,
        use_booktabs=True,
        caption_position='top'
    )
    
    print("2. Method Comparison Table...")
    
    # Prepare method results for table
    method_results = {}
    for method, info in data['method_data'].items():
        method_results[method] = {
            'MFD': (info['mfd'], info['mfd_std']),
            'Alpha Diversity': (info['alpha_div'], info['alpha_std']),
            'Beta Diversity': (info['beta_div'], info['beta_std'])
        }
    
    # Generate method comparison table
    comparison_table = table_gen.generate_method_comparison_table(
        method_results=method_results,
        caption='Quantitative comparison of microbiome generation methods',
        label='tab:method_comparison',
        include_std=True,
        highlight_best=True
    )
    
    # Save table
    with open(output_dir / 'method_comparison.tex', 'w') as f:
        f.write(comparison_table)
    
    print(f"  Saved: {output_dir / 'method_comparison.tex'}")
    
    print("3. Statistical Significance Table...")
    
    # Create statistical significance results
    significance_results = {
        ('Our Method', 'VAE'): {
            'p_value': 0.001,
            'effect_size': 0.8,
            'significant': True,
            'test_statistic': 3.45
        },
        ('Our Method', 'GAN'): {
            'p_value': 0.023,
            'effect_size': 0.6,
            'significant': True,
            'test_statistic': 2.31
        },
        ('VAE', 'GAN'): {
            'p_value': 0.156,
            'effect_size': 0.3,
            'significant': False,
            'test_statistic': 1.42
        }
    }
    
    # Generate statistical significance table
    significance_table = table_gen.generate_statistical_significance_table(
        significance_results=significance_results,
        caption='Statistical significance of pairwise method comparisons',
        label='tab:statistical_tests',
        include_effect_size=True,
        correction_method='Holm'
    )
    
    # Save table
    with open(output_dir / 'statistical_significance.tex', 'w') as f:
        f.write(significance_table)
    
    print(f"  Saved: {output_dir / 'statistical_significance.tex'}")
    
    print("4. Ablation Study Table...")
    
    # Generate ablation study table
    ablation_table = table_gen.generate_ablation_study_table(
        ablation_results=data['ablation_results'],
        caption='Ablation study results showing component contributions',
        label='tab:ablation_study',
        metric_name='MFD',
        highlight_best=True
    )
    
    # Save table
    with open(output_dir / 'ablation_study.tex', 'w') as f:
        f.write(ablation_table)
    
    print(f"  Saved: {output_dir / 'ablation_study.tex'}")
    
    print("5. Prediction Metrics Table...")
    
    # Create prediction metrics data
    prediction_metrics = {
        'Our Method': {
            'MAE': (0.045, 0.008),
            'Top-5 Acc': (0.82, 0.04),
            'Top-10 Acc': (0.91, 0.03),
            'Top-20 Acc': (0.96, 0.02)
        },
        'LSTM': {
            'MAE': (0.067, 0.012),
            'Top-5 Acc': (0.75, 0.05),
            'Top-10 Acc': (0.86, 0.04),
            'Top-20 Acc': (0.93, 0.03)
        },
        'ARIMA': {
            'MAE': (0.089, 0.015),
            'Top-5 Acc': (0.68, 0.06),
            'Top-10 Acc': (0.79, 0.05),
            'Top-20 Acc': (0.88, 0.04)
        }
    }
    
    # Generate prediction metrics table
    prediction_table = table_gen.generate_method_comparison_table(
        method_results=prediction_metrics,
        caption='Prediction accuracy metrics across different time horizons',
        label='tab:prediction_metrics',
        include_std=True,
        highlight_best=True
    )
    
    # Save table
    with open(output_dir / 'prediction_metrics.tex', 'w') as f:
        f.write(prediction_table)
    
    print(f"  Saved: {output_dir / 'prediction_metrics.tex'}")
    
    print("6. Custom Table Formatting...")
    
    # Custom table generator with different settings
    custom_table_gen = LaTeXTableGenerator(
        precision=4,
        use_booktabs=False,
        caption_position='bottom',
        table_position='htbp'
    )
    
    # Generate custom formatted table
    custom_table = custom_table_gen.generate_method_comparison_table(
        method_results=method_results,
        caption='Custom formatted method comparison table',
        label='tab:custom_format',
        include_std=False,
        highlight_best=False
    )
    
    # Save table
    with open(output_dir / 'custom_format.tex', 'w') as f:
        f.write(custom_table)
    
    print(f"  Saved: {output_dir / 'custom_format.tex'}")
    
    print(f"\nLaTeX tables saved to: {output_dir}")
    print("To use these tables in your LaTeX document:")
    print("1. Include \\usepackage{booktabs} in your preamble (for booktabs tables)")
    print("2. Use \\input{table_file.tex} to include the table")


def advanced_figure_example():
    """Demonstrate advanced figure generation features."""
    print("=" * 60)
    print("ADVANCED FIGURE GENERATION EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('examples/figure_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Multi-Format Output...")
    
    # Initialize figure generator
    fig_gen = FigureGenerator(dpi=300)
    
    # Generate figure in multiple formats
    base_path = output_dir / 'multi_format_mfd'
    mfd_data = {
        method: {'mfd': info['mfd'], 'std': info['mfd_std']}
        for method, info in data['method_data'].items()
    }
    
    # PDF for publication
    pdf_path = fig_gen.plot_mfd_comparison(
        method_data=mfd_data,
        output_path=f'{base_path}.pdf',
        title='MFD Comparison (PDF)'
    )
    
    # PNG for presentations
    png_path = fig_gen.plot_mfd_comparison(
        method_data=mfd_data,
        output_path=f'{base_path}.png',
        title='MFD Comparison (PNG)'
    )
    
    # SVG for web
    svg_path = fig_gen.plot_mfd_comparison(
        method_data=mfd_data,
        output_path=f'{base_path}.svg',
        title='MFD Comparison (SVG)'
    )
    
    print(f"  Generated multi-format outputs:")
    print(f"    PDF: {pdf_path}")
    print(f"    PNG: {png_path}")
    print(f"    SVG: {svg_path}")
    
    print("2. Statistical Annotations...")
    
    # Generate plot with statistical annotations
    stat_path = fig_gen.plot_diversity_distributions(
        real_diversity=data['real_alpha'],
        generated_diversity=alpha_diversity(data['method_data']['Our Method']['compositions']),
        output_path=output_dir / 'statistical_annotations.pdf',
        diversity_type='alpha',
        method_name='Our Method',
        statistical_test=True,
        show_p_value=True,
        show_effect_size=True
    )
    print(f"  Saved statistical plot: {stat_path}")
    
    print("3. Publication-Ready Styling...")
    
    # High-quality publication figure
    pub_fig_gen = FigureGenerator(
        style='seaborn-v0_8-paper',
        dpi=600,  # High DPI for publication
        figsize_single=(3.5, 2.8),  # Journal column width
        font_size=8,  # Small font for publication
        output_format='pdf'
    )
    
    pub_path = pub_fig_gen.plot_method_comparison_summary(
        method_data=data['method_data'],
        metrics=['mfd', 'alpha_div'],
        output_path=output_dir / 'publication_ready.pdf',
        title='',  # No title for publication
        show_legend=True,
        tight_layout=True
    )
    print(f"  Saved publication figure: {pub_path}")
    
    print("4. Interactive Elements...")
    
    # Generate figure with interactive elements (saved as static)
    interactive_path = fig_gen.plot_sensitivity_heatmap(
        x_values=data['batch_sizes'],
        y_values=data['learning_rates'],
        performance_grid=data['performance_grid'],
        x_label='Batch Size',
        y_label='Learning Rate',
        output_path=output_dir / 'interactive_heatmap.pdf',
        title='Interactive Sensitivity Analysis',
        show_values=True,
        colorbar_label='Performance Score'
    )
    print(f"  Saved interactive heatmap: {interactive_path}")
    
    print("5. Comprehensive Figure Summary...")
    
    # Create a comprehensive summary document
    summary_data = {
        'method_comparison': {
            'data': data['method_data'],
            'best_method': min(data['method_data'].items(), key=lambda x: x[1]['mfd'])[0]
        },
        'prediction_performance': {
            'data': data['prediction_accuracies'],
            'best_horizon': data['horizons'][0]  # First horizon typically best
        },
        'ablation_insights': {
            'data': data['ablation_results'],
            'most_important': min(data['ablation_results'].items(), key=lambda x: x[1])[0]
        }
    }
    
    # Save summary
    with open(output_dir / 'figure_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"  Saved figure summary: {output_dir / 'figure_summary.json'}")
    
    print("6. Batch Figure Generation...")
    
    # Generate multiple figures in batch
    batch_configs = [
        {
            'type': 'mfd_comparison',
            'data': mfd_data,
            'output': 'batch_mfd.pdf',
            'title': 'Batch MFD Comparison'
        },
        {
            'type': 'ablation_study',
            'data': data['ablation_results'],
            'output': 'batch_ablation.pdf',
            'title': 'Batch Ablation Study'
        }
    ]
    
    batch_paths = fig_gen.generate_figure_batch(
        configs=batch_configs,
        output_dir=output_dir
    )
    
    print(f"  Generated batch figures:")
    for path in batch_paths:
        print(f"    {path}")
    
    print(f"\nAdvanced figures saved to: {output_dir}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description='Figure Generation Examples')
    parser.add_argument(
        '--example',
        choices=['basic', 'plots', 'tables', 'advanced'],
        default='basic',
        help='Which example to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    print("Microbiome Figure Generation Examples")
    print("=" * 60)
    print(f"Running example: {args.example}")
    print()
    
    try:
        if args.example == 'basic':
            basic_figure_example()
        elif args.example == 'plots':
            plot_generation_example()
        elif args.example == 'tables':
            table_generation_example()
        elif args.example == 'advanced':
            advanced_figure_example()
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())