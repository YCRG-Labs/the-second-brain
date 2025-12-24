#!/usr/bin/env python3
"""
Complete Publication Pipeline Example

This example demonstrates the end-to-end publication pipeline using all
publication-ready components, including:
- Comprehensive evaluation with biological validation
- Performance benchmarking
- Statistical method comparison
- Publication-quality figure generation
- LaTeX table generation
- Complete publication package compilation

Requirements: All requirements from publication-ready spec
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import warnings

# Import all publication components
from src.evaluation import MicrobiomeEvaluator, BiologicalValidator, MethodComparator
from src.benchmarking import BenchmarkSuite, MemoryProfiler
from src.figures import FigureGenerator, LaTeXTableGenerator
from src.publication import PublicationCompiler, PublicationPackage


class MicrobiomeGenerationModel(nn.Module):
    """Example microbiome generation model for the pipeline."""
    
    def __init__(self, num_taxa=100, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, batch_size=None, z=None):
        if z is None:
            if batch_size is None:
                batch_size = 32
            z = torch.randn(batch_size, self.latent_dim)
        
        return self.generator(z)


class VAEBaseline(nn.Module):
    """VAE baseline model."""
    
    def __init__(self, num_taxa=100, latent_dim=32):
        super().__init__()
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, batch_size=None, z=None):
        if z is None:
            if batch_size is None:
                batch_size = 32
            z = torch.randn(batch_size, self.latent_dim)
        
        return self.decoder(z)


class GANBaseline(nn.Module):
    """GAN baseline model."""
    
    def __init__(self, num_taxa=100, latent_dim=128):
        super().__init__()
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, batch_size=None, z=None):
        if z is None:
            if batch_size is None:
                batch_size = 32
            z = torch.randn(batch_size, self.latent_dim)
        
        return self.generator(z)


def create_realistic_microbiome_data(num_samples=500, num_taxa=100, seed=42):
    """Create realistic microbiome data for the pipeline."""
    np.random.seed(seed)
    
    # Create realistic abundance distributions
    # Use different concentration parameters to simulate diversity
    
    # Real microbiome data (high diversity, realistic distribution)
    alpha_real = np.ones(num_taxa) * 0.1
    alpha_real[:20] = 2.0  # Some dominant taxa
    alpha_real[20:50] = 0.5  # Moderate abundance taxa
    # Remaining taxa have low abundance (alpha = 0.1)
    
    real_compositions = np.random.dirichlet(alpha_real, size=num_samples)
    
    return real_compositions


def generate_method_data(real_compositions, models, device, num_samples=500):
    """Generate compositions from different methods."""
    method_data = {}
    
    # Add real data
    method_data['Real Data'] = {
        'real_compositions': real_compositions,
        'generated_compositions': real_compositions  # For reference
    }
    
    # Generate from each model
    for method_name, model in models.items():
        model.eval()
        generated_compositions = []
        
        batch_size = 64
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                if current_batch_size > 0:
                    batch_samples = model(batch_size=current_batch_size)
                    generated_compositions.append(batch_samples.cpu().numpy())
        
        generated_compositions = np.vstack(generated_compositions)[:num_samples]
        
        method_data[method_name] = {
            'real_compositions': real_compositions,
            'generated_compositions': generated_compositions
        }
    
    return method_data


def create_phylogenetic_kernel(num_taxa=100, seed=42):
    """Create a realistic phylogenetic kernel matrix."""
    np.random.seed(seed)
    
    # Create phylogenetic groups
    group_sizes = [25, 30, 25, 20]  # Four major phyla
    kernel = np.zeros((num_taxa, num_taxa))
    
    start_idx = 0
    for group_size in group_sizes:
        end_idx = start_idx + group_size
        
        # Within-group similarity (higher for closely related taxa)
        group_kernel = np.random.uniform(0.7, 1.0, (group_size, group_size))
        group_kernel = (group_kernel + group_kernel.T) / 2
        np.fill_diagonal(group_kernel, 1.0)
        
        kernel[start_idx:end_idx, start_idx:end_idx] = group_kernel
        start_idx = end_idx
    
    # Between-group similarity (lower for distantly related taxa)
    for i in range(len(group_sizes)):
        for j in range(i+1, len(group_sizes)):
            start_i = sum(group_sizes[:i])
            end_i = sum(group_sizes[:i+1])
            start_j = sum(group_sizes[:j])
            end_j = sum(group_sizes[:j+1])
            
            between_similarity = np.random.uniform(0.1, 0.4, (end_i - start_i, end_j - start_j))
            kernel[start_i:end_i, start_j:end_j] = between_similarity
            kernel[start_j:end_j, start_i:end_i] = between_similarity.T
    
    return kernel


def setup_biological_constraints():
    """Set up comprehensive biological constraints."""
    
    # Co-exclusion pairs (competing species)
    co_exclusion_pairs = [
        (0, 1),   # Bacteroides vs Prevotella
        (5, 6),   # Aerobic vs anaerobic
        (10, 11), # pH competitors
        (15, 16), # Nutrient competitors
        (20, 21), # Metabolic competitors
        (25, 26), # Oxygen tolerance
        (30, 31), # Temperature preference
        (35, 36)  # Salinity preference
    ]
    
    # Metabolic pathways
    metabolic_pathways = {
        'glycolysis': [2, 3, 4, 5, 7, 8],
        'nitrogen_fixation': [40, 41, 42, 43, 44],
        'methanogenesis': [45, 46, 47, 48, 49, 50],
        'sulfate_reduction': [51, 52, 53, 54, 55],
        'butyrate_production': [56, 57, 58, 59, 60],
        'acetate_production': [61, 62, 63, 64],
        'propionate_production': [65, 66, 67, 68],
        'lactate_fermentation': [69, 70, 71, 72]
    }
    
    # Disease biomarkers
    disease_biomarkers = {
        'inflammatory_bowel_disease': [12, 13, 14, 17],
        'type2_diabetes': [18, 19, 22, 23, 24],
        'obesity': [27, 28, 29, 32, 33],
        'colorectal_cancer': [73, 74, 75, 76, 77],
        'liver_disease': [78, 79, 80, 81],
        'cardiovascular_disease': [82, 83, 84, 85],
        'depression': [86, 87, 88],
        'anxiety': [89, 90, 91]
    }
    
    # Phylogenetic groups
    phylogenetic_groups = {
        'firmicutes': list(range(0, 25)),
        'bacteroidetes': list(range(25, 55)),
        'proteobacteria': list(range(55, 80)),
        'actinobacteria': list(range(80, 100))
    }
    
    return BiologicalValidator(
        co_exclusion_pairs=co_exclusion_pairs,
        metabolic_pathways=metabolic_pathways,
        disease_biomarkers=disease_biomarkers,
        phylogenetic_groups=phylogenetic_groups
    )


def run_complete_pipeline():
    """Run the complete publication pipeline."""
    
    print("Complete Publication Pipeline Example")
    print("=" * 50)
    
    # 1. Setup
    print("\n1. Setting up models and data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    num_taxa = 100
    num_samples = 500
    
    # Create models
    our_model = MicrobiomeGenerationModel(num_taxa=num_taxa, latent_dim=64)
    vae_baseline = VAEBaseline(num_taxa=num_taxa, latent_dim=32)
    gan_baseline = GANBaseline(num_taxa=num_taxa, latent_dim=128)
    
    models = {
        'Our Method': our_model.to(device),
        'VAE Baseline': vae_baseline.to(device),
        'GAN Baseline': gan_baseline.to(device)
    }
    
    # Create data
    real_compositions = create_realistic_microbiome_data(
        num_samples=num_samples, 
        num_taxa=num_taxa
    )
    
    method_data = generate_method_data(
        real_compositions=real_compositions,
        models=models,
        device=device,
        num_samples=num_samples
    )
    
    print(f"   Created {len(method_data)} method datasets")
    print(f"   Each dataset has {num_samples} samples with {num_taxa} taxa")
    
    # 2. Create phylogenetic kernel and biological constraints
    print("\n2. Setting up biological constraints...")
    
    phylogenetic_kernel = create_phylogenetic_kernel(num_taxa=num_taxa)
    biological_validator = setup_biological_constraints()
    
    print(f"   Created phylogenetic kernel: {phylogenetic_kernel.shape}")
    print(f"   Set up biological validator with:")
    print(f"     - {len(biological_validator.co_exclusion_pairs)} co-exclusion pairs")
    print(f"     - {len(biological_validator.metabolic_pathways)} metabolic pathways")
    print(f"     - {len(biological_validator.disease_biomarkers)} disease biomarkers")
    
    # 3. Initialize publication compiler
    print("\n3. Initializing publication compiler...")
    
    output_dir = Path('complete_publication_output')
    
    # Create custom components
    figure_generator = FigureGenerator(
        style='seaborn-v0_8-paper',
        dpi=300,
        figsize_single=(4.5, 3.5),
        font_size=10,
        output_format='pdf'
    )
    
    table_generator = LaTeXTableGenerator(
        precision=4,
        use_booktabs=True,
        caption_position='top'
    )
    
    method_comparator = MethodComparator(
        alpha=0.05,
        correction_method='bonferroni',
        effect_size_method='cohens_d'
    )
    
    benchmark_suite = BenchmarkSuite(device=device)
    
    # Initialize publication compiler
    compiler = PublicationCompiler(
        output_dir=output_dir,
        phylogenetic_kernel=phylogenetic_kernel,
        figure_generator=figure_generator,
        table_generator=table_generator,
        method_comparator=method_comparator,
        biological_validator=biological_validator,
        benchmark_suite=benchmark_suite
    )
    
    print(f"   Publication compiler initialized")
    print(f"   Output directory: {output_dir}")
    
    # 4. Compile publication package
    print("\n4. Compiling publication package...")
    
    # Remove 'Real Data' from method_data for evaluation (it's just a reference)
    evaluation_method_data = {k: v for k, v in method_data.items() if k != 'Real Data'}
    
    package = compiler.compile_publication_package(
        method_data=evaluation_method_data,
        project_name='Advanced Microbiome Generation Study',
        include_biological_validation=True,
        include_statistical_tests=True,
        include_benchmarking=True,
        figure_formats=['pdf', 'png'],
        k_values=[5, 10, 20],
        confidence_level=0.95
    )
    
    print(f"   Publication package compiled successfully!")
    print(f"   Generated {len(package.figures)} figures")
    print(f"   Generated {len(package.tables)} tables")
    
    # 5. Display results summary
    print("\n5. Results Summary:")
    print("-" * 20)
    
    results = package.results_summary
    
    # Method performance
    if 'method_metrics' in results:
        print("\nMethod Performance (MFD - lower is better):")
        method_metrics = results['method_metrics']
        
        # Sort by MFD
        sorted_methods = sorted(method_metrics.items(), key=lambda x: x[1]['mfd'])
        
        for method, metrics in sorted_methods:
            print(f"  {method:15s}: MFD = {metrics['mfd']:.4f}")
        
        best_method = sorted_methods[0][0]
        print(f"\nBest performing method: {best_method}")
    
    # Statistical significance
    if 'statistical_significance' in package.statistical_significance:
        print("\nStatistical Significance:")
        sig_tests = package.statistical_significance
        
        significant_comparisons = []
        for comparison, result in sig_tests.items():
            if result.get('significant', False):
                p_val = result.get('p_value', 1.0)
                effect_size = result.get('effect_size', 0.0)
                print(f"  {comparison}: p={p_val:.4f}, effect_size={effect_size:.3f}")
                significant_comparisons.append(comparison)
        
        if significant_comparisons:
            print(f"\nFound {len(significant_comparisons)} significant improvements")
        else:
            print("\nNo statistically significant differences found")
    
    # Biological validation
    if 'biological_validation' in results:
        print("\nBiological Validation:")
        bio_results = results['biological_validation']
        
        for method, bio_metrics in bio_results.items():
            if isinstance(bio_metrics, dict):
                plausibility = bio_metrics.get('overall_biological_plausibility', 0)
                violations = bio_metrics.get('co_exclusion_violations', 0)
                print(f"  {method:15s}: Plausibility = {plausibility:.3f}, Violations = {violations}")
    
    # 6. Generate validation report
    print("\n6. Generating validation report...")
    
    validation_report = compiler.generate_validation_report(package)
    print(f"   Validation report saved to: {validation_report}")
    
    # 7. Save additional outputs
    print("\n7. Saving additional outputs...")
    
    # Save package metadata
    metadata_file = output_dir / 'package_metadata.json'
    
    metadata = {
        'project_name': 'Advanced Microbiome Generation Study',
        'num_methods': len(evaluation_method_data),
        'num_samples': num_samples,
        'num_taxa': num_taxa,
        'figures_generated': list(package.figures.keys()),
        'tables_generated': list(package.tables.keys()),
        'best_method': best_method if 'method_metrics' in results else 'Unknown',
        'significant_comparisons': len(significant_comparisons) if 'statistical_significance' in package.statistical_significance else 0
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Package metadata saved to: {metadata_file}")
    
    # Create README for the publication package
    readme_file = output_dir / 'README.md'
    
    with open(readme_file, 'w') as f:
        f.write("# Advanced Microbiome Generation Study - Publication Package\n\n")
        f.write("This directory contains all publication-ready outputs from the microbiome generation study.\n\n")
        
        f.write("## Contents\n\n")
        f.write("### Figures\n")
        for fig_name, fig_path in package.figures.items():
            f.write(f"- `{fig_path.name}`: {fig_name}\n")
        
        f.write("\n### Tables\n")
        for table_name in package.tables.keys():
            f.write(f"- `{table_name}.tex`: LaTeX table for {table_name}\n")
        
        f.write("\n### Data Files\n")
        f.write("- `package_metadata.json`: Study metadata and summary statistics\n")
        f.write("- `validation_report.txt`: Comprehensive validation report\n")
        f.write("- `evaluation_results.json`: Detailed evaluation results\n")
        
        f.write(f"\n## Summary\n\n")
        f.write(f"- **Best Method**: {best_method if 'method_metrics' in results else 'Unknown'}\n")
        f.write(f"- **Methods Compared**: {len(evaluation_method_data)}\n")
        f.write(f"- **Samples per Method**: {num_samples}\n")
        f.write(f"- **Taxa Analyzed**: {num_taxa}\n")
        
        if 'statistical_significance' in package.statistical_significance:
            f.write(f"- **Significant Improvements**: {len(significant_comparisons)}\n")
        
        f.write("\n## Usage\n\n")
        f.write("1. Include figures in your manuscript using the PDF versions\n")
        f.write("2. Copy LaTeX table code from .tex files into your document\n")
        f.write("3. Reference the validation report for biological plausibility analysis\n")
        f.write("4. Use the metadata file for summary statistics in your paper\n")
    
    print(f"   README saved to: {readme_file}")
    
    print("\n" + "=" * 50)
    print("COMPLETE PUBLICATION PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 50)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nYour publication package is ready! 🎉")
    print("\nNext steps:")
    print("1. Review the generated figures and tables")
    print("2. Check the validation report for any issues")
    print("3. Incorporate the outputs into your manuscript")
    print("4. Reference the methodology from the evaluation and benchmarking modules")
    
    return package


def demonstrate_custom_analysis(package):
    """Demonstrate custom analysis of publication results."""
    
    print("\n" + "=" * 50)
    print("CUSTOM ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Extract results for custom analysis
    results = package.results_summary
    
    if 'method_metrics' not in results:
        print("No method metrics available for custom analysis")
        return
    
    method_metrics = results['method_metrics']
    
    # 1. Performance ranking analysis
    print("\n1. Performance Ranking Analysis:")
    
    metrics_to_analyze = ['mfd', 'alpha_diversity_ks_pvalue', 'beta_diversity_ks_pvalue']
    
    for metric in metrics_to_analyze:
        if all(metric in method_metrics[method] for method in method_metrics):
            print(f"\n   {metric.upper()} Rankings:")
            
            # Sort methods by this metric
            if 'pvalue' in metric:
                # Higher p-values are better (less significant difference from real)
                sorted_methods = sorted(method_metrics.items(), 
                                      key=lambda x: x[1][metric], reverse=True)
            else:
                # Lower values are better for MFD
                sorted_methods = sorted(method_metrics.items(), 
                                      key=lambda x: x[1][metric])
            
            for rank, (method, metrics_dict) in enumerate(sorted_methods, 1):
                value = metrics_dict[metric]
                print(f"     {rank}. {method:15s}: {value:.4f}")
    
    # 2. Biological validation analysis
    print("\n2. Biological Validation Analysis:")
    
    if 'biological_validation' in results:
        bio_results = results['biological_validation']
        
        # Find method with best biological plausibility
        best_bio_method = None
        best_bio_score = -1
        
        for method, bio_metrics in bio_results.items():
            if isinstance(bio_metrics, dict):
                plausibility = bio_metrics.get('overall_biological_plausibility', 0)
                if plausibility > best_bio_score:
                    best_bio_score = plausibility
                    best_bio_method = method
        
        if best_bio_method:
            print(f"   Best biological plausibility: {best_bio_method} ({best_bio_score:.3f})")
        
        # Analyze violations
        print("\n   Co-exclusion Violation Analysis:")
        for method, bio_metrics in bio_results.items():
            if isinstance(bio_metrics, dict):
                violations = bio_metrics.get('co_exclusion_violations', 0)
                print(f"     {method:15s}: {violations} violations")
    
    # 3. Statistical significance summary
    print("\n3. Statistical Significance Summary:")
    
    if hasattr(package, 'statistical_significance') and package.statistical_significance:
        sig_tests = package.statistical_significance
        
        total_comparisons = len(sig_tests)
        significant_comparisons = sum(1 for result in sig_tests.values() 
                                    if result.get('significant', False))
        
        print(f"   Total comparisons: {total_comparisons}")
        print(f"   Significant comparisons: {significant_comparisons}")
        print(f"   Significance rate: {significant_comparisons/total_comparisons*100:.1f}%")
        
        # Effect size analysis
        effect_sizes = [result.get('effect_size', 0) for result in sig_tests.values() 
                       if result.get('significant', False)]
        
        if effect_sizes:
            avg_effect_size = np.mean(effect_sizes)
            max_effect_size = np.max(effect_sizes)
            print(f"   Average effect size: {avg_effect_size:.3f}")
            print(f"   Maximum effect size: {max_effect_size:.3f}")
    
    print("\nCustom analysis completed!")


def main():
    """Run the complete publication pipeline example."""
    
    try:
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
        # Run the complete pipeline
        package = run_complete_pipeline()
        
        # Demonstrate custom analysis
        demonstrate_custom_analysis(package)
        
        print("\n🎉 Publication pipeline example completed successfully!")
        
    except Exception as e:
        print(f"\nError in publication pipeline: {e}")
        print("Please check your configuration and try again.")
        raise


if __name__ == "__main__":
    main()