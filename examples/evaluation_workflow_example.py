#!/usr/bin/env python3
"""
Comprehensive Evaluation Workflow Example

This example demonstrates how to use the evaluation system to assess
microbiome generation quality, prediction accuracy, and biological validation.

Usage:
    python examples/evaluation_workflow_example.py
    python examples/evaluation_workflow_example.py --example advanced
    python examples/evaluation_workflow_example.py --example biological
    python examples/evaluation_workflow_example.py --example statistical
"""

import argparse
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
import sys
sys.path.append('src')

from evaluation import (
    MicrobiomeEvaluator, BiologicalValidator, MethodComparator,
    microbiome_frechet_distance, alpha_diversity, beta_diversity,
    abundance_mae, top_k_accuracy, shannon_entropy
)


def create_sample_data(num_samples=200, num_taxa=100, seed=42):
    """Create sample microbiome data for demonstration."""
    np.random.seed(seed)
    
    # Create phylogenetic kernel (simplified)
    phylo_kernel = np.eye(num_taxa) + 0.1 * np.random.rand(num_taxa, num_taxa)
    phylo_kernel = (phylo_kernel + phylo_kernel.T) / 2
    
    # Real compositions (more diverse)
    real_compositions = np.random.dirichlet(np.ones(num_taxa) * 2, size=num_samples)
    
    # Generated compositions (slightly less diverse)
    our_generated = np.random.dirichlet(np.ones(num_taxa) * 1.8, size=num_samples)
    vae_generated = np.random.dirichlet(np.ones(num_taxa) * 1.5, size=num_samples)
    gan_generated = np.random.dirichlet(np.ones(num_taxa) * 1.3, size=num_samples)
    
    # Prediction data (temporal)
    num_horizons = 5
    predictions_ours = np.random.dirichlet(np.ones(num_taxa) * 1.8, size=(num_samples, num_horizons, num_taxa))
    predictions_vae = np.random.dirichlet(np.ones(num_taxa) * 1.5, size=(num_samples, num_horizons, num_taxa))
    ground_truth = np.random.dirichlet(np.ones(num_taxa) * 2, size=(num_samples, num_horizons, num_taxa))
    
    return {
        'phylogenetic_kernel': phylo_kernel,
        'real_compositions': real_compositions,
        'generated_compositions': {
            'Our Method': our_generated,
            'VAE': vae_generated,
            'GAN': gan_generated
        },
        'predictions': {
            'Our Method': predictions_ours,
            'VAE': predictions_vae
        },
        'ground_truth': ground_truth
    }


def basic_evaluation_example():
    """Demonstrate basic evaluation metrics."""
    print("=" * 60)
    print("BASIC EVALUATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Computing individual metrics...")
    
    # Alpha diversity
    real_alpha = alpha_diversity(data['real_compositions'])
    gen_alpha = alpha_diversity(data['generated_compositions']['Our Method'])
    
    print(f"Alpha Diversity:")
    print(f"  Real: {real_alpha.mean():.4f} ± {real_alpha.std():.4f}")
    print(f"  Generated: {gen_alpha.mean():.4f} ± {gen_alpha.std():.4f}")
    
    # Beta diversity
    real_beta = beta_diversity(data['real_compositions'])
    gen_beta = beta_diversity(data['generated_compositions']['Our Method'])
    
    print(f"Beta Diversity (mean dissimilarity):")
    print(f"  Real: {real_beta.mean():.4f} ± {real_beta.std():.4f}")
    print(f"  Generated: {gen_beta.mean():.4f} ± {gen_beta.std():.4f}")
    
    # Microbiome Fréchet Distance
    mfd = microbiome_frechet_distance(
        data['real_compositions'],
        data['generated_compositions']['Our Method'],
        data['phylogenetic_kernel']
    )
    print(f"Microbiome Fréchet Distance: {mfd:.4f}")
    
    # Prediction metrics
    mae = abundance_mae(
        data['predictions']['Our Method'][:, 0, :],  # First horizon
        data['ground_truth'][:, 0, :]
    )
    
    top_k_acc = top_k_accuracy(
        data['predictions']['Our Method'][:, 0, :],
        data['ground_truth'][:, 0, :],
        k=10
    )
    
    print(f"Prediction Metrics (horizon 1):")
    print(f"  MAE: {mae:.4f}")
    print(f"  Top-10 Accuracy: {top_k_acc:.4f}")
    
    print("\n2. Using MicrobiomeEvaluator...")
    
    # Comprehensive evaluation
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=data['phylogenetic_kernel'],
        k_values=[5, 10, 20]
    )
    
    # Evaluate generation
    gen_metrics = evaluator.evaluate_generation(
        real_compositions=data['real_compositions'],
        generated_compositions=data['generated_compositions']['Our Method']
    )
    
    print(f"Generation Evaluation Results:")
    for metric, value in gen_metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Evaluate prediction
    pred_metrics = evaluator.evaluate_prediction(
        predictions=data['predictions']['Our Method'],
        ground_truth=data['ground_truth']
    )
    
    print(f"\nPrediction Evaluation Results:")
    for metric, value in pred_metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        elif isinstance(value, list):
            print(f"  {metric}: {[f'{v:.4f}' for v in value]}")
        else:
            print(f"  {metric}: {value:.4f}")


def biological_validation_example():
    """Demonstrate biological validation capabilities."""
    print("=" * 60)
    print("BIOLOGICAL VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Setting up biological constraints...")
    
    # Define biological constraints
    co_exclusion_pairs = [
        (0, 1),   # Taxa 0 and 1 compete
        (5, 6),   # Taxa 5 and 6 compete
        (10, 11), # Taxa 10 and 11 compete
    ]
    
    metabolic_pathways = {
        'glycolysis': [2, 3, 4, 5],
        'nitrogen_fixation': [15, 16, 17],
        'methanogenesis': [20, 21, 22, 23],
        'sulfate_reduction': [25, 26, 27]
    }
    
    disease_biomarkers = {
        'ibd': [8, 9],
        'diabetes': [12, 13, 14],
        'obesity': [18, 19]
    }
    
    # Create biological validator
    validator = BiologicalValidator(
        co_exclusion_pairs=co_exclusion_pairs,
        metabolic_pathways=metabolic_pathways,
        disease_biomarkers=disease_biomarkers,
        co_exclusion_threshold=0.1,
        pathway_coherence_threshold=0.3
    )
    
    print(f"Defined constraints:")
    print(f"  Co-exclusion pairs: {len(co_exclusion_pairs)}")
    print(f"  Metabolic pathways: {len(metabolic_pathways)}")
    print(f"  Disease biomarkers: {len(disease_biomarkers)}")
    
    print("\n2. Validating compositions...")
    
    # Validate each method
    for method_name, compositions in data['generated_compositions'].items():
        print(f"\n{method_name}:")
        
        # Comprehensive validation
        results = validator.validate_all(compositions)
        
        print(f"  Overall biological plausibility: {results['overall_biological_plausibility']:.4f}")
        print(f"  Co-exclusion violations: {results['co_exclusion_violations']}")
        print(f"  Metabolic consistency: {results['metabolic_consistency']:.4f}")
        print(f"  Phylogenetic coherence: {results['phylogenetic_coherence']:.4f}")
        print(f"  Biomarker relationships: {results['biomarker_relationships']:.4f}")
        
        # Individual constraint validation
        co_exclusion_results = validator.validate_co_exclusion_patterns(compositions)
        metabolic_results = validator.validate_metabolic_consistency(compositions)
        
        print(f"  Detailed results:")
        print(f"    Co-exclusion pass rate: {co_exclusion_results['pass_rate']:.4f}")
        print(f"    Metabolic coherence: {metabolic_results['overall_coherence']:.4f}")
    
    print("\n3. Using evaluator with biological validation...")
    
    # Create evaluator with biological validation
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=data['phylogenetic_kernel'],
        biological_validator=validator
    )
    
    # Evaluate with biological constraints
    gen_metrics = evaluator.evaluate_generation(
        real_compositions=data['real_compositions'],
        generated_compositions=data['generated_compositions']['Our Method']
    )
    
    print(f"Evaluation with biological validation:")
    print(f"  MFD: {gen_metrics['mfd']:.4f}")
    print(f"  Biological plausibility: {gen_metrics['biological_plausibility']:.4f}")


def statistical_comparison_example():
    """Demonstrate statistical method comparison."""
    print("=" * 60)
    print("STATISTICAL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    print("1. Evaluating multiple methods...")
    
    # Initialize evaluator
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=data['phylogenetic_kernel'],
        k_values=[5, 10, 20]
    )
    
    # Evaluate all methods
    all_results = {}
    for method_name, compositions in data['generated_compositions'].items():
        print(f"  Evaluating {method_name}...")
        
        gen_results = evaluator.evaluate_generation(
            real_compositions=data['real_compositions'],
            generated_compositions=compositions
        )
        
        # Add prediction results if available
        if method_name in data['predictions']:
            pred_results = evaluator.evaluate_prediction(
                predictions=data['predictions'][method_name],
                ground_truth=data['ground_truth']
            )
            all_results[method_name] = {**gen_results, **pred_results}
        else:
            all_results[method_name] = gen_results
    
    print("\n2. Statistical comparison...")
    
    # Initialize method comparator
    comparator = MethodComparator(
        alpha=0.05,
        correction_method='holm',
        effect_size_method='cohens_d'
    )
    
    # Compare methods
    comparison_results = comparator.compare_methods(all_results)
    
    print(f"Pairwise comparisons:")
    for comparison, result in comparison_results['pairwise_tests'].items():
        significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        effect_size_desc = "large" if abs(result['effect_size']) > 0.8 else "medium" if abs(result['effect_size']) > 0.5 else "small"
        
        print(f"  {comparison}:")
        print(f"    p-value: {result['p_value']:.4f} {significance}")
        print(f"    Effect size: {result['effect_size']:.3f} ({effect_size_desc})")
        print(f"    Significant: {result['significant']}")
    
    # Overall ranking
    if 'method_ranking' in comparison_results:
        print(f"\nMethod ranking (by MFD):")
        for i, (method, score) in enumerate(comparison_results['method_ranking'].items(), 1):
            print(f"  {i}. {method}: {score:.4f}")
    
    print("\n3. Effect size analysis...")
    
    # Analyze effect sizes
    significant_comparisons = [
        (comp, result) for comp, result in comparison_results['pairwise_tests'].items()
        if result['significant']
    ]
    
    if significant_comparisons:
        print(f"Significant improvements found:")
        for comparison, result in significant_comparisons:
            print(f"  {comparison}: effect size = {result['effect_size']:.3f}")
    else:
        print("No significant differences found between methods.")


def advanced_evaluation_example():
    """Demonstrate advanced evaluation features."""
    print("=" * 60)
    print("ADVANCED EVALUATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(num_samples=500, num_taxa=150)
    
    print("1. Setting up comprehensive evaluation...")
    
    # Create biological validator
    validator = BiologicalValidator(
        co_exclusion_pairs=[(i, i+1) for i in range(0, 20, 2)],
        metabolic_pathways={
            f'pathway_{i}': list(range(i*5, (i+1)*5))
            for i in range(10)
        },
        disease_biomarkers={
            'condition_1': [50, 51, 52],
            'condition_2': [60, 61, 62],
            'condition_3': [70, 71, 72]
        }
    )
    
    # Create method comparator with strict settings
    comparator = MethodComparator(
        alpha=0.01,  # Stricter significance level
        correction_method='bonferroni',
        effect_size_method='cliff_delta',
        bootstrap_iterations=1000
    )
    
    # Create comprehensive evaluator
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=data['phylogenetic_kernel'],
        biological_validator=validator,
        method_comparator=comparator,
        k_values=[5, 10, 15, 20, 25],
        confidence_level=0.99
    )
    
    print("2. Comprehensive evaluation...")
    
    # Evaluate all methods with full analysis
    comprehensive_results = {}
    
    for method_name, compositions in data['generated_compositions'].items():
        print(f"  Analyzing {method_name}...")
        
        # Generation evaluation
        gen_results = evaluator.evaluate_generation(
            real_compositions=data['real_compositions'],
            generated_compositions=compositions
        )
        
        # Prediction evaluation (if available)
        pred_results = {}
        if method_name in data['predictions']:
            pred_results = evaluator.evaluate_prediction(
                predictions=data['predictions'][method_name],
                ground_truth=data['ground_truth']
            )
        
        # Combine results
        comprehensive_results[method_name] = {
            **gen_results,
            **pred_results
        }
        
        # Print summary
        print(f"    MFD: {gen_results['mfd']:.4f}")
        print(f"    Biological plausibility: {gen_results.get('biological_plausibility', 'N/A')}")
        if pred_results:
            print(f"    Prediction MAE: {pred_results['mae']:.4f}")
            print(f"    Top-10 Accuracy: {pred_results['top_k_accuracy'][10]:.4f}")
    
    print("\n3. Advanced statistical analysis...")
    
    # Comprehensive method comparison
    comparison_results = evaluator.method_comparator.compare_methods(
        comprehensive_results
    )
    
    # Print detailed results
    print(f"Statistical Analysis Results:")
    print(f"  Total comparisons: {len(comparison_results['pairwise_tests'])}")
    
    significant_count = sum(
        1 for result in comparison_results['pairwise_tests'].values()
        if result['significant']
    )
    print(f"  Significant differences: {significant_count}")
    
    # Effect size distribution
    effect_sizes = [
        result['effect_size'] for result in comparison_results['pairwise_tests'].values()
    ]
    print(f"  Effect size range: {min(effect_sizes):.3f} to {max(effect_sizes):.3f}")
    print(f"  Mean effect size: {np.mean(effect_sizes):.3f}")
    
    # Best performing method
    if 'method_ranking' in comparison_results:
        best_method = list(comparison_results['method_ranking'].keys())[0]
        print(f"  Best performing method: {best_method}")
    
    print("\n4. Saving results...")
    
    # Save comprehensive results
    output_dir = Path('examples/evaluation_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save evaluation results
    with open(output_dir / 'comprehensive_evaluation.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for method, results in comprehensive_results.items():
            json_results[method] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[method][key] = value.tolist()
                elif isinstance(value, dict):
                    json_results[method][key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[method][key] = value
        
        json.dump(json_results, f, indent=2)
    
    # Save comparison results
    with open(output_dir / 'statistical_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"  - comprehensive_evaluation.json")
    print(f"  - statistical_comparison.json")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description='Evaluation Workflow Examples')
    parser.add_argument(
        '--example',
        choices=['basic', 'biological', 'statistical', 'advanced'],
        default='basic',
        help='Which example to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    print("Microbiome Evaluation Workflow Examples")
    print("=" * 60)
    print(f"Running example: {args.example}")
    print()
    
    try:
        if args.example == 'basic':
            basic_evaluation_example()
        elif args.example == 'biological':
            biological_validation_example()
        elif args.example == 'statistical':
            statistical_comparison_example()
        elif args.example == 'advanced':
            advanced_evaluation_example()
        
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