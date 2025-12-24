#!/usr/bin/env python3
"""
Biological Validation Example

This example demonstrates how to use the BiologicalValidator to ensure
generated microbiome compositions respect known biological constraints
and relationships.

Usage:
    python examples/biological_validation_example.py
    python examples/biological_validation_example.py --validation-type comprehensive
    python examples/biological_validation_example.py --validation-type custom
"""

import argparse
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
import sys
sys.path.append('src')

from evaluation import BiologicalValidator, MicrobiomeEvaluator


def create_sample_data(num_samples=200, num_taxa=100, seed=42):
    """Create sample microbiome data with known biological patterns."""
    np.random.seed(seed)
    
    # Create compositions with some biological structure
    compositions = []
    
    for i in range(num_samples):
        # Start with random composition
        comp = np.random.dirichlet(np.ones(num_taxa) * 2)
        
        # Add some biological constraints
        # Simulate co-exclusion: taxa 0 and 1 rarely co-occur
        if comp[0] > 0.1:
            comp[1] *= 0.1  # Reduce taxa 1 when taxa 0 is abundant
        elif comp[1] > 0.1:
            comp[0] *= 0.1  # Reduce taxa 0 when taxa 1 is abundant
        
        # Simulate metabolic pathway coherence
        # Taxa 2-5 form a metabolic pathway and should co-occur
        pathway_strength = np.random.uniform(0.5, 1.5)
        for j in range(2, 6):
            comp[j] *= pathway_strength
        
        # Renormalize to maintain simplex constraint
        comp = comp / comp.sum()
        compositions.append(comp)
    
    return np.array(compositions)


def basic_validation_example():
    """Demonstrate basic biological validation."""
    print("=" * 60)
    print("BASIC BIOLOGICAL VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    compositions = create_sample_data()
    
    print("1. Setting up biological constraints...")
    
    # Define co-exclusion pairs (competing taxa)
    co_exclusion_pairs = [
        (0, 1),   # Taxa 0 and 1 compete for resources
        (5, 6),   # Taxa 5 and 6 have different pH preferences
        (10, 11), # Taxa 10 and 11 compete for the same niche
    ]
    
    # Define metabolic pathways (taxa that work together)
    metabolic_pathways = {
        'glycolysis': [2, 3, 4, 5],
        'nitrogen_fixation': [15, 16, 17],
        'methanogenesis': [20, 21, 22, 23],
        'sulfate_reduction': [25, 26, 27]
    }
    
    # Define disease biomarkers
    disease_biomarkers = {
        'inflammatory_bowel_disease': [8, 9],
        'type2_diabetes': [12, 13, 14],
        'obesity': [18, 19]
    }
    
    print(f"Defined constraints:")
    print(f"  Co-exclusion pairs: {len(co_exclusion_pairs)}")
    print(f"  Metabolic pathways: {len(metabolic_pathways)}")
    print(f"  Disease biomarkers: {len(disease_biomarkers)}")
    
    # Create biological validator
    validator = BiologicalValidator(
        co_exclusion_pairs=co_exclusion_pairs,
        metabolic_pathways=metabolic_pathways,
        disease_biomarkers=disease_biomarkers,
        co_exclusion_threshold=0.1,
        pathway_coherence_threshold=0.3
    )
    
    print("\n2. Validating compositions...")
    
    # Comprehensive validation
    results = validator.validate_all(compositions)
    
    print(f"Validation Results:")
    print(f"  Overall biological plausibility: {results['overall_biological_plausibility']:.4f}")
    print(f"  Co-exclusion violations: {results['co_exclusion_violations']}")
    print(f"  Metabolic consistency: {results['metabolic_consistency']:.4f}")
    print(f"  Phylogenetic coherence: {results['phylogenetic_coherence']:.4f}")
    print(f"  Biomarker relationships: {results['biomarker_relationships']:.4f}")
    
    print("\n3. Individual constraint validation...")
    
    # Co-exclusion validation
    co_exclusion_results = validator.validate_co_exclusion_patterns(compositions)
    print(f"Co-exclusion Analysis:")
    print(f"  Pass rate: {co_exclusion_results['pass_rate']:.4f}")
    print(f"  Total violations: {co_exclusion_results['total_violations']}")
    print(f"  Violation details: {co_exclusion_results['violation_details']}")
    
    # Metabolic pathway validation
    metabolic_results = validator.validate_metabolic_consistency(compositions)
    print(f"Metabolic Pathway Analysis:")
    print(f"  Overall coherence: {metabolic_results['overall_coherence']:.4f}")
    for pathway, coherence in metabolic_results['pathway_coherences'].items():
        print(f"    {pathway}: {coherence:.4f}")
    
    # Disease biomarker validation
    biomarker_results = validator.validate_disease_biomarkers(compositions)
    print(f"Disease Biomarker Analysis:")
    print(f"  Overall validity: {biomarker_results['overall_validity']:.4f}")
    for condition, validity in biomarker_results['condition_validities'].items():
        print(f"    {condition}: {validity:.4f}")


def comprehensive_validation_example():
    """Demonstrate comprehensive biological validation with detailed analysis."""
    print("=" * 60)
    print("COMPREHENSIVE BIOLOGICAL VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create multiple datasets to compare
    print("1. Creating datasets with different biological properties...")
    
    # Good biological dataset (follows constraints)
    good_compositions = create_sample_data(num_samples=100, seed=42)
    
    # Poor biological dataset (violates constraints)
    np.random.seed(123)
    poor_compositions = np.random.dirichlet(np.ones(100) * 0.5, size=100)  # More random
    
    # Create comprehensive validator
    validator = BiologicalValidator(
        co_exclusion_pairs=[(i, i+1) for i in range(0, 20, 2)],  # More pairs
        metabolic_pathways={
            f'pathway_{i}': list(range(i*4, (i+1)*4))
            for i in range(10)
        },
        disease_biomarkers={
            'condition_1': [50, 51, 52],
            'condition_2': [60, 61, 62],
            'condition_3': [70, 71, 72],
            'condition_4': [80, 81, 82]
        },
        phylogenetic_groups={
            'firmicutes': list(range(30, 50)),
            'bacteroidetes': list(range(50, 70)),
            'proteobacteria': list(range(70, 90))
        }
    )
    
    print("2. Comparing datasets...")
    
    datasets = {
        'Good Biological': good_compositions,
        'Poor Biological': poor_compositions
    }
    
    all_results = {}
    
    for name, compositions in datasets.items():
        print(f"\n{name} Dataset:")
        
        # Comprehensive validation
        results = validator.validate_all(compositions)
        all_results[name] = results
        
        print(f"  Overall plausibility: {results['overall_biological_plausibility']:.4f}")
        print(f"  Co-exclusion violations: {results['co_exclusion_violations']}")
        print(f"  Metabolic consistency: {results['metabolic_consistency']:.4f}")
        print(f"  Phylogenetic coherence: {results['phylogenetic_coherence']:.4f}")
        print(f"  Biomarker relationships: {results['biomarker_relationships']:.4f}")
        
        # Detailed violation analysis
        if results['co_exclusion_violations'] > 0:
            violations = validator.get_detailed_violations(compositions)
            print(f"  Violation examples: {violations[:3]}")  # Show first 3
    
    print("\n3. Statistical comparison of biological validity...")
    
    # Compare biological plausibility scores
    good_scores = [all_results['Good Biological']['overall_biological_plausibility']]
    poor_scores = [all_results['Poor Biological']['overall_biological_plausibility']]
    
    print(f"Biological Plausibility Comparison:")
    print(f"  Good dataset: {good_scores[0]:.4f}")
    print(f"  Poor dataset: {poor_scores[0]:.4f}")
    print(f"  Difference: {good_scores[0] - poor_scores[0]:.4f}")
    
    # Analyze specific constraint violations
    print(f"\nConstraint Violation Analysis:")
    for constraint in ['co_exclusion_violations', 'metabolic_consistency', 
                      'phylogenetic_coherence', 'biomarker_relationships']:
        good_val = all_results['Good Biological'][constraint]
        poor_val = all_results['Poor Biological'][constraint]
        
        if isinstance(good_val, (int, float)) and isinstance(poor_val, (int, float)):
            diff = good_val - poor_val
            print(f"  {constraint}: {good_val:.4f} vs {poor_val:.4f} (diff: {diff:.4f})")


def custom_validation_example():
    """Demonstrate custom biological validation with user-defined constraints."""
    print("=" * 60)
    print("CUSTOM BIOLOGICAL VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    compositions = create_sample_data(num_samples=150, num_taxa=80)
    
    print("1. Defining custom biological constraints...")
    
    # Custom co-exclusion based on ecological competition
    ecological_competitors = [
        (0, 1),   # Aerobic vs anaerobic bacteria
        (5, 6),   # Acid-loving vs alkaline-preferring
        (10, 11), # Fast vs slow growers
        (15, 16), # Temperature specialists
        (20, 21)  # Nutrient competitors
    ]
    
    # Custom metabolic networks
    metabolic_networks = {
        'carbon_metabolism': [2, 3, 4, 5, 6],
        'nitrogen_cycle': [12, 13, 14, 15],
        'sulfur_cycle': [22, 23, 24],
        'phosphorus_cycle': [32, 33, 34],
        'iron_metabolism': [42, 43, 44]
    }
    
    # Disease-specific biomarkers with confidence levels
    disease_signatures = {
        'crohns_disease': [7, 8, 9],
        'ulcerative_colitis': [17, 18, 19],
        'irritable_bowel': [27, 28, 29],
        'celiac_disease': [37, 38, 39]
    }
    
    # Phylogenetic relationships
    phylogenetic_clades = {
        'gram_positive': list(range(50, 65)),
        'gram_negative': list(range(65, 80)),
        'archaea': list(range(75, 80))
    }
    
    print(f"Custom constraints defined:")
    print(f"  Ecological competitors: {len(ecological_competitors)}")
    print(f"  Metabolic networks: {len(metabolic_networks)}")
    print(f"  Disease signatures: {len(disease_signatures)}")
    print(f"  Phylogenetic clades: {len(phylogenetic_clades)}")
    
    # Create custom validator with strict thresholds
    custom_validator = BiologicalValidator(
        co_exclusion_pairs=ecological_competitors,
        metabolic_pathways=metabolic_networks,
        disease_biomarkers=disease_signatures,
        phylogenetic_groups=phylogenetic_clades,
        co_exclusion_threshold=0.05,  # Stricter threshold
        pathway_coherence_threshold=0.4,  # Higher coherence required
        biomarker_threshold=0.2
    )
    
    print("\n2. Custom validation analysis...")
    
    # Run custom validation
    custom_results = custom_validator.validate_all(compositions)
    
    print(f"Custom Validation Results:")
    print(f"  Overall biological plausibility: {custom_results['overall_biological_plausibility']:.4f}")
    print(f"  Ecological competition violations: {custom_results['co_exclusion_violations']}")
    print(f"  Metabolic network consistency: {custom_results['metabolic_consistency']:.4f}")
    print(f"  Phylogenetic clade coherence: {custom_results['phylogenetic_coherence']:.4f}")
    print(f"  Disease signature validity: {custom_results['biomarker_relationships']:.4f}")
    
    print("\n3. Detailed network analysis...")
    
    # Analyze each metabolic network
    metabolic_results = custom_validator.validate_metabolic_consistency(compositions)
    
    print(f"Metabolic Network Analysis:")
    for network, coherence in metabolic_results['pathway_coherences'].items():
        status = "PASS" if coherence > 0.4 else "FAIL"
        print(f"  {network}: {coherence:.4f} [{status}]")
    
    # Analyze phylogenetic coherence
    phylo_results = custom_validator.validate_phylogenetic_coherence(compositions)
    
    print(f"Phylogenetic Coherence Analysis:")
    for clade, coherence in phylo_results['group_coherences'].items():
        status = "PASS" if coherence > 0.3 else "FAIL"
        print(f"  {clade}: {coherence:.4f} [{status}]")
    
    print("\n4. Saving custom validation results...")
    
    # Save results
    output_dir = Path('examples/validation_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / 'custom_validation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in custom_results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_dir / 'custom_validation_results.json'}")


def integration_with_evaluator_example():
    """Demonstrate integration of biological validation with MicrobiomeEvaluator."""
    print("=" * 60)
    print("BIOLOGICAL VALIDATION INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    real_compositions = create_sample_data(num_samples=100, seed=42)
    generated_compositions = create_sample_data(num_samples=100, seed=123)
    
    # Create phylogenetic kernel
    num_taxa = real_compositions.shape[1]
    phylo_kernel = np.eye(num_taxa) + 0.1 * np.random.rand(num_taxa, num_taxa)
    phylo_kernel = (phylo_kernel + phylo_kernel.T) / 2
    
    print("1. Setting up integrated evaluation...")
    
    # Create biological validator
    validator = BiologicalValidator(
        co_exclusion_pairs=[(0, 1), (5, 6), (10, 11)],
        metabolic_pathways={
            'pathway_1': [2, 3, 4, 5],
            'pathway_2': [15, 16, 17, 18],
            'pathway_3': [25, 26, 27, 28]
        },
        disease_biomarkers={
            'condition_1': [8, 9],
            'condition_2': [18, 19]
        }
    )
    
    # Create evaluator with biological validation
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=phylo_kernel,
        biological_validator=validator,
        k_values=[5, 10, 20]
    )
    
    print("2. Running integrated evaluation...")
    
    # Evaluate with biological constraints
    results = evaluator.evaluate_generation(
        real_compositions=real_compositions,
        generated_compositions=generated_compositions
    )
    
    print(f"Integrated Evaluation Results:")
    print(f"  MFD: {results['mfd']:.4f}")
    print(f"  Alpha diversity KS p-value: {results['alpha_diversity_ks_pvalue']:.4f}")
    print(f"  Beta diversity KS p-value: {results['beta_diversity_ks_pvalue']:.4f}")
    print(f"  Biological plausibility: {results['biological_plausibility']:.4f}")
    
    # Compare biological validity of real vs generated
    real_bio_results = validator.validate_all(real_compositions)
    gen_bio_results = validator.validate_all(generated_compositions)
    
    print(f"\nBiological Validity Comparison:")
    print(f"  Real compositions plausibility: {real_bio_results['overall_biological_plausibility']:.4f}")
    print(f"  Generated compositions plausibility: {gen_bio_results['overall_biological_plausibility']:.4f}")
    
    bio_diff = real_bio_results['overall_biological_plausibility'] - gen_bio_results['overall_biological_plausibility']
    print(f"  Biological validity gap: {bio_diff:.4f}")
    
    if bio_diff > 0.1:
        print("  ⚠️  Generated compositions have significantly lower biological plausibility")
    elif bio_diff < -0.1:
        print("  ✅ Generated compositions have higher biological plausibility than real data")
    else:
        print("  ✅ Generated compositions have similar biological plausibility to real data")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description='Biological Validation Examples')
    parser.add_argument(
        '--validation-type',
        choices=['basic', 'comprehensive', 'custom', 'integration'],
        default='basic',
        help='Which validation example to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    print("Microbiome Biological Validation Examples")
    print("=" * 60)
    print(f"Running validation type: {args.validation_type}")
    print()
    
    try:
        if args.validation_type == 'basic':
            basic_validation_example()
        elif args.validation_type == 'comprehensive':
            comprehensive_validation_example()
        elif args.validation_type == 'custom':
            custom_validation_example()
        elif args.validation_type == 'integration':
            integration_with_evaluator_example()
        
        print("\n" + "=" * 60)
        print("Biological validation example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running validation example: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())