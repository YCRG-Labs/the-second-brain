#!/usr/bin/env python3
"""
Quick Start Guide for Microbiome Simulation

This example provides a simple, step-by-step introduction to using the
microbiome simulation system for new users.

Usage:
    python examples/quick_start_guide.py
    python examples/quick_start_guide.py --step evaluation
    python examples/quick_start_guide.py --step benchmarking
    python examples/quick_start_guide.py --step publication
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from evaluation import MicrobiomeEvaluator, microbiome_frechet_distance
from benchmarking import BenchmarkSuite
from publication_pipeline import run_publication_pipeline, create_default_config


def create_simple_data():
    """Create simple example data for demonstration."""
    np.random.seed(42)
    
    # Small dataset for quick demonstration
    num_samples = 50
    num_taxa = 20
    
    # Create phylogenetic kernel (identity matrix for simplicity)
    phylo_kernel = np.eye(num_taxa)
    
    # Real compositions
    real_compositions = np.random.dirichlet(np.ones(num_taxa), size=num_samples)
    
    # Generated compositions (two methods)
    method_a = np.random.dirichlet(np.ones(num_taxa) * 0.8, size=num_samples)
    method_b = np.random.dirichlet(np.ones(num_taxa) * 1.2, size=num_samples)
    
    return {
        'phylogenetic_kernel': phylo_kernel,
        'real_compositions': real_compositions,
        'generated_compositions': {
            'Method A': method_a,
            'Method B': method_b
        }
    }


def step1_basic_evaluation():
    """Step 1: Basic evaluation of generated samples."""
    print("=" * 50)
    print("STEP 1: BASIC EVALUATION")
    print("=" * 50)
    
    print("Creating sample data...")
    data = create_simple_data()
    
    print(f"Dataset info:")
    print(f"  Real samples: {data['real_compositions'].shape}")
    print(f"  Methods: {list(data['generated_compositions'].keys())}")
    
    print("\n1.1 Computing Microbiome Fréchet Distance...")
    
    # Compute MFD for each method
    for method_name, compositions in data['generated_compositions'].items():
        mfd = microbiome_frechet_distance(
            data['real_compositions'],
            compositions,
            data['phylogenetic_kernel']
        )
        print(f"  {method_name}: MFD = {mfd:.4f}")
    
    print("\n1.2 Using MicrobiomeEvaluator...")
    
    # Create evaluator
    evaluator = MicrobiomeEvaluator(
        phylogenetic_kernel=data['phylogenetic_kernel']
    )
    
    # Evaluate each method
    for method_name, compositions in data['generated_compositions'].items():
        print(f"\nEvaluating {method_name}:")
        
        results = evaluator.evaluate_generation(
            real_compositions=data['real_compositions'],
            generated_compositions=compositions
        )
        
        print(f"  MFD: {results['mfd']:.4f}")
        print(f"  Alpha diversity KS p-value: {results.get('alpha_diversity_ks_pvalue', 'N/A')}")
        print(f"  Beta diversity KS p-value: {results.get('beta_diversity_ks_pvalue', 'N/A')}")
    
    print("\n✓ Step 1 completed! You've learned basic evaluation.")
    print("  Next: Try --step benchmarking")


def step2_basic_benchmarking():
    """Step 2: Basic benchmarking of generation speed."""
    print("=" * 50)
    print("STEP 2: BASIC BENCHMARKING")
    print("=" * 50)
    
    # Import torch for mock model
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available - skipping benchmarking example")
        return
    
    print("Creating mock model for benchmarking...")
    
    # Simple mock model
    class SimpleModel(nn.Module):
        def __init__(self, num_taxa=20):
            super().__init__()
            self.linear = nn.Linear(10, num_taxa)
            self.softmax = nn.Softmax(dim=-1)
        
        def generate(self, batch_size):
            noise = torch.randn(batch_size, 10)
            return self.softmax(self.linear(noise))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(num_taxa=20)
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    print("\n2.1 Measuring generation speed...")
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(device=device)
    
    # Benchmark generation speed
    speed_results = suite.benchmark_generation_speed(
        model=model,
        batch_sizes=[16, 32],
        num_taxa=20,
        num_iterations=5
    )
    
    print(f"Generation speeds:")
    for batch_size, speed in speed_results.generation_speed.items():
        print(f"  {batch_size}: {speed:.1f} samples/sec")
    
    print("\n2.2 Measuring memory usage...")
    
    # Benchmark memory usage
    memory_results = suite.benchmark_memory_usage(
        model=model,
        batch_size=32,
        num_taxa=20
    )
    
    print(f"Memory usage:")
    for key, value in memory_results.memory_usage.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Step 2 completed! You've learned basic benchmarking.")
    print("  Next: Try --step publication")


def step3_basic_publication():
    """Step 3: Basic publication pipeline."""
    print("=" * 50)
    print("STEP 3: BASIC PUBLICATION PIPELINE")
    print("=" * 50)
    
    print("Creating data for publication...")
    data = create_simple_data()
    
    # Prepare method data for publication pipeline
    method_data = {}
    for method_name, compositions in data['generated_compositions'].items():
        method_data[method_name] = {
            'real_compositions': data['real_compositions'],
            'generated_compositions': compositions
        }
    
    print("\n3.1 Setting up publication configuration...")
    
    # Create simple configuration
    config = create_default_config(
        output_dir='examples/quick_start_outputs',
        project_name='Quick Start Example'
    )
    
    # Simplify for quick start
    config.include_biological_validation = False
    config.include_benchmarking = False
    config.figure_formats = ['png']  # Faster than PDF
    config.k_values = [5, 10]
    
    print(f"  Output directory: {config.output_dir}")
    print(f"  Project name: {config.project_name}")
    
    print("\n3.2 Running publication pipeline...")
    
    try:
        package = run_publication_pipeline(
            method_data=method_data,
            phylogenetic_kernel=data['phylogenetic_kernel'],
            config=config
        )
        
        print(f"\n✓ Publication pipeline completed!")
        print(f"  Figures generated: {len(package.figures)}")
        print(f"  Tables generated: {len(package.tables)}")
        print(f"  Output directory: {config.output_dir}")
        
        # Show best method
        if package.results_summary.get('best_methods'):
            best_mfd = package.results_summary['best_methods'].get('mfd', {})
            if best_mfd:
                print(f"  Best method (MFD): {best_mfd['method']}")
        
    except Exception as e:
        print(f"Publication pipeline failed: {e}")
        print("This is normal for a quick start - some dependencies might be missing")
    
    print("\n✓ Step 3 completed! You've learned the publication pipeline.")
    print("  Check the output directory for results!")


def complete_quick_start():
    """Run all steps in sequence."""
    print("MICROBIOME SIMULATION QUICK START GUIDE")
    print("=" * 60)
    print("This guide will walk you through the basic features:")
    print("1. Evaluation of generated microbiome samples")
    print("2. Benchmarking of generation performance")
    print("3. Creating publication-ready outputs")
    print()
    
    # Run all steps
    step1_basic_evaluation()
    print("\n" + "="*60 + "\n")
    
    step2_basic_benchmarking()
    print("\n" + "="*60 + "\n")
    
    step3_basic_publication()
    
    print("\n" + "="*60)
    print("QUICK START GUIDE COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("- Try examples/evaluation_workflow_example.py for detailed evaluation")
    print("- Try examples/benchmarking_example.py for comprehensive benchmarking")
    print("- Try examples/publication_pipeline_example.py for full publication workflow")
    print("- Check the documentation in docs/ for more information")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Quick Start Guide for Microbiome Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete quick start guide
    python examples/quick_start_guide.py
    
    # Run specific step
    python examples/quick_start_guide.py --step evaluation
    python examples/quick_start_guide.py --step benchmarking
    python examples/quick_start_guide.py --step publication
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['evaluation', 'benchmarking', 'publication', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.step == 'evaluation':
            step1_basic_evaluation()
        elif args.step == 'benchmarking':
            step2_basic_benchmarking()
        elif args.step == 'publication':
            step3_basic_publication()
        elif args.step == 'all':
            complete_quick_start()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())