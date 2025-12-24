#!/usr/bin/env python3
"""
Benchmarking Usage Example

This example demonstrates how to use the benchmarking suite to measure
generation speed, memory usage, and scaling performance of microbiome models.

Usage:
    python examples/benchmarking_example.py
    python examples/benchmarking_example.py --example memory
    python examples/benchmarking_example.py --example scaling
    python examples/benchmarking_example.py --example comparison
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add src to path for imports
import sys
sys.path.append('src')

from benchmarking import BenchmarkSuite, MemoryProfiler, BenchmarkResults


class MockMicrobiomeModel(nn.Module):
    """Mock microbiome generation model for benchmarking."""
    
    def __init__(self, num_taxa=100, hidden_dim=256):
        super().__init__()
        self.num_taxa = num_taxa
        self.hidden_dim = hidden_dim
        
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(num_taxa, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def generate(self, batch_size, device=None):
        """Generate microbiome compositions."""
        if device is None:
            device = next(self.parameters()).device
        
        # Generate random latent codes
        latent = torch.randn(batch_size, self.hidden_dim // 4, device=device)
        
        # Decode to compositions
        with torch.no_grad():
            compositions = self.decoder(latent)
        
        return compositions


class MockVAEModel(nn.Module):
    """Mock VAE baseline model."""
    
    def __init__(self, num_taxa=100, latent_dim=32):
        super().__init__()
        self.num_taxa = num_taxa
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_taxa, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def generate(self, batch_size, device=None):
        """Generate compositions from VAE."""
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        with torch.no_grad():
            compositions = self.decoder(z)
        
        return compositions


class MockGANModel(nn.Module):
    """Mock GAN baseline model."""
    
    def __init__(self, num_taxa=100, noise_dim=64):
        super().__init__()
        self.num_taxa = num_taxa
        self.noise_dim = noise_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_taxa),
            nn.Softmax(dim=-1)
        )
    
    def generate(self, batch_size, device=None):
        """Generate compositions from GAN."""
        if device is None:
            device = next(self.parameters()).device
        
        # Generate noise
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        
        with torch.no_grad():
            compositions = self.generator(noise)
        
        return compositions


def basic_benchmarking_example():
    """Demonstrate basic benchmarking capabilities."""
    print("=" * 60)
    print("BASIC BENCHMARKING EXAMPLE")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MockMicrobiomeModel(num_taxa=100, hidden_dim=256)
    model.to(device)
    model.eval()
    
    print("\n1. Generation Speed Benchmarking...")
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(device=device)
    
    # Benchmark generation speed across different batch sizes
    batch_sizes = [16, 32, 64, 128]
    speed_results = suite.benchmark_generation_speed(
        model=model,
        batch_sizes=batch_sizes,
        num_taxa=100,
        num_iterations=10
    )
    
    print(f"Generation Speed Results:")
    for batch_size, speed in speed_results.generation_speed.items():
        print(f"  {batch_size}: {speed:.1f} samples/sec")
    
    print(f"\nMemory Usage:")
    for key, value in speed_results.memory_usage.items():
        print(f"  {key}: {value:.1f} MB")
    
    print("\n2. Individual Speed Measurement...")
    
    # Measure speed for specific configuration
    batch_size = 64
    num_iterations = 20
    
    start_time = time.time()
    total_samples = 0
    
    for _ in range(num_iterations):
        compositions = model.generate(batch_size, device)
        total_samples += batch_size
    
    end_time = time.time()
    elapsed = end_time - start_time
    samples_per_sec = total_samples / elapsed
    
    print(f"Manual measurement (batch_size={batch_size}):")
    print(f"  Total samples: {total_samples}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Speed: {samples_per_sec:.1f} samples/sec")
    
    print("\n3. Memory Usage Analysis...")
    
    if torch.cuda.is_available():
        # GPU memory analysis
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**2
        
        # Generate large batch
        large_batch = model.generate(256, device)
        peak_memory = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"GPU Memory Analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory increase: {peak_memory - initial_memory:.1f} MB")
        
        torch.cuda.empty_cache()
    else:
        print("GPU not available - skipping GPU memory analysis")


def memory_profiling_example():
    """Demonstrate memory profiling capabilities."""
    print("=" * 60)
    print("MEMORY PROFILING EXAMPLE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MockMicrobiomeModel(num_taxa=150, hidden_dim=512)
    model.to(device)
    model.eval()
    
    print("1. Basic Memory Profiling...")
    
    # Initialize memory profiler
    profiler = MemoryProfiler(
        track_gpu=torch.cuda.is_available(),
        track_cpu=True,
        sampling_interval=0.1
    )
    
    # Start profiling
    profiler.start_profiling()
    
    print("  Running memory-intensive operations...")
    
    # Simulate training-like memory usage
    batch_sizes = [32, 64, 128, 256]
    for batch_size in batch_sizes:
        print(f"    Generating batch of size {batch_size}...")
        
        # Generate multiple batches
        for _ in range(5):
            compositions = model.generate(batch_size, device)
            
            # Simulate some processing
            loss = torch.sum(compositions ** 2)
            
            # Small delay to see memory changes
            time.sleep(0.1)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    peak_gpu = profiler.get_peak_gpu_memory()
    peak_cpu = profiler.get_peak_cpu_memory()
    memory_timeline = profiler.get_memory_timeline()
    
    print(f"\nMemory Profiling Results:")
    print(f"  Peak GPU memory: {peak_gpu:.1f} MB")
    print(f"  Peak CPU memory: {peak_cpu:.1f} MB")
    print(f"  Timeline samples: {len(memory_timeline)}")
    
    if memory_timeline:
        gpu_usage = [sample['gpu_mb'] for sample in memory_timeline if sample['gpu_mb'] > 0]
        cpu_usage = [sample['cpu_mb'] for sample in memory_timeline]
        
        if gpu_usage:
            print(f"  GPU usage range: {min(gpu_usage):.1f} - {max(gpu_usage):.1f} MB")
        print(f"  CPU usage range: {min(cpu_usage):.1f} - {max(cpu_usage):.1f} MB")
    
    print("\n2. Detailed Memory Analysis...")
    
    # Analyze memory usage by batch size
    suite = BenchmarkSuite(device=device)
    
    memory_by_batch = {}
    for batch_size in [16, 32, 64, 128, 256]:
        print(f"  Analyzing batch size {batch_size}...")
        
        memory_results = suite.benchmark_memory_usage(
            model=model,
            batch_size=batch_size,
            num_taxa=150
        )
        
        memory_by_batch[batch_size] = memory_results.memory_usage
    
    print(f"\nMemory Usage by Batch Size:")
    for batch_size, memory_info in memory_by_batch.items():
        print(f"  Batch {batch_size}:")
        for key, value in memory_info.items():
            print(f"    {key}: {value:.1f} MB")


def scaling_analysis_example():
    """Demonstrate scaling analysis capabilities."""
    print("=" * 60)
    print("SCALING ANALYSIS EXAMPLE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Dataset Size Scaling...")
    
    # Test different numbers of taxa
    taxa_sizes = [50, 100, 200, 300]
    scaling_results = {}
    
    for num_taxa in taxa_sizes:
        print(f"  Testing {num_taxa} taxa...")
        
        # Create model for this size
        model = MockMicrobiomeModel(num_taxa=num_taxa, hidden_dim=256)
        model.to(device)
        model.eval()
        
        # Initialize benchmark suite
        suite = BenchmarkSuite(device=device)
        
        # Benchmark this configuration
        speed_results = suite.benchmark_generation_speed(
            model=model,
            batch_sizes=[32, 64],
            num_taxa=num_taxa,
            num_iterations=5
        )
        
        scaling_results[num_taxa] = {
            'speed_batch_32': speed_results.generation_speed.get('batch_32', 0),
            'speed_batch_64': speed_results.generation_speed.get('batch_64', 0),
            'peak_memory': speed_results.memory_usage.get('peak_gpu_mb', 0)
        }
    
    print(f"\nScaling Results:")
    print(f"{'Taxa':<8} {'Speed (32)':<12} {'Speed (64)':<12} {'Memory (MB)':<12}")
    print("-" * 50)
    for num_taxa, results in scaling_results.items():
        print(f"{num_taxa:<8} {results['speed_batch_32']:<12.1f} "
              f"{results['speed_batch_64']:<12.1f} {results['peak_memory']:<12.1f}")
    
    print("\n2. Batch Size Scaling...")
    
    # Test scaling with batch size
    model = MockMicrobiomeModel(num_taxa=100, hidden_dim=256)
    model.to(device)
    model.eval()
    
    suite = BenchmarkSuite(device=device)
    
    batch_sizes = [8, 16, 32, 64, 128, 256]
    batch_scaling_results = suite.benchmark_generation_speed(
        model=model,
        batch_sizes=batch_sizes,
        num_taxa=100,
        num_iterations=8
    )
    
    print(f"\nBatch Size Scaling:")
    print(f"{'Batch Size':<12} {'Speed (samples/sec)':<20} {'Efficiency':<12}")
    print("-" * 50)
    
    base_speed = batch_scaling_results.generation_speed.get('batch_8', 1)
    for batch_size in batch_sizes:
        speed = batch_scaling_results.generation_speed.get(f'batch_{batch_size}', 0)
        efficiency = speed / (batch_size * base_speed / 8) if base_speed > 0 else 0
        print(f"{batch_size:<12} {speed:<20.1f} {efficiency:<12.2f}")
    
    print("\n3. Creating Scaling Plots...")
    
    # Create output directory
    output_dir = Path('examples/benchmarking_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Plot taxa scaling
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    taxa_list = list(scaling_results.keys())
    speeds_32 = [scaling_results[t]['speed_batch_32'] for t in taxa_list]
    speeds_64 = [scaling_results[t]['speed_batch_64'] for t in taxa_list]
    
    plt.plot(taxa_list, speeds_32, 'o-', label='Batch 32')
    plt.plot(taxa_list, speeds_64, 's-', label='Batch 64')
    plt.xlabel('Number of Taxa')
    plt.ylabel('Generation Speed (samples/sec)')
    plt.title('Speed vs Number of Taxa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot batch scaling
    plt.subplot(1, 2, 2)
    batch_list = batch_sizes
    speed_list = [batch_scaling_results.generation_speed.get(f'batch_{b}', 0) for b in batch_list]
    
    plt.plot(batch_list, speed_list, 'o-', color='green')
    plt.xlabel('Batch Size')
    plt.ylabel('Generation Speed (samples/sec)')
    plt.title('Speed vs Batch Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scaling plots saved to {output_dir / 'scaling_analysis.png'}")


def baseline_comparison_example():
    """Demonstrate baseline method comparison."""
    print("=" * 60)
    print("BASELINE COMPARISON EXAMPLE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Setting up models...")
    
    # Create different models
    our_model = MockMicrobiomeModel(num_taxa=100, hidden_dim=256)
    vae_model = MockVAEModel(num_taxa=100, latent_dim=32)
    gan_model = MockGANModel(num_taxa=100, noise_dim=64)
    
    models = {
        'Our Method': our_model,
        'VAE': vae_model,
        'GAN': gan_model
    }
    
    # Move to device
    for model in models.values():
        model.to(device)
        model.eval()
    
    print(f"Models created:")
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {param_count:,} parameters")
    
    print("\n2. Speed Comparison...")
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(device=device)
    
    # Compare generation speeds
    comparison_results = suite.compare_with_baselines(
        our_model=our_model,
        baseline_models={'VAE': vae_model, 'GAN': gan_model},
        batch_size=64,
        num_taxa=100,
        num_iterations=15
    )
    
    print(f"Speed Comparison (samples/sec):")
    for method, speed in comparison_results.baseline_comparisons.items():
        print(f"  {method}: {speed:.1f}")
    
    # Calculate relative performance
    our_speed = comparison_results.baseline_comparisons.get('Our Method', 0)
    if our_speed > 0:
        print(f"\nRelative Performance (vs Our Method):")
        for method, speed in comparison_results.baseline_comparisons.items():
            if method != 'Our Method':
                relative = speed / our_speed
                print(f"  {method}: {relative:.2f}x")
    
    print("\n3. Memory Comparison...")
    
    # Compare memory usage
    memory_comparison = {}
    for name, model in models.items():
        print(f"  Benchmarking {name}...")
        
        memory_results = suite.benchmark_memory_usage(
            model=model,
            batch_size=64,
            num_taxa=100
        )
        
        memory_comparison[name] = memory_results.memory_usage
    
    print(f"\nMemory Usage Comparison:")
    print(f"{'Method':<12} {'Peak GPU (MB)':<15} {'Peak CPU (MB)':<15}")
    print("-" * 45)
    for method, memory_info in memory_comparison.items():
        gpu_mem = memory_info.get('peak_gpu_mb', 0)
        cpu_mem = memory_info.get('peak_cpu_mb', 0)
        print(f"{method:<12} {gpu_mem:<15.1f} {cpu_mem:<15.1f}")
    
    print("\n4. Comprehensive Analysis...")
    
    # Run comprehensive benchmark for each model
    comprehensive_results = {}
    
    for name, model in models.items():
        print(f"  Running comprehensive benchmark for {name}...")
        
        results = suite.run_comprehensive_benchmark(
            model=model,
            batch_sizes=[16, 32, 64, 128],
            num_taxa=100,
            dataset_sizes=[100, 200, 500],
            num_iterations=5
        )
        
        comprehensive_results[name] = results
    
    # Summary comparison
    print(f"\nComprehensive Comparison Summary:")
    print(f"{'Method':<12} {'Avg Speed':<12} {'Peak Memory':<15} {'Efficiency':<12}")
    print("-" * 55)
    
    for method, results in comprehensive_results.items():
        # Calculate average speed across batch sizes
        speeds = list(results.generation_speed.values())
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Get peak memory
        peak_memory = results.memory_usage.get('peak_gpu_mb', 0)
        
        # Calculate efficiency (speed per MB)
        efficiency = avg_speed / peak_memory if peak_memory > 0 else 0
        
        print(f"{method:<12} {avg_speed:<12.1f} {peak_memory:<15.1f} {efficiency:<12.2f}")
    
    print("\n5. Saving Results...")
    
    # Save comparison results
    output_dir = Path('examples/benchmarking_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for method, results in comprehensive_results.items():
        json_results[method] = {
            'generation_speed': results.generation_speed,
            'memory_usage': results.memory_usage,
            'scaling_metrics': results.scaling_metrics,
            'baseline_comparisons': results.baseline_comparisons,
            'metadata': results.metadata
        }
    
    with open(output_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_dir / 'baseline_comparison.json'}")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description='Benchmarking Examples')
    parser.add_argument(
        '--example',
        choices=['basic', 'memory', 'scaling', 'comparison'],
        default='basic',
        help='Which example to run (default: basic)'
    )
    
    args = parser.parse_args()
    
    print("Microbiome Benchmarking Examples")
    print("=" * 60)
    print(f"Running example: {args.example}")
    print()
    
    try:
        if args.example == 'basic':
            basic_benchmarking_example()
        elif args.example == 'memory':
            memory_profiling_example()
        elif args.example == 'scaling':
            scaling_analysis_example()
        elif args.example == 'comparison':
            baseline_comparison_example()
        
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