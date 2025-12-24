"""Performance benchmarking suite for microbiome generation models.

This module implements comprehensive benchmarking capabilities including:
- Generation speed measurement across different batch sizes
- GPU and CPU memory usage tracking
- Scaling analysis for different dataset sizes
- Baseline method comparison timing
- Memory profiling utilities

Requirements: 3.1, 3.2, 3.4 from publication-ready spec
"""

import time
import psutil
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class BenchmarkResults:
    """Results from performance benchmarking.
    
    Attributes:
        generation_speed: Dict mapping batch sizes to samples/second
        memory_usage: Dict with peak GPU/CPU memory consumption
        scaling_metrics: Dict with performance vs dataset size metrics
        baseline_comparisons: Dict with timing comparisons vs baseline methods
        metadata: Additional metadata about the benchmark run
    """
    generation_speed: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    scaling_metrics: Dict[str, List[float]] = field(default_factory=dict)
    baseline_comparisons: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """Memory usage tracking utility for GPU and CPU.
    
    Tracks peak memory consumption during model operations and provides
    detailed memory profiling over time.
    
    Attributes:
        track_gpu: Whether to track GPU memory (requires CUDA)
        track_cpu: Whether to track CPU memory
        sampling_interval: Time interval between memory samples (seconds)
    """
    
    def __init__(
        self,
        track_gpu: bool = True,
        track_cpu: bool = True,
        sampling_interval: float = 0.1
    ):
        """Initialize memory profiler.
        
        Args:
            track_gpu: Whether to track GPU memory usage
            track_cpu: Whether to track CPU memory usage
            sampling_interval: Time between memory samples in seconds
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_cpu = track_cpu
        self.sampling_interval = sampling_interval
        
        # Initialize NVML for GPU monitoring if available
        self.nvml_initialized = False
        if self.track_gpu and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception:
                pass
        
        # Memory tracking state
        self.is_profiling = False
        self.memory_samples = []
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        self.baseline_gpu_memory = 0.0
        self.baseline_cpu_memory = 0.0
    
    def start_profiling(self) -> None:
        """Start memory profiling session."""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.memory_samples = []
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        
        # Record baseline memory usage
        self.baseline_gpu_memory = self._get_gpu_memory()
        self.baseline_cpu_memory = self._get_cpu_memory()
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return memory statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        
        # Final memory sample
        self._sample_memory()
        
        # Compute statistics
        stats = {
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'peak_cpu_memory_mb': self.peak_cpu_memory,
            'baseline_gpu_memory_mb': self.baseline_gpu_memory,
            'baseline_cpu_memory_mb': self.baseline_cpu_memory,
            'gpu_memory_increase_mb': self.peak_gpu_memory - self.baseline_gpu_memory,
            'cpu_memory_increase_mb': self.peak_cpu_memory - self.baseline_cpu_memory,
            'num_samples': len(self.memory_samples)
        }
        
        return stats
    
    def sample_memory(self) -> Dict[str, float]:
        """Take a single memory sample.
        
        Returns:
            Current memory usage in MB
        """
        return self._sample_memory()
    
    def _sample_memory(self) -> Dict[str, float]:
        """Internal method to sample current memory usage."""
        sample = {
            'timestamp': time.time(),
            'gpu_memory_mb': 0.0,
            'cpu_memory_mb': 0.0
        }
        
        if self.track_gpu:
            gpu_memory = self._get_gpu_memory()
            sample['gpu_memory_mb'] = gpu_memory
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory)
        
        if self.track_cpu:
            cpu_memory = self._get_cpu_memory()
            sample['cpu_memory_mb'] = cpu_memory
            self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_memory)
        
        if self.is_profiling:
            self.memory_samples.append(sample)
        
        return sample
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.track_gpu:
            return 0.0
        
        try:
            if torch.cuda.is_available():
                # PyTorch memory tracking
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                return float(allocated)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        if not self.track_cpu:
            return 0.0
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return float(memory_info.rss / (1024 ** 2))  # Convert to MB
        except Exception:
            return 0.0
    
    def get_memory_timeline(self) -> List[Dict[str, float]]:
        """Get complete memory usage timeline.
        
        Returns:
            List of memory samples with timestamps
        """
        return self.memory_samples.copy()
    
    def reset(self) -> None:
        """Reset profiler state."""
        self.is_profiling = False
        self.memory_samples = []
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        self.baseline_gpu_memory = 0.0
        self.baseline_cpu_memory = 0.0


class BenchmarkSuite:
    """Comprehensive performance benchmarking suite.
    
    Provides benchmarking capabilities for microbiome generation models:
    - Generation speed measurement across batch sizes
    - Memory usage tracking during training and inference
    - Scaling analysis for different dataset sizes
    - Baseline method comparison timing
    
    Attributes:
        device: Device to run benchmarks on
        memory_profiler: MemoryProfiler instance for memory tracking
        warmup_iterations: Number of warmup iterations before timing
        timing_iterations: Number of iterations for timing measurements
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_iterations: int = 5,
        timing_iterations: int = 10,
        memory_profiling: bool = True
    ):
        """Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on (auto-detected if None)
            warmup_iterations: Number of warmup iterations before timing
            timing_iterations: Number of iterations for timing measurements
            memory_profiling: Whether to enable memory profiling
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.warmup_iterations = warmup_iterations
        self.timing_iterations = timing_iterations
        
        # Initialize memory profiler
        self.memory_profiler = MemoryProfiler() if memory_profiling else None
        
        # Results storage
        self.results = BenchmarkResults()
    
    def benchmark_generation_speed(
        self,
        model: nn.Module,
        batch_sizes: List[int] = None,
        num_samples_per_batch: int = 1000,
        **generation_kwargs
    ) -> Dict[str, float]:
        """Benchmark generation speed across different batch sizes.
        
        Args:
            model: Model to benchmark (must have generate() method)
            batch_sizes: List of batch sizes to test (default: [1, 8, 16, 32, 64, 128])
            num_samples_per_batch: Number of samples to generate per batch size
            **generation_kwargs: Additional arguments for model.generate()
        
        Returns:
            Dictionary mapping batch sizes to samples/second
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
        
        model.eval()
        model.to(self.device)
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            if not hasattr(model, 'generate'):
                # Skip if model doesn't have generate method
                continue
            
            # Calculate number of batches needed
            num_batches = max(1, num_samples_per_batch // batch_size)
            total_samples = num_batches * batch_size
            
            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    with torch.no_grad():
                        _ = model.generate(
                            num_samples=batch_size,
                            device=self.device,
                            **generation_kwargs
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    # Skip this batch size if generation fails
                    break
            else:
                # Timing runs (only if warmup succeeded)
                times = []
                
                for _ in range(self.timing_iterations):
                    start_time = time.time()
                    
                    for _ in range(num_batches):
                        try:
                            with torch.no_grad():
                                _ = model.generate(
                                    num_samples=batch_size,
                                    device=self.device,
                                    **generation_kwargs
                                )
                        except Exception:
                            break
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                if times:
                    avg_time = np.mean(times)
                    samples_per_second = total_samples / avg_time
                    speed_results[f'batch_size_{batch_size}'] = float(samples_per_second)
        
        self.results.generation_speed.update(speed_results)
        return speed_results
    
    def benchmark_memory_usage(
        self,
        model: nn.Module,
        operation: str = 'generation',
        batch_size: int = 32,
        num_samples: int = 1000,
        **operation_kwargs
    ) -> Dict[str, float]:
        """Benchmark memory usage during model operations.
        
        Args:
            model: Model to benchmark
            operation: Type of operation ('generation', 'training', 'inference')
            batch_size: Batch size for the operation
            num_samples: Number of samples for the operation
            **operation_kwargs: Additional arguments for the operation
        
        Returns:
            Dictionary with memory usage statistics
        """
        if self.memory_profiler is None:
            return {}
        
        model.to(self.device)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Start memory profiling
        self.memory_profiler.start_profiling()
        
        try:
            if operation == 'generation':
                self._benchmark_generation_memory(
                    model, batch_size, num_samples, **operation_kwargs
                )
            elif operation == 'training':
                self._benchmark_training_memory(
                    model, batch_size, num_samples, **operation_kwargs
                )
            elif operation == 'inference':
                self._benchmark_inference_memory(
                    model, batch_size, num_samples, **operation_kwargs
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        except Exception as e:
            print(f"Memory benchmarking failed: {e}")
        
        # Stop profiling and get results
        memory_stats = self.memory_profiler.stop_profiling()
        
        # Store results
        operation_key = f'{operation}_batch_{batch_size}'
        self.results.memory_usage[operation_key] = memory_stats
        
        return memory_stats
    
    def _benchmark_generation_memory(
        self,
        model: nn.Module,
        batch_size: int,
        num_samples: int,
        **kwargs
    ) -> None:
        """Benchmark memory usage during generation."""
        model.eval()
        
        num_batches = max(1, num_samples // batch_size)
        
        with torch.no_grad():
            for _ in range(num_batches):
                if hasattr(model, 'generate'):
                    _ = model.generate(
                        num_samples=batch_size,
                        device=self.device,
                        **kwargs
                    )
                
                # Sample memory periodically
                self.memory_profiler.sample_memory()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    def _benchmark_training_memory(
        self,
        model: nn.Module,
        batch_size: int,
        num_samples: int,
        **kwargs
    ) -> None:
        """Benchmark memory usage during training."""
        model.train()
        
        # Create dummy data
        num_taxa = getattr(model, 'num_taxa', 100)
        dummy_data = torch.randn(batch_size, num_taxa, device=self.device)
        dummy_data = torch.softmax(dummy_data, dim=1)  # Make it compositional
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        num_batches = max(1, num_samples // batch_size)
        
        for _ in range(min(num_batches, 10)):  # Limit to 10 batches for memory test
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'compute_loss'):
                losses = model.compute_loss(dummy_data)
                loss = losses.get('total_loss', losses.get('loss', list(losses.values())[0]))
            else:
                # Fallback for models without compute_loss
                output = model(dummy_data)
                loss = torch.nn.functional.mse_loss(output, dummy_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Sample memory
            self.memory_profiler.sample_memory()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    def _benchmark_inference_memory(
        self,
        model: nn.Module,
        batch_size: int,
        num_samples: int,
        **kwargs
    ) -> None:
        """Benchmark memory usage during inference."""
        model.eval()
        
        # Create dummy data
        num_taxa = getattr(model, 'num_taxa', 100)
        dummy_data = torch.randn(batch_size, num_taxa, device=self.device)
        dummy_data = torch.softmax(dummy_data, dim=1)  # Make it compositional
        
        num_batches = max(1, num_samples // batch_size)
        
        with torch.no_grad():
            for _ in range(num_batches):
                # Forward pass
                _ = model(dummy_data)
                
                # Sample memory
                self.memory_profiler.sample_memory()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    def benchmark_scaling_analysis(
        self,
        model_factory: Callable[[int], nn.Module],
        dataset_sizes: List[int] = None,
        num_taxa_list: List[int] = None,
        operation: str = 'generation',
        **operation_kwargs
    ) -> Dict[str, List[float]]:
        """Analyze performance scaling with dataset size and model complexity.
        
        Args:
            model_factory: Function that creates model given num_taxa
            dataset_sizes: List of dataset sizes to test
            num_taxa_list: List of num_taxa values to test
            operation: Operation to benchmark ('generation', 'training')
            **operation_kwargs: Additional arguments for the operation
        
        Returns:
            Dictionary with scaling metrics
        """
        if dataset_sizes is None:
            dataset_sizes = [100, 500, 1000, 2000, 5000]
        
        if num_taxa_list is None:
            num_taxa_list = [50, 100, 200, 500]
        
        scaling_results = {
            'dataset_sizes': dataset_sizes,
            'num_taxa_list': num_taxa_list,
            'generation_times': [],
            'memory_usage': [],
            'samples_per_second': []
        }
        
        for num_taxa in num_taxa_list:
            try:
                # Create model
                model = model_factory(num_taxa)
                model.to(self.device)
                model.eval()
                
                # Test different dataset sizes
                for dataset_size in dataset_sizes:
                    batch_size = min(32, dataset_size)
                    
                    # Measure generation time
                    start_time = time.time()
                    
                    if self.memory_profiler:
                        self.memory_profiler.start_profiling()
                    
                    try:
                        if operation == 'generation' and hasattr(model, 'generate'):
                            with torch.no_grad():
                                _ = model.generate(
                                    num_samples=batch_size,
                                    device=self.device,
                                    **operation_kwargs
                                )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        generation_time = end_time - start_time
                        
                        scaling_results['generation_times'].append(generation_time)
                        scaling_results['samples_per_second'].append(batch_size / generation_time)
                        
                        # Memory usage
                        if self.memory_profiler:
                            memory_stats = self.memory_profiler.stop_profiling()
                            scaling_results['memory_usage'].append(
                                memory_stats.get('peak_gpu_memory_mb', 0.0)
                            )
                        else:
                            scaling_results['memory_usage'].append(0.0)
                    
                    except Exception as e:
                        print(f"Scaling test failed for num_taxa={num_taxa}, dataset_size={dataset_size}: {e}")
                        scaling_results['generation_times'].append(float('inf'))
                        scaling_results['samples_per_second'].append(0.0)
                        scaling_results['memory_usage'].append(0.0)
            
            except Exception as e:
                print(f"Model creation failed for num_taxa={num_taxa}: {e}")
                # Fill with placeholder values
                for _ in dataset_sizes:
                    scaling_results['generation_times'].append(float('inf'))
                    scaling_results['samples_per_second'].append(0.0)
                    scaling_results['memory_usage'].append(0.0)
        
        self.results.scaling_metrics.update(scaling_results)
        return scaling_results
    
    def benchmark_baseline_comparison(
        self,
        models: Dict[str, nn.Module],
        operation: str = 'generation',
        batch_size: int = 32,
        num_samples: int = 1000,
        **operation_kwargs
    ) -> Dict[str, float]:
        """Compare timing against baseline methods.
        
        Args:
            models: Dictionary mapping method names to model instances
            operation: Operation to benchmark ('generation', 'training')
            batch_size: Batch size for the operation
            num_samples: Number of samples for the operation
            **operation_kwargs: Additional arguments for the operation
        
        Returns:
            Dictionary mapping method names to timing results
        """
        timing_results = {}
        
        for method_name, model in models.items():
            try:
                model.to(self.device)
                model.eval()
                
                # Warmup
                for _ in range(self.warmup_iterations):
                    try:
                        if operation == 'generation' and hasattr(model, 'generate'):
                            with torch.no_grad():
                                _ = model.generate(
                                    num_samples=batch_size,
                                    device=self.device,
                                    **operation_kwargs
                                )
                        elif operation == 'generation' and hasattr(model, 'sample'):
                            with torch.no_grad():
                                _ = model.sample(
                                    num_samples=batch_size,
                                    device=self.device,
                                    **operation_kwargs
                                )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception:
                        break
                
                # Timing runs
                times = []
                num_batches = max(1, num_samples // batch_size)
                
                for _ in range(self.timing_iterations):
                    start_time = time.time()
                    
                    try:
                        for _ in range(num_batches):
                            if operation == 'generation' and hasattr(model, 'generate'):
                                with torch.no_grad():
                                    _ = model.generate(
                                        num_samples=batch_size,
                                        device=self.device,
                                        **operation_kwargs
                                    )
                            elif operation == 'generation' and hasattr(model, 'sample'):
                                with torch.no_grad():
                                    _ = model.sample(
                                        num_samples=batch_size,
                                        device=self.device,
                                        **operation_kwargs
                                    )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    except Exception as e:
                        print(f"Timing failed for {method_name}: {e}")
                        break
                
                if times:
                    avg_time = np.mean(times)
                    samples_per_second = (num_batches * batch_size) / avg_time
                    timing_results[method_name] = float(samples_per_second)
                else:
                    timing_results[method_name] = 0.0
            
            except Exception as e:
                print(f"Baseline comparison failed for {method_name}: {e}")
                timing_results[method_name] = 0.0
        
        self.results.baseline_comparisons.update(timing_results)
        return timing_results
    
    def get_results(self) -> BenchmarkResults:
        """Get complete benchmark results.
        
        Returns:
            BenchmarkResults object with all collected metrics
        """
        # Add metadata
        self.results.metadata.update({
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'warmup_iterations': self.warmup_iterations,
            'timing_iterations': self.timing_iterations,
            'timestamp': time.time()
        })
        
        if torch.cuda.is_available():
            self.results.metadata.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            })
        
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """Save benchmark results to JSON file.
        
        Args:
            output_path: Path to save results JSON file
        """
        results = self.get_results()
        
        # Convert to serializable format
        results_dict = {
            'generation_speed': results.generation_speed,
            'memory_usage': results.memory_usage,
            'scaling_metrics': results.scaling_metrics,
            'baseline_comparisons': results.baseline_comparisons,
            'metadata': results.metadata
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def reset(self) -> None:
        """Reset benchmark results."""
        self.results = BenchmarkResults()
        if self.memory_profiler:
            self.memory_profiler.reset()


def benchmark_model(
    model: nn.Module,
    output_dir: str = 'benchmark_results',
    batch_sizes: List[int] = None,
    baseline_models: Optional[Dict[str, nn.Module]] = None,
    **kwargs
) -> BenchmarkResults:
    """Convenience function to run comprehensive benchmarks on a model.
    
    Args:
        model: Model to benchmark
        output_dir: Directory to save results
        batch_sizes: List of batch sizes to test
        baseline_models: Optional baseline models for comparison
        **kwargs: Additional arguments for benchmarking
    
    Returns:
        BenchmarkResults with all benchmark metrics
    """
    suite = BenchmarkSuite(**kwargs)
    
    # Generation speed benchmark
    print("Benchmarking generation speed...")
    suite.benchmark_generation_speed(model, batch_sizes=batch_sizes)
    
    # Memory usage benchmark
    print("Benchmarking memory usage...")
    suite.benchmark_memory_usage(model, operation='generation')
    
    # Baseline comparison if provided
    if baseline_models:
        print("Comparing against baselines...")
        all_models = {'target_model': model, **baseline_models}
        suite.benchmark_baseline_comparison(all_models)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suite.save_results(str(output_path / 'benchmark_results.json'))
    
    print(f"Benchmark results saved to {output_path}")
    
    return suite.get_results()