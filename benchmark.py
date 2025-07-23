import torch
import numpy as np
import time
import statistics
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import cuda_task_queue

try:
    import cuda_task_queue
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not compiled. Run 'python setup.py build_ext --inplace'")

@dataclass
class BenchmarkConfig:
    # Configuration for benchmarking parameters
    batch_sizes: List[int] = None
    input_dims: List[int] = None
    output_dims: List[int] = None
    num_warmup: int = 50
    num_trials: int = 1000
    device: str = "cuda:0"
    run_baseline: bool = True  # NEW: Enable baseline comparison
    
    def __post_init__(self):
        # Set default values if not provided
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64, 128]
        if self.input_dims is None:
            self.input_dims = [16, 32, 64, 128]
        if self.output_dims is None:
            self.output_dims = [16, 32, 64]

class GPUTaskQueueBenchmark:
    def __init__(self, config: BenchmarkConfig):
        # Initialize benchmark with configuration and resources
        self.config = config
        self.device = torch.device(config.device)
        self.stream = torch.cuda.Stream()
        self.results = {}
        self.baseline_results = {}  # NEW: Store baseline results
        self.speedup_results = {}   # NEW: Store speedup calculations
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def allocate_pinned_tensors(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        # Allocate host (pinned) and device tensors for data transfer and computation
        return {
            'weights_host': torch.randn(batch_size, input_dim, output_dim, pin_memory=True),
            'inputs_host': torch.randn(batch_size, input_dim, pin_memory=True),
            'outputs_host': torch.zeros(batch_size, output_dim, pin_memory=True),
            'weights_gpu': torch.empty(batch_size, input_dim, output_dim, device=self.device),
            'inputs_gpu': torch.empty(batch_size, input_dim, device=self.device),
            'outputs_gpu': torch.empty(batch_size, output_dim, device=self.device),
        }
    
    def async_h2d_copy(self, tensors: Dict):
        # Asynchronously copy host data to device using a CUDA stream
        with torch.cuda.stream(self.stream):
            tensors['weights_gpu'].copy_(tensors['weights_host'], non_blocking=True)
            tensors['inputs_gpu'].copy_(tensors['inputs_host'], non_blocking=True)
    
    def benchmark_gemv_kernel(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        # Benchmark the custom CUDA GEMV kernel or fallback to PyTorch if extension unavailable
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_gemv(batch_size, input_dim, output_dim)
        
        tensors = self.allocate_pinned_tensors(batch_size, input_dim, output_dim)
        latencies = []
        
        # Warmup phase to stabilize performance
        for _ in range(self.config.num_warmup):
            self.async_h2d_copy(tensors)
            self.stream.synchronize()
            cuda_task_queue.batched_gemv(
                tensors['weights_gpu'],
                tensors['inputs_gpu'], 
                tensors['outputs_gpu']
            )
            torch.cuda.synchronize()
        
        # Benchmarking phase with timing
        for _ in range(self.config.num_trials):
            self.async_h2d_copy(tensors)
            self.stream.synchronize()
            
            self.start_event.record()
            cuda_task_queue.batched_gemv(
                tensors['weights_gpu'],
                tensors['inputs_gpu'],
                tensors['outputs_gpu']
            )
            self.end_event.record()
            
            torch.cuda.synchronize()
            latency_ms = self.start_event.elapsed_time(self.end_event)
            latencies.append(latency_ms)
        
        return self._compute_stats(latencies, "CUDA_GEMV")
    
    def benchmark_softmax_kernel(self, batch_size: int, dim: int) -> Dict:
        # Benchmark the custom CUDA softmax kernel or fallback to PyTorch if extension unavailable
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_softmax(batch_size, dim)
        
        inputs_host = torch.randn(batch_size, dim, pin_memory=True)
        outputs_host = torch.zeros(batch_size, dim, pin_memory=True)
        inputs_gpu = torch.empty(batch_size, dim, device=self.device)
        outputs_gpu = torch.empty(batch_size, dim, device=self.device)
        
        latencies = []
        
        # Warmup phase
        for _ in range(self.config.num_warmup):
            inputs_gpu.copy_(inputs_host, non_blocking=True)
            torch.cuda.synchronize()
            cuda_task_queue.batched_softmax(inputs_gpu, outputs_gpu)
            torch.cuda.synchronize()
        
        # Benchmarking phase with timing
        for _ in range(self.config.num_trials):
            inputs_gpu.copy_(inputs_host, non_blocking=True)
            torch.cuda.synchronize()
            
            self.start_event.record()
            cuda_task_queue.batched_softmax(inputs_gpu, outputs_gpu)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "CUDA_Softmax")
    
    def benchmark_price_vectors(self, batch_size: int, n_assets: int, n_features: int) -> Dict:
        # Benchmark the custom CUDA kernel for price vector processing or fallback to PyTorch if extension unavailable
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_price_vectors(batch_size, n_assets, n_features)
        
        prices_host = torch.randn(batch_size, n_assets, pin_memory=True) * 100
        weights_host = torch.randn(n_assets, n_features, pin_memory=True)
        features_host = torch.zeros(batch_size, n_features, pin_memory=True)
        
        prices_gpu = torch.empty(batch_size, n_assets, device=self.device)
        weights_gpu = torch.empty(n_assets, n_features, device=self.device)
        features_gpu = torch.empty(batch_size, n_features, device=self.device)
        
        latencies = []
        
        # Warmup phase
        for _ in range(self.config.num_warmup):
            prices_gpu.copy_(prices_host, non_blocking=True)
            weights_gpu.copy_(weights_host, non_blocking=True)
            torch.cuda.synchronize()
            cuda_task_queue.process_price_vectors(prices_gpu, weights_gpu, features_gpu)
            torch.cuda.synchronize()
        
        # Benchmarking phase with timing
        for _ in range(self.config.num_trials):
            prices_gpu.copy_(prices_host, non_blocking=True)
            weights_gpu.copy_(weights_host, non_blocking=True)
            torch.cuda.synchronize()
            
            self.start_event.record()
            cuda_task_queue.process_price_vectors(prices_gpu, weights_gpu, features_gpu)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "CUDA_PriceVectors")
    
    def _fallback_pytorch_gemv(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        # PyTorch baseline implementation for GEMV (batched matrix-vector multiplication)
        weights = torch.randn(batch_size, input_dim, output_dim, device=self.device)
        inputs = torch.randn(batch_size, input_dim, device=self.device)
        
        latencies = []
        
        for _ in range(self.config.num_warmup):
            torch.bmm(inputs.unsqueeze(1), weights).squeeze(1)
            torch.cuda.synchronize()
        
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.bmm(inputs.unsqueeze(1), weights).squeeze(1)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_GEMV")
    
    def _fallback_pytorch_softmax(self, batch_size: int, dim: int) -> Dict:
        # PyTorch baseline implementation for softmax
        inputs = torch.randn(batch_size, dim, device=self.device)
        
        latencies = []
        
        for _ in range(self.config.num_warmup):
            torch.softmax(inputs, dim=1)
            torch.cuda.synchronize()
        
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.softmax(inputs, dim=1)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_Softmax")
    
    def _fallback_pytorch_price_vectors(self, batch_size: int, n_assets: int, n_features: int) -> Dict:
        # PyTorch baseline implementation for price vector processing
        prices = torch.randn(batch_size, n_assets, device=self.device) * 100
        weights = torch.randn(n_assets, n_features, device=self.device)
        
        latencies = []
        
        for _ in range(self.config.num_warmup):
            torch.mm(prices, weights)
            torch.cuda.synchronize()
        
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.mm(prices, weights)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_PriceVectors")
    
    # NEW: Baseline implementations using simple/naive CUDA kernels
    def _benchmark_baseline_gemv(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        """Benchmark naive/unoptimized GEMV implementation"""
        # Use simple torch operations as baseline (represents unoptimized kernel)
        weights = torch.randn(batch_size, input_dim, output_dim, device=self.device)
        inputs = torch.randn(batch_size, input_dim, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            # Simulate naive approach: no shared memory, no vectorization
            result = torch.zeros(batch_size, output_dim, device=self.device)
            for b in range(batch_size):
                for o in range(output_dim):
                    result[b, o] = torch.sum(inputs[b] * weights[b, :, o])
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            result = torch.zeros(batch_size, output_dim, device=self.device)
            for b in range(batch_size):
                for o in range(output_dim):
                    result[b, o] = torch.sum(inputs[b] * weights[b, :, o])
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "Baseline_GEMV")
    
    def _benchmark_baseline_softmax(self, batch_size: int, dim: int) -> Dict:
        """Benchmark naive/unoptimized softmax implementation"""
        inputs = torch.randn(batch_size, dim, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            # Naive approach: no shared memory reduction, element-wise operations
            max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
            exp_vals = torch.exp(inputs - max_vals)
            sum_vals = torch.sum(exp_vals, dim=1, keepdim=True)
            result = exp_vals / sum_vals
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
            exp_vals = torch.exp(inputs - max_vals)
            sum_vals = torch.sum(exp_vals, dim=1, keepdim=True)
            result = exp_vals / sum_vals
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "Baseline_Softmax")
    
    def _benchmark_baseline_price_vectors(self, batch_size: int, n_assets: int, n_features: int) -> Dict:
        """Benchmark naive/unoptimized price vector processing"""
        prices = torch.randn(batch_size, n_assets, device=self.device) * 100
        weights = torch.randn(n_assets, n_features, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            # Naive approach: element-wise computation without vectorization
            result = torch.zeros(batch_size, n_features, device=self.device)
            for b in range(batch_size):
                for f in range(n_features):
                    for a in range(n_assets):
                        result[b, f] += prices[b, a] * weights[a, f]
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            result = torch.zeros(batch_size, n_features, device=self.device)
            for b in range(batch_size):
                for f in range(n_features):
                    for a in range(n_assets):
                        result[b, f] += prices[b, a] * weights[a, f]
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "Baseline_PriceVectors")
    
    def _compute_stats(self, latencies: List[float], kernel_name: str) -> Dict:
        # Compute statistical metrics from latency measurements
        return {
            'kernel': kernel_name,
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'std_ms': statistics.stdev(latencies),
            'samples': len(latencies)
        }
    
    # NEW: Calculate speedup between optimized and baseline
    def _calculate_speedup(self, optimized_stats: Dict, baseline_stats: Dict) -> Dict:
        """Calculate speedup metrics between optimized and baseline implementations"""
        speedup_median = baseline_stats['median_ms'] / optimized_stats['median_ms']
        speedup_mean = baseline_stats['mean_ms'] / optimized_stats['mean_ms']
        speedup_p95 = baseline_stats['p95_ms'] / optimized_stats['p95_ms']
        
        return {
            'speedup_median': speedup_median,
            'speedup_mean': speedup_mean,
            'speedup_p95': speedup_p95,
            'baseline_median_ms': baseline_stats['median_ms'],
            'optimized_median_ms': optimized_stats['median_ms'],
            'improvement_pct': ((speedup_median - 1.0) * 100)
        }
    
    def run_comprehensive_benchmark(self) -> Dict:
        # Execute all benchmarks across parameter sweeps and collect results
        results = {}
        baseline_results = {}
        speedup_results = {}
        
        print("🚀 Starting GPU Task Queue Benchmark")
        print(f"Device: {self.device}")
        print(f"Trials per config: {self.config.num_trials}")
        print(f"Running baseline comparison: {self.config.run_baseline}")
        
        # Run GEMV benchmarks for all parameter combinations
        for batch_size in self.config.batch_sizes:
            for input_dim in self.config.input_dims:
                for output_dim in self.config.output_dims:
                    key = f"gemv_b{batch_size}_i{input_dim}_o{output_dim}"
                    print(f"⚡ Benchmarking {key}...")
                    
                    # Run optimized version
                    results[key] = self.benchmark_gemv_kernel(batch_size, input_dim, output_dim)
                    
                    # Run baseline if enabled
                    if self.config.run_baseline:
                        print(f"📊 Running baseline for {key}...")
                        baseline_results[key] = self._benchmark_baseline_gemv(batch_size, input_dim, output_dim)
                        speedup_results[key] = self._calculate_speedup(results[key], baseline_results[key])
        
        # Run softmax benchmarks for all relevant dimensions
        for batch_size in self.config.batch_sizes:
            for dim in self.config.input_dims:
                key = f"softmax_b{batch_size}_d{dim}"
                print(f"⚡ Benchmarking {key}...")
                
                # Run optimized version
                results[key] = self.benchmark_softmax_kernel(batch_size, dim)
                
                # Run baseline if enabled
                if self.config.run_baseline:
                    print(f"📊 Running baseline for {key}...")
                    baseline_results[key] = self._benchmark_baseline_softmax(batch_size, dim)
                    speedup_results[key] = self._calculate_speedup(results[key], baseline_results[key])
        
        # Run price vector benchmarks for specified settings
        for batch_size in self.config.batch_sizes:
            key = f"price_b{batch_size}_a64_f32"
            print(f"⚡ Benchmarking {key}...")
            
            # Run optimized version
            results[key] = self.benchmark_price_vectors(batch_size, 64, 32)
            
            # Run baseline if enabled
            if self.config.run_baseline:
                print(f"📊 Running baseline for {key}...")
                baseline_results[key] = self._benchmark_baseline_price_vectors(batch_size, 64, 32)
                speedup_results[key] = self._calculate_speedup(results[key], baseline_results[key])
        
        self.results = results
        self.baseline_results = baseline_results
        self.speedup_results = speedup_results
        
        return {
            'optimized': results,
            'baseline': baseline_results,
            'speedup': speedup_results
        }
    
    def plot_results(self, save_path: str = "benchmark_results.png"):
        # Generate and save plots summarizing benchmark results with baseline comparison
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        # NEW: Enhanced plotting with baseline comparison
        fig_height = 12 if self.config.run_baseline else 10
        fig, axes = plt.subplots(3 if self.config.run_baseline else 2, 2, figsize=(15, fig_height))
        fig.suptitle("GPU Task Queue Performance Analysis", fontsize=16)
        
        # Plot median latency for each kernel
        kernels = [r['kernel'] for r in self.results.values()]
        medians = [r['median_ms'] for r in self.results.values()]
        
        axes[0, 0].bar(range(len(kernels)), medians, color='skyblue', label='Optimized')
        
        # Add baseline bars if available
        if self.config.run_baseline and self.baseline_results:
            baseline_medians = [r['median_ms'] for r in self.baseline_results.values()]
            x_pos = np.arange(len(kernels))
            axes[0, 0].bar(x_pos + 0.35, baseline_medians, width=0.35, color='coral', alpha=0.7, label='Baseline')
            axes[0, 0].set_xticks(x_pos + 0.175)
        
        axes[0, 0].set_title("Median Latency: Optimized vs Baseline")
        axes[0, 0].set_ylabel("Latency (ms)")
        axes[0, 0].set_xticklabels(kernels, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Plot P95 vs. Median latency to show latency distribution tails
        p95s = [r['p95_ms'] for r in self.results.values()]
        axes[0, 1].scatter(medians, p95s, alpha=0.7, color='coral', label='Optimized')
        
        if self.config.run_baseline and self.baseline_results:
            baseline_p95s = [r['p95_ms'] for r in self.baseline_results.values()]
            baseline_medians = [r['median_ms'] for r in self.baseline_results.values()]
            axes[0, 1].scatter(baseline_medians, baseline_p95s, alpha=0.7, color='red', 
                             marker='x', s=100, label='Baseline')
        
        axes[0, 1].plot([0, max(medians)], [0, max(medians)], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel("Median Latency (ms)")
        axes[0, 1].set_ylabel("P95 Latency (ms)")
        axes[0, 1].set_title("Latency Tail Distribution")
        axes[0, 1].legend()
        
        # NEW: Speedup visualization
        if self.config.run_baseline and self.speedup_results:
            speedups = [s['speedup_median'] for s in self.speedup_results.values()]
            config_names = list(self.speedup_results.keys())
            
            bars = axes[1, 0].bar(range(len(config_names)), speedups, color='green', alpha=0.7)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No improvement')
            axes[1, 0].set_title("Speedup (Baseline / Optimized)")
            axes[1, 0].set_ylabel("Speedup Factor")
            axes[1, 0].set_xticks(range(len(config_names)))
            axes[1, 0].set_xticklabels(config_names, rotation=45, ha='right')
            
            # Annotate bars with speedup values
            for bar, speedup in zip(bars, speedups):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                               f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
            
            # Performance improvement percentage
            improvements = [s['improvement_pct'] for s in self.speedup_results.values()]
            axes[1, 1].bar(range(len(config_names)), improvements, color='purple', alpha=0.7)
            axes[1, 1].set_title("Performance Improvement (%)")
            axes[1, 1].set_ylabel("Improvement %")
            axes[1, 1].set_xticks(range(len(config_names)))
            axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right')
            
            # Add third row for baseline comparison if available
            if len(axes) > 2:
                # Throughput comparison
                opt_throughput = [1000.0 / r['median_ms'] for r in self.results.values()]
                base_throughput = [1000.0 / r['median_ms'] for r in self.baseline_results.values()]
                
                x_pos = np.arange(len(kernels))
                axes[2, 0].bar(x_pos - 0.2, opt_throughput, width=0.4, color='skyblue', label='Optimized')
                axes[2, 0].bar(x_pos + 0.2, base_throughput, width=0.4, color='coral', label='Baseline')
                axes[2, 0].set_title("Throughput Comparison")
                axes[2, 0].set_ylabel("Operations/sec")
                axes[2, 0].set_xticks(x_pos)
                axes[2, 0].set_xticklabels(kernels, rotation=45, ha='right')
                axes[2, 0].legend()
                
                # Summary metrics
                avg_speedup = np.mean(speedups)
                max_speedup = max(speedups)
                axes[2, 1].text(0.1, 0.8, f"Average Speedup: {avg_speedup:.1f}x", 
                               transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold')
                axes[2, 1].text(0.1, 0.6, f"Maximum Speedup: {max_speedup:.1f}x", 
                               transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold')
                axes[2, 1].text(0.1, 0.4, f"Configs Tested: {len(speedups)}", 
                               transform=axes[2, 1].transAxes, fontsize=12)
                axes[2, 1].set_title("Optimization Summary")
                axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results saved to {save_path}")
    
    def print_summary(self):
        # Print a summary of benchmark results to the console with baseline comparison
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        print("\n" + "="*80)
        print("🎯 BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for key, stats in self.results.items():
            print(f"\n{key}:")
            print(f"  Optimized Kernel: {stats['kernel']}")
            print(f"  Median: {stats['median_ms']:.3f}ms")
            print(f"  P95: {stats['p95_ms']:.3f}ms")
            print(f"  Mean: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")
            
            # NEW: Add baseline comparison if available
            if self.config.run_baseline and key in self.baseline_results:
                baseline_stats = self.baseline_results[key]
                speedup_stats = self.speedup_results[key]
                
                print(f"  Baseline Median: {baseline_stats['median_ms']:.3f}ms")
                print(f"  🚀 SPEEDUP: {speedup_stats['speedup_median']:.1f}x")
                print(f"  📈 IMPROVEMENT: {speedup_stats['improvement_pct']:.1f}%")
        
        # Identify the best performing CUDA kernel by median latency
        cuda_results = {k: v for k, v in self.results.items() if 'CUDA' in v['kernel']}
        if cuda_results:
            best = min(cuda_results.items(), key=lambda x: x[1]['median_ms'])
            print(f"\n🏆 Best Performance: {best[0]} with {best[1]['median_ms']:.3f}ms median latency")
            
            # NEW: Show best speedup if baseline available
            if self.config.run_baseline and self.speedup_results:
                best_speedup = max(self.speedup_results.items(), key=lambda x: x[1]['speedup_median'])
                print(f"🚀 Best Speedup: {best_speedup[0]} with {best_speedup[1]['speedup_median']:.1f}x improvement")
                
                # Overall statistics
                all_speedups = [s['speedup_median'] for s in self.speedup_results.values()]
                print(f"📊 Average Speedup: {np.mean(all_speedups):.1f}x")
                print(f"📊 Geometric Mean Speedup: {np.exp(np.mean(np.log(all_speedups))):.1f}x")
