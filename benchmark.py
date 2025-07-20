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
    batch_sizes: List[int] = None
    input_dims: List[int] = None
    output_dims: List[int] = None
    num_warmup: int = 50
    num_trials: int = 1000
    device: str = "cuda:0"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64, 128]
        if self.input_dims is None:
            self.input_dims = [16, 32, 64, 128]
        if self.output_dims is None:
            self.output_dims = [16, 32, 64]

class GPUTaskQueueBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.stream = torch.cuda.Stream()
        self.results = {}
        
        # Pre-allocate CUDA events for precise timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def allocate_pinned_tensors(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        """Allocate pinned memory for zero-copy GPU transfers"""
        return {
            'weights_host': torch.randn(batch_size, input_dim, output_dim, pin_memory=True),
            'inputs_host': torch.randn(batch_size, input_dim, pin_memory=True),
            'outputs_host': torch.zeros(batch_size, output_dim, pin_memory=True),
            'weights_gpu': torch.empty(batch_size, input_dim, output_dim, device=self.device),
            'inputs_gpu': torch.empty(batch_size, input_dim, device=self.device),
            'outputs_gpu': torch.empty(batch_size, output_dim, device=self.device),
        }
    
    def async_h2d_copy(self, tensors: Dict):
        """Asynchronous host-to-device copy using pinned memory"""
        with torch.cuda.stream(self.stream):
            tensors['weights_gpu'].copy_(tensors['weights_host'], non_blocking=True)
            tensors['inputs_gpu'].copy_(tensors['inputs_host'], non_blocking=True)
    
    def benchmark_gemv_kernel(self, batch_size: int, input_dim: int, output_dim: int) -> Dict:
        """Benchmark custom CUDA GEMV kernel"""
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_gemv(batch_size, input_dim, output_dim)
        
        tensors = self.allocate_pinned_tensors(batch_size, input_dim, output_dim)
        latencies = []
        
        # Warmup runs
        for _ in range(self.config.num_warmup):
            self.async_h2d_copy(tensors)
            self.stream.synchronize()
            cuda_task_queue.batched_gemv(
                tensors['weights_gpu'],
                tensors['inputs_gpu'], 
                tensors['outputs_gpu']
            )
            torch.cuda.synchronize()
        
        # Benchmark runs with precise timing
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
        """Benchmark custom CUDA softmax kernel"""
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_softmax(batch_size, dim)
        
        inputs_host = torch.randn(batch_size, dim, pin_memory=True)
        outputs_host = torch.zeros(batch_size, dim, pin_memory=True)
        inputs_gpu = torch.empty(batch_size, dim, device=self.device)
        outputs_gpu = torch.empty(batch_size, dim, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            inputs_gpu.copy_(inputs_host, non_blocking=True)
            torch.cuda.synchronize()
            cuda_task_queue.batched_softmax(inputs_gpu, outputs_gpu)
            torch.cuda.synchronize()
        
        # Benchmark
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
        """Benchmark price vector processing kernel"""
        if not CUDA_AVAILABLE:
            return self._fallback_pytorch_price_vectors(batch_size, n_assets, n_features)
        
        prices_host = torch.randn(batch_size, n_assets, pin_memory=True) * 100  # Realistic prices
        weights_host = torch.randn(n_assets, n_features, pin_memory=True)
        features_host = torch.zeros(batch_size, n_features, pin_memory=True)
        
        prices_gpu = torch.empty(batch_size, n_assets, device=self.device)
        weights_gpu = torch.empty(n_assets, n_features, device=self.device)
        features_gpu = torch.empty(batch_size, n_features, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            prices_gpu.copy_(prices_host, non_blocking=True)
            weights_gpu.copy_(weights_host, non_blocking=True)
            torch.cuda.synchronize()
            cuda_task_queue.process_price_vectors(prices_gpu, weights_gpu, features_gpu)
            torch.cuda.synchronize()
        
        # Benchmark
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
        """PyTorch baseline for GEMV operation"""
        weights = torch.randn(batch_size, input_dim, output_dim, device=self.device)
        inputs = torch.randn(batch_size, input_dim, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            torch.bmm(inputs.unsqueeze(1), weights).squeeze(1)
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.bmm(inputs.unsqueeze(1), weights).squeeze(1)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_GEMV")
    
    def _fallback_pytorch_softmax(self, batch_size: int, dim: int) -> Dict:
        """PyTorch baseline for softmax operation"""
        inputs = torch.randn(batch_size, dim, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            torch.softmax(inputs, dim=1)
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.softmax(inputs, dim=1)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_Softmax")
    
    def _fallback_pytorch_price_vectors(self, batch_size: int, n_assets: int, n_features: int) -> Dict:
        """PyTorch baseline for price vector processing"""
        prices = torch.randn(batch_size, n_assets, device=self.device) * 100
        weights = torch.randn(n_assets, n_features, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(self.config.num_warmup):
            torch.mm(prices, weights)
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_trials):
            self.start_event.record()
            torch.mm(prices, weights)
            self.end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(self.start_event.elapsed_time(self.end_event))
        
        return self._compute_stats(latencies, "PyTorch_PriceVectors")
    
    def _compute_stats(self, latencies: List[float], kernel_name: str) -> Dict:
        """Compute comprehensive latency statistics"""
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
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run all benchmarks across parameter sweep"""
        results = {}
        
        print("🚀 Starting GPU Task Queue Benchmark")
        print(f"Device: {self.device}")
        print(f"Trials per config: {self.config.num_trials}")
        
        # GEMV benchmark sweep
        for batch_size in self.config.batch_sizes:
            for input_dim in self.config.input_dims:
                for output_dim in self.config.output_dims:
                    key = f"gemv_b{batch_size}_i{input_dim}_o{output_dim}"
                    print(f"⚡ Benchmarking {key}...")
                    
                    results[key] = self.benchmark_gemv_kernel(batch_size, input_dim, output_dim)
        
        # Softmax benchmark sweep
        for batch_size in self.config.batch_sizes:
            for dim in self.config.input_dims:
                key = f"softmax_b{batch_size}_d{dim}"
                print(f"⚡ Benchmarking {key}...")
                
                results[key] = self.benchmark_softmax_kernel(batch_size, dim)
        
        # Price vector benchmark sweep
        for batch_size in self.config.batch_sizes:
            key = f"price_b{batch_size}_a64_f32"
            print(f"⚡ Benchmarking {key}...")
            
            results[key] = self.benchmark_price_vectors(batch_size, 64, 32)
        
        self.results = results
        return results
    
    def plot_results(self, save_path: str = "benchmark_results.png"):
        """Generate performance visualization"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("GPU Task Queue Performance Analysis", fontsize=16)
        
        # Latency comparison
        kernels = [r['kernel'] for r in self.results.values()]
        medians = [r['median_ms'] for r in self.results.values()]
        
        axes[0, 0].bar(range(len(kernels)), medians, color='skyblue')
        axes[0, 0].set_title("Median Latency by Kernel")
        axes[0, 0].set_ylabel("Latency (ms)")
        axes[0, 0].set_xticks(range(len(kernels)))
        axes[0, 0].set_xticklabels(kernels, rotation=45, ha='right')
        
        # P95 vs Median
        p95s = [r['p95_ms'] for r in self.results.values()]
        axes[0, 1].scatter(medians, p95s, alpha=0.7, color='coral')
        axes[0, 1].plot([0, max(medians)], [0, max(medians)], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel("Median Latency (ms)")
        axes[0, 1].set_ylabel("P95 Latency (ms)")
        axes[0, 1].set_title("Latency Tail Distribution")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results saved to {save_path}")
    
    def print_summary(self):
        """Print benchmark summary to console"""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        print("\n" + "="*80)
        print("🎯 BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for key, stats in self.results.items():
            print(f"\n{key}:")
            print(f"  Kernel: {stats['kernel']}")
            print(f"  Median: {stats['median_ms']:.3f}ms")
            print(f"  P95: {stats['p95_ms']:.3f}ms")
            print(f"  Mean: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")
        
        # Find best performing kernel
        cuda_results = {k: v for k, v in self.results.items() if 'CUDA' in v['kernel']}
        if cuda_results:
            best = min(cuda_results.items(), key=lambda x: x[1]['median_ms'])
            print(f"\n🏆 Best Performance: {best[0]} with {best[1]['median_ms']:.3f}ms median latency")