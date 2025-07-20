import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import json
import time
from pathlib import Path
from typing import Callable, Any, List, Dict, Tuple
from contextlib import contextmanager
import warnings

# Suppress matplotlib user warnings for improved output clarity
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class CUDATimer:
    """Provides high-precision timing for CUDA kernel benchmarking using CUDA events."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    @contextmanager
    def time_block(self):
        """Context manager for measuring the duration of CUDA operations."""
        self.start_event.record()
        yield
        self.end_event.record()
        torch.cuda.synchronize()
    
    def elapsed_ms(self) -> float:
        """Returns the elapsed time between events in milliseconds."""
        return self.start_event.elapsed_time(self.end_event)

def time_cuda_kernel(func: Callable, args: Tuple, stream: torch.cuda.Stream = None, 
                    warmup_runs: int = 10, benchmark_runs: int = 100) -> Dict[str, float]:
    """
    Measures execution time of a CUDA kernel with warmup and returns detailed statistics.
    
    Args:
        func: The CUDA kernel function to be benchmarked.
        args: Arguments to be passed to the kernel function.
        stream: Optional CUDA stream for asynchronous execution.
        warmup_runs: Number of warmup iterations prior to benchmarking.
        benchmark_runs: Number of iterations used for benchmarking.
    
    Returns:
        A dictionary containing statistical timing metrics.
    """
    timer = CUDATimer()
    
    # Perform warmup iterations to reduce cold-start effects
    for _ in range(warmup_runs):
        if stream:
            with torch.cuda.stream(stream):
                func(*args)
        else:
            func(*args)
        torch.cuda.synchronize()
    
    # Benchmark iterations with precise timing
    latencies = []
    for _ in range(benchmark_runs):
        if stream:
            with torch.cuda.stream(stream):
                with timer.time_block():
                    func(*args)
        else:
            with timer.time_block():
                func(*args)
        
        latencies.append(timer.elapsed_ms())
    
    return compute_latency_stats(latencies)

def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculates and returns a set of latency statistics from the provided measurements."""
    latencies_array = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies_array)),
        'median_ms': float(np.median(latencies_array)),
        'std_ms': float(np.std(latencies_array)),
        'min_ms': float(np.min(latencies_array)),
        'max_ms': float(np.max(latencies_array)),
        'p90_ms': float(np.percentile(latencies_array, 90)),
        'p95_ms': float(np.percentile(latencies_array, 95)),
        'p99_ms': float(np.percentile(latencies_array, 99)),
        'samples': len(latencies)
    }

def plot_latency_distribution(latencies: List[float], title: str = "Kernel Latency Distribution",
                            save_path: str = None, show_stats: bool = True) -> None:
    """
    Generates and displays a latency distribution plot with optional statistics annotation.
    
    Args:
        latencies: List of measured latencies in milliseconds.
        title: Title for the plot.
        save_path: Optional file path to save the plot image.
        show_stats: If True, displays summary statistics on the plot.
    """
    # Apply a modern plotting style for readability
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histogram with kernel density estimate (KDE)
    ax1.hist(latencies, bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black')
    
    # Overlay KDE curve if scipy is available
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(latencies)
        x_range = np.linspace(min(latencies), max(latencies), 200)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    except ImportError:
        pass  # KDE is omitted if scipy is not installed
    
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Distribution')
    ax1.legend()
    
    # Add a box plot for quartile visualization
    ax2.boxplot(latencies, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title(f'{title} - Quartiles')
    ax2.grid(True, alpha=0.3)
    
    # Optionally display summary statistics on the plot
    if show_stats:
        stats = compute_latency_stats(latencies)
        stats_text = f"Mean: {stats['mean_ms']:.3f}ms\n" \
                    f"Median: {stats['median_ms']:.3f}ms\n" \
                    f"P95: {stats['p95_ms']:.3f}ms\n" \
                    f"P99: {stats['p99_ms']:.3f}ms"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()

def plot_comparative_performance(results_dict: Dict[str, Dict], save_path: str = None) -> None:
    """
    Creates comparative visualizations of performance metrics across multiple configurations.
    
    Args:
        results_dict: Dictionary mapping configuration names to their statistics.
        save_path: Optional file path to save the generated plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    configs = list(results_dict.keys())
    median_latencies = [results_dict[cfg]['median_ms'] for cfg in configs]
    p95_latencies = [results_dict[cfg]['p95_ms'] for cfg in configs]
    throughputs = [1000.0 / results_dict[cfg]['median_ms'] for cfg in configs]  # ops/sec
    
    # Bar chart for median latency
    bars1 = ax1.bar(range(len(configs)), median_latencies, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Median Latency (ms)')
    ax1.set_title('Median Latency Comparison')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    
    # Annotate bars with values
    for bar, latency in zip(bars1, median_latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{latency:.3f}ms', ha='center', va='bottom', fontsize=9)
    
    # Scatter plot for median vs. P95 latency
    ax2.scatter(median_latencies, p95_latencies, c=throughputs, cmap='viridis', s=100, alpha=0.7)
    ax2.plot([min(median_latencies), max(median_latencies)], 
             [min(median_latencies), max(median_latencies)], 'k--', alpha=0.5)
    ax2.set_xlabel('Median Latency (ms)')
    ax2.set_ylabel('P95 Latency (ms)')
    ax2.set_title('Tail Latency Analysis')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Throughput (ops/sec)')
    
    # Bar chart for throughput
    bars3 = ax3.bar(range(len(configs)), throughputs, color='coral', alpha=0.8)
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Throughput (ops/sec)')
    ax3.set_title('Throughput Comparison')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    
    # Heatmap to visualize latency variance and other metrics
    variances = [results_dict[cfg]['std_ms'] for cfg in configs]
    efficiency_matrix = np.array([median_latencies, p95_latencies, variances]).T
    
    im = ax4.imshow(efficiency_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4.set_title('Performance Heatmap')
    ax4.set_ylabel('Configuration')
    ax4.set_xlabel('Metric')
    ax4.set_yticks(range(len(configs)))
    ax4.set_yticklabels(configs)
    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(['Median', 'P95', 'StdDev'])
    plt.colorbar(im, ax=ax4, label='Latency (ms)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparative plot saved to {save_path}")
    
    plt.show()

def save_results_csv(results_dict: Dict[str, Dict], filename: str = "benchmark_results.csv") -> None:
    """Exports benchmark results to a CSV file for further analysis."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        if not results_dict:
            return
        
        # Extract field names from the first entry
        sample_stats = next(iter(results_dict.values()))
        fieldnames = ['config'] + list(sample_stats.keys())
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for config, stats in results_dict.items():
            row = {'config': config}
            row.update(stats)
            writer.writerow(row)
    
    print(f"💾 Results exported to {filename}")

def save_results_json(results_dict: Dict[str, Dict], filename: str = "benchmark_results.json") -> None:
    """Saves the benchmark results to a JSON file, including metadata such as timestamps and device information."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device_info': {
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__
        },
        'results': results_dict
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results with metadata saved to {filename}")

def format_performance_report(results_dict: Dict[str, Dict]) -> str:
    """Produces a formatted string report summarizing performance across all configurations."""
    if not results_dict:
        return "No results to report."
    
    report = ["=" * 80]
    report.append("🚀 GPU TASK QUEUE PERFORMANCE REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Identify the best and worst performing configurations
    median_latencies = [(cfg, stats['median_ms']) for cfg, stats in results_dict.items()]
    best_config, best_latency = min(median_latencies, key=lambda x: x[1])
    worst_config, worst_latency = max(median_latencies, key=lambda x: x[1])
    
    report.append(f"🏆 Best Performer: {best_config} ({best_latency:.3f}ms median)")
    report.append(f"🐌 Worst Performer: {worst_config} ({worst_latency:.3f}ms median)")
    report.append(f"⚡ Speedup: {worst_latency/best_latency:.1f}x improvement")
    report.append("")
    
    # Append detailed results for each configuration
    report.append("📊 DETAILED RESULTS:")
    report.append("-" * 80)
    
    for config, stats in results_dict.items():
        throughput = 1000.0 / stats['median_ms']
        report.append(f"\n{config}:")
        report.append(f"  Latency: {stats['median_ms']:.3f}ms (median), {stats['p95_ms']:.3f}ms (P95)")
        report.append(f"  Throughput: {throughput:.0f} ops/sec")
        report.append(f"  Stability: ±{stats['std_ms']:.3f}ms std dev")
    
    return "\n".join(report)

@contextmanager
def cuda_memory_profiler():
    """Context manager for monitoring CUDA memory allocation during code execution."""
    if not torch.cuda.is_available():
        yield None
        return
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    initial_memory = torch.cuda.memory_allocated()
    peak_memory = initial_memory
    
    class MemoryTracker:
        def __init__(self):
            self.initial = initial_memory
            self.peak = initial_memory
        
        def update(self):
            current = torch.cuda.memory_allocated()
            self.peak = max(self.peak, current)
        
        def get_stats(self):
            return {
                'initial_mb': self.initial / 1024**2,
                'peak_mb': self.peak / 1024**2,
                'allocated_mb': (self.peak - self.initial) / 1024**2
            }
    
    tracker = MemoryTracker()
    
    try:
        yield tracker
    finally:
        tracker.update()
        torch.cuda.synchronize()
