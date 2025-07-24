import torch
import csv
import json
import time
from pathlib import Path
from typing import Dict

def save_results_csv(results_data: Dict, filename: str = "benchmark_results.csv") -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    results_dict = results_data.get('optimized', {})
    baseline_dict = results_data.get('baseline', {})
    speedup_dict = results_data.get('speedup', {})

    if not results_dict:
        print("No optimized results to save.")
        return

    all_keys = set()
    for stats in results_dict.values():
        all_keys.update(stats.keys())

    fieldnames = ['config'] + sorted(all_keys)
    if baseline_dict:
        fieldnames += ['baseline_median_ms']
    if speedup_dict:
        extra_keys = set()
        for s in speedup_dict.values():
            extra_keys.update(s.keys())
        fieldnames += sorted(extra_keys)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for config, stats in results_dict.items():
            row = {'config': config, **stats}
            if config in baseline_dict:
                row['baseline_median_ms'] = baseline_dict[config].get('median_ms')
            if config in speedup_dict:
                row.update(speedup_dict[config])
            writer.writerow(row)

    print(f"💾 Results successfully exported to {filename}")

def save_results_json(results_data: Dict, filename: str = "benchmark_results.json") -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device_info': {
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__
        },
        'benchmark_results': results_data
    }
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"💾 Results and metadata successfully saved to {filename}")

def merge_results(results_data):
    merged = {}
    for k in results_data.get("optimized", {}):
        merged[k] = results_data["optimized"][k].copy()
        if k in results_data.get("speedup", {}):
            merged[k].update(results_data["speedup"][k])
    return merged

def format_performance_report(results_data: Dict) -> str:
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("🚀 GPU TASK QUEUE PERFORMANCE REPORT")
    report_lines.append("="*80)

    if not results_data:
        return "No results available."

    latencies = {k: v['median_ms'] for k, v in results_data.items() if 'median_ms' in v}
    if not latencies:
        return "No valid latency data found."

    best_kernel = min(latencies.items(), key=lambda x: x[1])
    worst_kernel = max(latencies.items(), key=lambda x: x[1])
    speedups = [v.get("speedup_median", 0.0) for v in results_data.values()]
    avg_speedup = sum(speedups) / len(speedups)
    geo_speedup = 1.0
    for s in speedups:
        geo_speedup *= s
    geo_speedup = geo_speedup ** (1 / len(speedups)) if speedups else 1.0
    max_speedup = max(speedups)

    report_lines.append("")
    report_lines.append(f"🏆 Best Performer: {best_kernel[0]} ({best_kernel[1]:.3f}ms median)")
    report_lines.append(f"🐌 Worst Performer: {worst_kernel[0]} ({worst_kernel[1]:.3f}ms median)")
    report_lines.append(f"⚡ Average Speedup: {avg_speedup:.1f}x")
    report_lines.append(f"🚀 Maximum Speedup: {max_speedup:.1f}x")
    report_lines.append("")

    report_lines.append("📊 DETAILED RESULTS:")
    report_lines.append("-"*80)

    for kernel, stats in results_data.items():
        median_ms = stats.get('median_ms', 0.0)
        p95_ms = stats.get('p95_ms', 0.0)
        std_ms = stats.get('std_ms', 0.0)
        throughput = 1000.0 / median_ms if median_ms > 0 else 0.0

        report_lines.append(f"\n{kernel}:")
        report_lines.append(f"  Latency: {median_ms:.3f}ms (median), {p95_ms:.3f}ms (P95)")
        report_lines.append(f"  Throughput: {throughput:.0f} ops/sec")
        report_lines.append(f"  🚀 Speedup: {stats.get('speedup_median', 0.0):.1f}x ({stats.get('improvement_pct', 0.0):.1f}% improvement)")
        report_lines.append(f"  Stability: ±{std_ms:.3f}ms std dev")

    return "\n".join(report_lines)
