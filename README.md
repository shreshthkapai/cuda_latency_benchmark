# Sub-Millisecond GPU Task Queue for Real-Time Inference

This project implements a high-performance, sub-millisecond GPU task queue designed for real-time inference and other latency-sensitive workloads. By leveraging custom CUDA kernels, it provides significant performance improvements over standard PyTorch implementations for specific operations like batched GEMV (Generalized Matrix-Vector Multiplication), softmax, and financial feature engineering.

The core of the repository is a C++/CUDA extension for PyTorch, which is benchmarked to demonstrate its low-latency capabilities. The system is designed for scenarios where minimizing computational latency is critical, such as in high-frequency trading, real-time bidding, or interactive services.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Highlights](#performance-highlights)
- [Technical Specifications](#technical-specifications)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Building the Extension](#building-the-extension)
- [Running Benchmarks](#running-benchmarks)
- [Core Components](#core-components)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements a high-performance, sub-millisecond GPU task queue designed for real-time inference and other latency-sensitive workloads. By leveraging custom CUDA kernels, it provides significant performance improvements over standard PyTorch implementations for specific operations like batched GEMV (Generalized Matrix-Vector Multiplication), softmax, and financial feature engineering.

The core of the repository is a C++/CUDA extension for PyTorch, which is benchmarked to demonstrate its low-latency capabilities. The system is designed for scenarios where minimizing computational latency is critical, such as in high-frequency trading, real-time bidding, or interactive services.

## Key Features
- **Custom CUDA Kernels**: Highly-optimized kernels for common machine learning and financial tasks (batched_gemv, batched_softmax, process_price_vectors).
- **Low-Level Optimization**: Utilizes shared memory, vectorized operations, and manual loop unrolling for maximum instruction-level parallelism.
- **Asynchronous Execution**: Leverages CUDA streams for efficient data transfer and kernel execution, overlapping computation and memory copies.
- **PyTorch Integration**: Seamlessly exposed to Python via Pybind11 and PyTorch's C++ extension mechanism, allowing for easy integration into existing ML pipelines.
- **Comprehensive Benchmarking**: Includes a robust benchmarking suite to measure and analyze kernel performance, including latency statistics (median, P95, P99) and throughput.
- **Automated Tooling**: A `run_all.sh` script automates the entire process of cleaning, building, benchmarking, and reporting.

## Performance Highlights
The benchmarks below were executed on the hardware specified in the Technical Specifications section. They demonstrate the exceptional performance of the custom CUDA kernels, achieving sub-millisecond latencies across various configurations.

```
🚀 GPU TASK QUEUE PERFORMANCE REPORT
================================================================================

🏆 Best Performer: gemv_b32_i64_o32 (0.008ms median)
🐌 Worst Performer: price_b32_a64_f32 (0.043ms median)
⚡ Average Speedup: 4.5x
🚀 Maximum Speedup: 7.5x

📊 DETAILED RESULTS:
--------------------------------------------------------------------------------

gemv_b32_i64_o32:
  Latency: 0.008ms (median), 0.032ms (P95)
  Throughput: 131579 ops/sec
  🚀 Speedup: 7.5x (650.1% improvement)
  Stability: ±0.009ms std dev

gemv_b32_i64_o64:
  Latency: 0.010ms (median), 0.078ms (P95)
  Throughput: 96154 ops/sec
  🚀 Speedup: 5.1x (414.5% improvement)
  Stability: ±0.020ms std dev

softmax_b32_d64:
  Latency: 0.040ms (median), 0.123ms (P95)
  Throughput: 24733 ops/sec
  🚀 Speedup: 1.2x (23.2% improvement)
  Stability: ±0.145ms std dev

price_b32_a64_f32:
  Latency: 0.043ms (median), 0.069ms (P95)
  Throughput: 23391 ops/sec
  🚀 Speedup: 4.2x (319.6% improvement)
  Stability: ±0.014ms std dev
```

## Technical Specifications
The performance benchmarks were conducted on the following hardware:

- **GPU**: NVIDIA GeForce GTX 1650
- **VRAM**: 4 GB GDDR6
- **CUDA Compute Capability**: 7.5

## Architecture
The project is composed of several key components that work together:

- **CUDA Kernels (`kernels.cu`)**: The C++ source file containing the low-level CUDA C++ code for the `batched_gemv`, `batched_softmax`, and `process_price_vectors` kernels. This is where the core GPU computations are defined.
- **Python Bindings (`python_bindings.cpp`)**: This file uses Pybind11 to create a bridge between the C++ functions that launch the CUDA kernels and the Python interpreter. It converts PyTorch tensors into C++ pointers that can be used by the kernels.
- **Setup Script (`setup.py`)**: A standard Python setuptools script used to compile the CUDA and C++ code into a Python extension module named `cuda_task_queue`.
- **Benchmarking Framework (`benchmark.py`)**: A Python class that orchestrates the performance tests. It handles tensor allocation, GPU synchronization, and runs both the custom CUDA kernels and their PyTorch equivalents for comparison.
- **Utility Functions (`utils.py`)**: Helper functions for timing, statistical computation, plotting results, and generating reports.
- **Automation Script (`run_all.sh`)**: A bash script that automates the entire workflow: cleaning old builds, compiling the CUDA extension, running the benchmarks, and generating a final performance report.

## Repository Structure
```
.
├── kernels.cu
├── python_bindings.cpp
├── setup.py
├── benchmark.py
├── utils.py
└── runa_all.sh
```

## Building the Extension
The CUDA kernels are compiled into a Python module using `setuptools`. The `run_all.sh` script automates this process.

### Prerequisites:
- NVIDIA CUDA Toolkit (`nvcc`)
- PyTorch with CUDA support
- Pybind11

### Build Steps:
1. Navigate to the project root directory.
2. Run the setup script:
```bash
python setup.py build_ext --inplace
```
This command will compile the `kernels.cu` and `python_bindings.cpp` files and create a `cuda_task_queue*.so` file in the current directory, which is the importable Python module.

## Running Benchmarks
The `run_all.sh` script is the easiest way to build the extension and run the full benchmark suite.

### Usage:
```bash
# Run with default parameters (batch_size=32, dim=64, repeats=100)
./run_all.sh

# Run with custom parameters
# Usage: ./run_all.sh [BATCH_SIZE] [DIM] [REPEATS]
./run_all.sh 64 128 200
```
The script will:
1. Clean any previous builds.
2. Compile the CUDA kernels.
3. Run the benchmark suite with the specified parameters.
4. Print a summary to the console and save detailed results and plots to the `./results/` directory.

## Core Components

### CUDA Kernels
- **`batched_gemv_kernel`**: Optimized for small vectors, this kernel uses shared memory to cache input vectors, enabling coalesced memory access and reducing global memory bandwidth consumption.
- **`batched_softmax_kernel`**: Implements a single-pass parallel reduction algorithm to find the max value and sum for the softmax calculation, significantly improving efficiency over naive approaches.
- **`process_price_vectors_kernel`**: A high-throughput kernel for financial applications, performing a batched dot product with vectorized memory access and manual unrolling to maximize throughput.

### Benchmarking
The `GPUTaskQueueBenchmark` class in `benchmark.py` provides a structured way to evaluate performance. It uses CUDA events for precise timing (`torch.cuda.Event`) and pre-allocates pinned memory (`pin_memory=True`) to enable fast, asynchronous data transfers between the host and the GPU.

## Dependencies
- `torch >= 1.12.0`
- `numpy >= 1.21.0`
- `pybind11 >= 2.6.0`
- `matplotlib`
- `seaborn`

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improvements.

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

