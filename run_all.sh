#!/bin/bash

# CUDA Task Queue - Production Build & Benchmark Automation
# Directory: scripts/run_all.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_BATCH_SIZE=32
DEFAULT_DIM=64
DEFAULT_REPEATS=100
RESULTS_DIR="./results"
BUILD_DIR="./build"

# Parse command line arguments
BATCH_SIZE=${1:-$DEFAULT_BATCH_SIZE}
DIM=${2:-$DEFAULT_DIM}
REPEATS=${3:-$DEFAULT_REPEATS}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}🚀 CUDA Task Queue Benchmark${NC}"
    echo -e "${BLUE}================================${NC}"
    echo -e "Batch Size: ${YELLOW}${BATCH_SIZE}${NC}"
    echo -e "Dimensions: ${YELLOW}${DIM}${NC}"
    echo -e "Repeats: ${YELLOW}${REPEATS}${NC}"
    echo -e "${BLUE}================================${NC}"
}

cleanup_build() {
    echo -e "${YELLOW}[0/4] Cleaning previous builds...${NC}"
    rm -rf ${BUILD_DIR} *.so cuda_task_queue.* || true
    mkdir -p ${RESULTS_DIR}
}

compile_kernels() {
    echo -e "${BLUE}[1/4] Compiling CUDA kernels...${NC}"
    
    # Check CUDA availability
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}❌ NVCC not found. Please install CUDA toolkit.${NC}"
        exit 1
    fi
    
    # Check PyTorch CUDA support
    python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA not available'" 2>/dev/null || {
        echo -e "${RED}❌ PyTorch CUDA support not detected.${NC}"
        exit 1
    }
    
    # Compile extension
    python setup.py build_ext --inplace || {
        echo -e "${RED}❌ CUDA compilation failed.${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ CUDA kernels compiled successfully${NC}"
}

run_benchmark() {
    echo -e "${BLUE}[2/4] Running performance benchmark...${NC}"
    
    # Create benchmark config
    python -c "
import sys
from benchmark import BenchmarkConfig, GPUTaskQueueBenchmark

config = BenchmarkConfig(
    batch_sizes=[${BATCH_SIZE}],
    input_dims=[${DIM}],
    output_dims=[${DIM}//2, ${DIM}],
    num_trials=${REPEATS}
)

benchmark = GPUTaskQueueBenchmark(config)
results = benchmark.run_comprehensive_benchmark()
benchmark.print_summary()
benchmark.plot_results('${RESULTS_DIR}/benchmark_plot.png')

# Save results
from utils import save_results_csv, save_results_json
save_results_csv(results, '${RESULTS_DIR}/results.csv')
save_results_json(results, '${RESULTS_DIR}/results.json')
" || {
        echo -e "${RED}❌ Benchmark execution failed.${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Benchmark completed successfully${NC}"
}

generate_report() {
    echo -e "${BLUE}[3/4] Generating performance report...${NC}"
    
    python -c "
import json
from utils import format_performance_report

with open('${RESULTS_DIR}/results.json', 'r') as f:
    data = json.load(f)

report = format_performance_report(data['results'])
print(report)

with open('${RESULTS_DIR}/performance_report.txt', 'w') as f:
    f.write(report)
"
    
    echo -e "${GREEN}✅ Report generated: ${RESULTS_DIR}/performance_report.txt${NC}"
}

finalize() {
    echo -e "${BLUE}[4/4] Finalizing results...${NC}"
    
    # List generated files
    echo -e "\n${GREEN}📊 Generated Files:${NC}"
    ls -la ${RESULTS_DIR}/ | grep -E '\.(csv|json|png|txt)$' | while read line; do
        echo -e "  ${YELLOW}•${NC} $line"
    done
    
    # Quick performance summary
    if [ -f "${RESULTS_DIR}/results.json" ]; then
        echo -e "\n${GREEN}⚡ Quick Summary:${NC}"
        python -c "
import json
with open('${RESULTS_DIR}/results.json', 'r') as f:
    data = json.load(f)
results = data['results']
if results:
    best_cuda = min([v for v in results.values() if 'CUDA' in v.get('kernel', '')], 
                   key=lambda x: x['median_ms'], default=None)
    if best_cuda:
        print(f'  Best CUDA Kernel: {best_cuda[\"median_ms\"]:.3f}ms median latency')
        print(f'  Throughput: {1000/best_cuda[\"median_ms\"]:.0f} ops/sec')
"
    fi
    
    echo -e "\n${GREEN}🎯 Benchmark completed! Check ${RESULTS_DIR}/ for detailed results.${NC}"
}

# Error handler
handle_error() {
    echo -e "\n${RED}❌ Error occurred. Cleaning up...${NC}"
    cleanup_build > /dev/null 2>&1 || true
    exit 1
}

# Set error trap
trap handle_error ERR

# Main execution
main() {
    print_header
    cleanup_build
    compile_kernels
    run_benchmark
    generate_report
    finalize
}

# Run with usage info
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [BATCH_SIZE] [DIM] [REPEATS]"
    echo "Example: $0 64 128 200"
    echo "Defaults: batch_size=32, dim=64, repeats=100"
    exit 0
fi

main "$@"

