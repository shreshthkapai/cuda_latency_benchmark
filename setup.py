import os
#Set CUDA_HOME FIRST before any other imports
os.environ['CUDA_HOME'] = '/usr'
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import pybind11

# Detect CUDA capability for optimal compilation
def get_cuda_arch():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return f"sm_{capability[0]}{capability[1]}"
    return "sm_75"  # Default to GTX 1650

# Windows-compatible compilation flags
cuda_flags = [
    "--use_fast_math",           # Fast math operations
    "-O3",                       # Maximum optimization
    f"-gencode=arch=compute_75,code={get_cuda_arch()}",  # Target architecture
    "--extended-lambda",         # C++11 lambda support
    "-DNVTX_DISABLE",           # Disable NVTX on Windows
]

cpp_flags = [
    "-O3",                      # Windows optimization flag
    "-std=c++17",              # C++17 standard
    "-DWITH_CUDA",
    "-DNVTX_DISABLE",          # Disable NVTX on Windows
]

# CUDA extension with optimized kernels
cuda_extension = CUDAExtension(
    name="cuda_task_queue",
    sources=[
        "kernels.cu",
        "python_bindings.cpp"
    ],
    extra_compile_args={
        "cxx": cpp_flags,
        "nvcc": cuda_flags
    },
    include_dirs=[
        pybind11.get_cmake_dir() + "/../../../include",
    ],
)

setup(
    name="cuda-latency-bench",
    version="1.0.0",
    description="Sub-millisecond GPU task queue for real-time inference",
    author="GPU Performance Engineer",
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pybind11>=2.6.0",
    ],
    zip_safe=False,
)
