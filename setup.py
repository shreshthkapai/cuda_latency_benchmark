import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import pybind11

# Safe fallback for CUDA_HOME
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = torch.utils.cpp_extension.CUDA_HOME or "/usr/local/cuda"

# Get compute capability (e.g. sm_75 for GTX 1650)
def get_cuda_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return "-gencode=arch=compute_75,code=sm_75"

cuda_flags = [
    "--use_fast_math",
    "-O3",
    get_cuda_arch(),
    "--extended-lambda",
    "-DNVTX_DISABLE",
]

cpp_flags = [
    "-O3",
    "-std=c++17",
    "-DWITH_CUDA",
    "-DNVTX_DISABLE",
]

cuda_extension = CUDAExtension(
    name="cuda_task_queue",
    sources=[
        "kernels.cu",
        "python_bindings.cpp"
    ],
    extra_compile_args={
        "cxx": cpp_flags,
        "nvcc": cuda_flags,
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
