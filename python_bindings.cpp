#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Forward declarations from kernels.cu
extern "C" {
void launch_batched_gemv(
    const float* weights, const float* inputs, float* outputs,
    int batch_size, int input_dim, int output_dim,
    cudaStream_t stream
);

void launch_batched_softmax(
    const float* inputs, float* outputs,
    int batch_size, int dim,
    cudaStream_t stream
);

void launch_price_vector_processing(
    const float* prices, const float* weights, float* features,
    int batch_size, int n_assets, int n_features,
    cudaStream_t stream
);
}

// PyTorch tensor wrappers for zero-copy GPU operations
torch::Tensor batched_gemv(
    torch::Tensor weights,
    torch::Tensor inputs,
    torch::Tensor outputs
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(outputs.is_cuda(), "outputs must be CUDA tensor");

    int batch_size = inputs.size(0);
    int input_dim = inputs.size(1);
    int output_dim = outputs.size(1);

    launch_batched_gemv(
        weights.data_ptr<float>(),
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        batch_size, input_dim, output_dim,
        c10::cuda::getCurrentCUDAStream()
    );

    return outputs;
}

torch::Tensor batched_softmax(torch::Tensor inputs, torch::Tensor outputs) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensor");
    TORCH_CHECK(outputs.is_cuda(), "outputs must be CUDA tensor");

    int batch_size = inputs.size(0);
    int dim = inputs.size(1);

    launch_batched_softmax(
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        batch_size, dim,
        c10::cuda::getCurrentCUDAStream()
    );

    return outputs;
}

torch::Tensor process_price_vectors(
    torch::Tensor prices,
    torch::Tensor weights,
    torch::Tensor features
) {
    TORCH_CHECK(prices.is_cuda(), "prices must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(features.is_cuda(), "features must be CUDA tensor");

    int batch_size = prices.size(0);
    int n_assets = prices.size(1);
    int n_features = features.size(1);

    launch_price_vector_processing(
        prices.data_ptr<float>(),
        weights.data_ptr<float>(),
        features.data_ptr<float>(),
        batch_size, n_assets, n_features,
        c10::cuda::getCurrentCUDAStream()
    );

    return features;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA task queue kernels for sub-millisecond inference";

    m.def("batched_gemv", &batched_gemv, "Batched matrix-vector multiply");
    m.def("batched_softmax", &batched_softmax, "Batched softmax operation");
    m.def("process_price_vectors", &process_price_vectors, "Price vector processing");
}