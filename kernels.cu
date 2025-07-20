#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Windows NVTX fix
#ifdef _WIN32
    #define NVTX_DISABLE  // Disable NVTX on Windows for now
#endif

#ifdef NVTX_DISABLE
    // Dummy NVTX macros for Windows
    #define nvtxRangePush(name) 
    #define nvtxRangePop()
#else
    #include <nvtx3/nvToolsExt.h>
#endif

namespace cg = cooperative_groups;

// Fast batched GEMV kernel for small vectors (16-128 dims)
__global__ void batched_gemv_kernel(
    const float* __restrict__ weights,    // [batch_size, input_dim, output_dim]
    const float* __restrict__ inputs,     // [batch_size, input_dim]
    float* __restrict__ outputs,          // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int output_dim
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || output_idx >= output_dim) return;
    
    // Shared memory for input vector (coalesced loading)
    extern __shared__ float shared_input[];
    
    auto block = cg::this_thread_block();
    
    // Load input vector into shared memory
    if (output_idx < input_dim) {
        shared_input[output_idx] = inputs[batch_idx * input_dim + output_idx];
    }
    block.sync();
    
    // Compute dot product for this output dimension
    float result = 0.0f;
    const float* weight_row = weights + (batch_idx * input_dim + 0) * output_dim + output_idx;
    
    #pragma unroll 8
    for (int i = 0; i < input_dim; i++) {
        result += shared_input[i] * weight_row[i * output_dim];
    }
    
    outputs[batch_idx * output_dim + output_idx] = result;
}

// Optimized softmax with single-pass reduction
__global__ void batched_softmax_kernel(
    const float* __restrict__ inputs,     // [batch_size, dim]
    float* __restrict__ outputs,          // [batch_size, dim]
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    float* shared_max = sdata;
    float* shared_sum = sdata + blockDim.x;
    
    const float* input_batch = inputs + batch_idx * dim;
    float* output_batch = outputs + batch_idx * dim;
    
    auto block = cg::this_thread_block();
    
    // Find max value (parallel reduction)
    float local_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, input_batch[i]);
    }
    shared_max[tid] = local_max;
    block.sync();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        block.sync();
    }
    float global_max = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(input_batch[i] - global_max);
        output_batch[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    block.sync();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        block.sync();
    }
    float total_sum = shared_sum[0];
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        output_batch[i] /= total_sum;
    }
}

// High-throughput price vector processing kernel
__global__ void process_price_vectors_kernel(
    const float* __restrict__ prices,     // [batch_size, n_assets]
    const float* __restrict__ weights,    // [n_assets, n_features]
    float* __restrict__ features,         // [batch_size, n_features]
    int batch_size,
    int n_assets,
    int n_features
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || feature_idx >= n_features) return;
    
    const float* price_vector = prices + batch_idx * n_assets;
    float result = 0.0f;
    
    // Vectorized dot product with manual unrolling
    #pragma unroll 4
    for (int i = 0; i < n_assets; i++) {
        result += price_vector[i] * weights[i * n_features + feature_idx];
    }
    
    features[batch_idx * n_features + feature_idx] = result;
}

// C interface for Python interop
extern "C" {

void launch_batched_gemv(
    const float* weights, const float* inputs, float* outputs,
    int batch_size, int input_dim, int output_dim,
    cudaStream_t stream = 0
) {
    nvtxRangePush("batched_gemv");
    
    dim3 grid(batch_size);
    dim3 block(min(output_dim, 1024));
    int shared_mem = input_dim * sizeof(float);
    
    batched_gemv_kernel<<<grid, block, shared_mem, stream>>>(
        weights, inputs, outputs, batch_size, input_dim, output_dim
    );
    
    nvtxRangePop();
}

void launch_batched_softmax(
    const float* inputs, float* outputs,
    int batch_size, int dim,
    cudaStream_t stream = 0
) {
    nvtxRangePush("batched_softmax");
    
    dim3 grid(batch_size);
    dim3 block(min(dim, 1024));
    int shared_mem = 2 * block.x * sizeof(float);
    
    batched_softmax_kernel<<<grid, block, shared_mem, stream>>>(
        inputs, outputs, batch_size, dim
    );
    
    nvtxRangePop();
}

void launch_price_vector_processing(
    const float* prices, const float* weights, float* features,
    int batch_size, int n_assets, int n_features,
    cudaStream_t stream = 0
) {
    nvtxRangePush("price_vectors");
    
    dim3 grid(batch_size);
    dim3 block(min(n_features, 1024));
    
    process_price_vectors_kernel<<<grid, block, 0, stream>>>(
        prices, weights, features, batch_size, n_assets, n_features
    );
    
    nvtxRangePop();
}

} // extern "C"
