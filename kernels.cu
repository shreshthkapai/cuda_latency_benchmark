#include <cuda_runtime.h>
#include <cooperative_groups.h>

#ifdef _WIN32
    #define NVTX_DISABLE
#endif

#ifdef NVTX_DISABLE
    #define nvtxRangePush(name) 
    #define nvtxRangePop()
#else
    #include <nvtx3/nvToolsExt.h>
#endif

namespace cg = cooperative_groups;

// Keep the winning GEMV kernel exactly as it was
__global__ void batched_gemv_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    int batch_size,
    int input_dim,
    int output_dim
) {
    int batch_idx = blockIdx.x;
    int output_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || output_idx >= output_dim) return;
    
    extern __shared__ float shared_input[];
    auto block = cg::this_thread_block();
    
    const float* input_ptr = inputs + batch_idx * input_dim;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    if ((input_dim & 3) == 0 && ((uintptr_t)input_ptr & 15) == 0) {
        float4* shared_input4 = (float4*)shared_input;
        const float4* input_ptr4 = (const float4*)input_ptr;
        
        for (int i = tid; i * 4 < input_dim; i += num_threads) {
            if (i * 4 < input_dim) {
                shared_input4[i] = __ldg(&input_ptr4[i]);
            }
        }
    } else {
        for (int i = tid; i < input_dim; i += num_threads) {
            shared_input[i] = __ldg(&input_ptr[i]);
        }
    }
    __syncthreads();
    
    float result = 0.0f;
    const float* weight_row = weights + batch_idx * input_dim * output_dim + output_idx;
    
    int i = 0;
    for (; i <= input_dim - 8; i += 8) {
        result += shared_input[i] * weight_row[i * output_dim] +
                  shared_input[i+1] * weight_row[(i+1) * output_dim] +
                  shared_input[i+2] * weight_row[(i+2) * output_dim] +
                  shared_input[i+3] * weight_row[(i+3) * output_dim] +
                  shared_input[i+4] * weight_row[(i+4) * output_dim] +
                  shared_input[i+5] * weight_row[(i+5) * output_dim] +
                  shared_input[i+6] * weight_row[(i+6) * output_dim] +
                  shared_input[i+7] * weight_row[(i+7) * output_dim];
    }
    for (; i < input_dim; i++) {
        result += shared_input[i] * weight_row[i * output_dim];
    }
    
    outputs[batch_idx * output_dim + output_idx] = result;
}

// Back to working softmax with just block size tweak
__global__ void batched_softmax_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    const float* input_batch = inputs + batch_idx * dim;
    float* output_batch = outputs + batch_idx * dim;
    
    // Simple max reduction
    float local_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, input_batch[i]);
    }
    
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = sdata[0];
    
    // Simple exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = __expf(input_batch[i] - global_max);
        output_batch[i] = exp_val;
        local_sum += exp_val;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float total_sum = sdata[0];
    
    // Simple normalization
    float inv_sum = __fdividef(1.0f, total_sum);
    for (int i = tid; i < dim; i += blockDim.x) {
        output_batch[i] *= inv_sum;
    }
}

// Back to simple price vectors  
__global__ void process_price_vectors_kernel(
    const float* __restrict__ prices,
    const float* __restrict__ weights,
    float* __restrict__ features,
    int batch_size,
    int n_assets,
    int n_features
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || feature_idx >= n_features) return;
    
    const float* price_vector = prices + batch_idx * n_assets;
    float result = 0.0f;
    
    // Simple dot product with unrolling
    int i = 0;
    for (; i <= n_assets - 4; i += 4) {
        result += price_vector[i] * weights[i * n_features + feature_idx] +
                  price_vector[i+1] * weights[(i+1) * n_features + feature_idx] +
                  price_vector[i+2] * weights[(i+2) * n_features + feature_idx] +
                  price_vector[i+3] * weights[(i+3) * n_features + feature_idx];
    }
    for (; i < n_assets; i++) {
        result += price_vector[i] * weights[i * n_features + feature_idx];
    }
    
    features[batch_idx * n_features + feature_idx] = result;
}

extern "C" {

void launch_batched_gemv(
    const float* weights, const float* inputs, float* outputs,
    int batch_size, int input_dim, int output_dim,
    cudaStream_t stream = 0
) {
    nvtxRangePush("batched_gemv");
    
    dim3 grid(batch_size);
    dim3 block(min(output_dim, 1024));
    int shared_mem = ((input_dim * sizeof(float) + 127) & ~127);
    
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
    dim3 block(64); // Try smaller block size
    int shared_mem = block.x * sizeof(float);
    
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
    dim3 block(min(n_features, 256)); // Try smaller block
    
    process_price_vectors_kernel<<<grid, block, 0, stream>>>(
        prices, weights, features, batch_size, n_assets, n_features
    );
    
    nvtxRangePop();
}

}
