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
    
    // Vectorized coalesced loading with 128-byte alignment
    const float* input_ptr = inputs + batch_idx * input_dim;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Load input using float4 when possible for 128-bit aligned access
    if ((input_dim & 3) == 0 && tid * 4 < input_dim) {
        float4* shared_input4 = (float4*)shared_input;
        const float4* input_ptr4 = (const float4*)input_ptr;
        
        for (int i = tid; i * 4 < input_dim; i += num_threads) {
            shared_input4[i] = input_ptr4[i];
        }
    } else {
        for (int i = tid; i < input_dim; i += num_threads) {
            shared_input[i] = input_ptr[i];
        }
    }
    block.sync();
    
    // Optimized computation with cache-aligned access
    float result = 0.0f;
    const float* weight_row = weights + batch_idx * input_dim * output_dim + output_idx;
    
    #pragma unroll 8
    for (int i = 0; i < input_dim; i++) {
        result += shared_input[i] * weight_row[i * output_dim];
    }
    
    outputs[batch_idx * output_dim + output_idx] = result;
}

__global__ void batched_softmax_kernel(
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Bank conflict-free shared memory layout
    extern __shared__ float sdata[];
    float* shared_max = sdata;
    float* shared_sum = sdata + (blockDim.x + 1);  // +1 to avoid bank conflicts
    
    const float* input_batch = inputs + batch_idx * dim;
    float* output_batch = outputs + batch_idx * dim;
    
    auto block = cg::this_thread_block();
    
    // Vectorized max reduction with coalesced access
    float local_max = -INFINITY;
    if ((dim & 3) == 0) {
        const float4* input_batch4 = (const float4*)input_batch;
        for (int i = tid; i * 4 < dim; i += blockDim.x) {
            float4 val = input_batch4[i];
            local_max = fmaxf(local_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        }
    } else {
        for (int i = tid; i < dim; i += blockDim.x) {
            local_max = fmaxf(local_max, input_batch[i]);
        }
    }
    
    shared_max[tid] = local_max;
    block.sync();
    
    // Efficient reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        block.sync();
    }
    float global_max = shared_max[0];
    
    // Vectorized exp and sum computation
    float local_sum = 0.0f;
    if ((dim & 3) == 0) {
        float4* output_batch4 = (float4*)output_batch;
        const float4* input_batch4 = (const float4*)input_batch;
        
        for (int i = tid; i * 4 < dim; i += blockDim.x) {
            float4 input_val = input_batch4[i];
            float4 exp_val;
            exp_val.x = expf(input_val.x - global_max);
            exp_val.y = expf(input_val.y - global_max);
            exp_val.z = expf(input_val.z - global_max);
            exp_val.w = expf(input_val.w - global_max);
            
            output_batch4[i] = exp_val;
            local_sum += exp_val.x + exp_val.y + exp_val.z + exp_val.w;
        }
    } else {
        for (int i = tid; i < dim; i += blockDim.x) {
            float exp_val = expf(input_batch[i] - global_max);
            output_batch[i] = exp_val;
            local_sum += exp_val;
        }
    }
    
    shared_sum[tid] = local_sum;
    block.sync();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        block.sync();
    }
    float total_sum = shared_sum[0];
    
    // Vectorized normalization
    if ((dim & 3) == 0) {
        float4* output_batch4 = (float4*)output_batch;
        float inv_sum = 1.0f / total_sum;
        
        for (int i = tid; i * 4 < dim; i += blockDim.x) {
            float4 val = output_batch4[i];
            val.x *= inv_sum;
            val.y *= inv_sum;
            val.z *= inv_sum;
            val.w *= inv_sum;
            output_batch4[i] = val;
        }
    } else {
        for (int i = tid; i < dim; i += blockDim.x) {
            output_batch[i] /= total_sum;
        }
    }
}

__global__ void process_price_vectors_kernel(
    const float* __restrict__ prices,
    const float* __restrict__ weights,
    float* __restrict__ features,
    int batch_size,
    int n_assets,
    int n_features
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;
    
    if (batch_idx >= batch_size) return;
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const float* price_vector = prices + batch_idx * n_assets;
    
    for (int feature_base = warp_id * 32; feature_base < n_features; feature_base += (blockDim.x / 32) * 32) {
        int feature_idx = feature_base + lane_id;
        float result = 0.0f;
        
        if (feature_idx < n_features) {
            if ((n_assets & 3) == 0) {
                const float4* price_vector4 = (const float4*)price_vector;
                
                #pragma unroll 4
                for (int i = 0; i < n_assets / 4; i++) {
                    float4 price_val = price_vector4[i];
                    int base_idx = i * 4;
                    result += price_val.x * weights[base_idx * n_features + feature_idx];
                    result += price_val.y * weights[(base_idx + 1) * n_features + feature_idx];
                    result += price_val.z * weights[(base_idx + 2) * n_features + feature_idx];
                    result += price_val.w * weights[(base_idx + 3) * n_features + feature_idx];
                }
            } else {
                #pragma unroll 4
                for (int i = 0; i < n_assets; i++) {
                    result += price_vector[i] * weights[i * n_features + feature_idx];
                }
            }
        }
        
        // Warp-aggregated atomic using shuffle reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_result = warp.shfl_down(result, offset);
            int other_feature = warp.shfl_down(feature_idx, offset);
            
            if (lane_id + offset < 32 && feature_base + lane_id + offset < n_features) {
                if (lane_id < offset) {
                    result += other_result;
                } else if (other_feature < n_features) {
                    atomicAdd(&features[batch_idx * n_features + other_feature], other_result);
                }
            }
        }
        
        if (lane_id == 0 && feature_idx < n_features) {
            atomicAdd(&features[batch_idx * n_features + feature_idx], result);
        }
    }
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
    int shared_mem = ((input_dim + 31) / 32) * 32 * sizeof(float); // 128-byte aligned
    
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
    int shared_mem = (2 * block.x + 2) * sizeof(float); // Bank conflict avoidance
    
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

}
