/*
 * Cuda kernels for fused adamw and adamw4bit
 */

#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/Exceptions.h>

using torch::Tensor;

__device__ __forceinline__ float atomicMaxNonneg (float * addr, float value) {
    float old;
    // assert value >= 0
    old = __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    return old;
}


__device__ __forceinline__ int q_mapping(const float* __restrict__ qmap,
                                         int bits,
                                         float x) {
    int lo = 0;
    int hi = 1 << bits;
    
    if (x <= qmap[lo])
      return lo;
    if (qmap[hi - 1] <= x)
      return (hi - 1);

    while (lo < hi){
      int mi = (lo + hi) >> 1;
      if (qmap[mi] <= x) lo = mi + 1;
      else hi = mi;
    }
    // return lo - 1;

    int rank = 0;
    float mid_val = (qmap[lo - 1] + qmap[lo]) * 0.5f;
    rank = (mid_val < x) ? lo : lo - 1;
    return rank;
}


template <typename T>
__global__ void adamw_cuda_kernel(
        T* __restrict__ p,
        T* __restrict__ exp_avg,
        T* __restrict__ exp_avg_sq,
        const T * __restrict__ g,
        const float beta1,
        const float beta2,
        const float lr,
        const float weight_decay,
        const float eps,
        const float step,
        const size_t total_size)
{
        const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_id >= total_size) return;

        exp_avg[global_id] = beta1 * exp_avg[global_id] + (1 - beta1) * g[global_id];
        exp_avg_sq[global_id] = beta2 * exp_avg_sq[global_id] + (1 - beta2) * g[global_id] * g[global_id];

        const float correction1 = 1.0f - powf(beta1, step);
        const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));

        float denom = (sqrtf(exp_avg_sq[global_id]) / correction2_sqrt + eps) * correction1;
        float update = (exp_avg[global_id]/denom) + (weight_decay * p[global_id]);
        p[global_id] = p[global_id] - (lr * update);
}


template <typename T>
__global__ void adamw4bit_cuda_kernel(
        T* __restrict__ p,
        const T * __restrict__ g,
        int8_t* __restrict__ exp_avg,
        int8_t* __restrict__ exp_avg_sq,
        T* __restrict__ exp_avg_scale,
        T* __restrict__ exp_avg_sq_scale,
        const float* __restrict__ exp_avg_qmap,
        const float* __restrict__ exp_avg_sq_qmap,
        const float beta1,
        const float beta2,
        const float lr,
        const float weight_decay,
        const float eps,
        const float step,
        const size_t total_size)
{
        const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        const int scale_id = blockIdx.x;
        const int working_id0 = global_id << 1;
        const int working_id1 = (global_id << 1) + 1;
        const float correction1 = 1.0f - powf(beta1, step);
        const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));

        __shared__ float absmax_exp_avg;
        __shared__ float absmax_exp_avg_sq;
        if (threadIdx.x == 0) {
            absmax_exp_avg = 0.0f;
            absmax_exp_avg_sq = 0.0f;
        }
        __syncthreads();

        if (working_id0 >= total_size) return;

        const int8_t mask = (1 << 4) - 1;
        // dequantize optimizer state, and run optimizer
        // Note that we require the 'rank' of right 4-bits is smaller than that of left 4-bits in one byte
        const uint8_t exp_avg_idx0 = (exp_avg[global_id]) & mask;
        T exp_avg0 = (T)exp_avg_qmap[exp_avg_idx0] * exp_avg_scale[scale_id];
        exp_avg0 = beta1 * exp_avg0 + (1 - beta1) * g[working_id0];
        const uint8_t exp_avg_sq_idx0 = (exp_avg_sq[global_id]) & mask;
        T exp_avg_sq0 = (T)exp_avg_sq_qmap[exp_avg_sq_idx0] * exp_avg_sq_scale[scale_id];
        exp_avg_sq0 = beta2 * exp_avg_sq0 + (1 - beta2) * g[working_id0] * g[working_id0];

        float denom0 = (sqrtf(exp_avg_sq0) / correction2_sqrt + eps) * correction1;
        float update0 = (exp_avg0/denom0) + (weight_decay * p[working_id0]);
        p[working_id0] = p[working_id0] - (lr * update0);

        T exp_avg1 = 0;
        T exp_avg_sq1 = 0;
        if (working_id1 < total_size) {
            const uint8_t exp_avg_idx1 = (exp_avg[global_id] >> 4) & mask;
            exp_avg1 = (T)exp_avg_qmap[exp_avg_idx1] * exp_avg_scale[scale_id];
            exp_avg1 = beta1 * exp_avg1 + (1 - beta1) * g[working_id1];
            const uint8_t exp_avg_sq_idx1 = (exp_avg_sq[global_id] >> 4) & mask;
            exp_avg_sq1 = (T)exp_avg_sq_qmap[exp_avg_sq_idx1] * exp_avg_sq_scale[scale_id];
            exp_avg_sq1 = beta2 * exp_avg_sq1 + (1 - beta2) * g[working_id1] * g[working_id1];

            float denom1 = (sqrtf(exp_avg_sq1) / correction2_sqrt + eps) * correction1;
            float update1 = (exp_avg1/denom1) + (weight_decay * p[working_id1]);
            p[working_id1] = p[working_id1] - (lr * update1);
        }

        // compute new scale for quantization
        float local_absmax_exp_avg = fmaxf(fabsf((float)exp_avg0), fabsf((float)exp_avg1));
        float local_absmax_exp_avg_sq = fmaxf((float)exp_avg_sq0, (float)exp_avg_sq1);
        atomicMaxNonneg(&absmax_exp_avg, local_absmax_exp_avg);
        atomicMaxNonneg(&absmax_exp_avg_sq, local_absmax_exp_avg_sq);
        __syncthreads();

        // quantize optimizer state and write new scales
        int8_t local_packed_exp_avg = 0;
        int8_t local_packed_exp_avg_sq = 0;
        const int8_t q_exp_avg0 = (int8_t)q_mapping(exp_avg_qmap, 4, (float)exp_avg0 / absmax_exp_avg);
        const int8_t q_exp_avg_sq0 = (int8_t)q_mapping(exp_avg_sq_qmap, 4, (float)exp_avg_sq0 / absmax_exp_avg_sq);
        local_packed_exp_avg |= (q_exp_avg0 & mask);
        local_packed_exp_avg_sq |= (q_exp_avg_sq0 & mask);

        if (working_id1 < total_size) {
            const int8_t q_exp_avg1 = (int8_t)q_mapping(exp_avg_qmap, 4, (float)exp_avg1 / absmax_exp_avg);
            const int8_t q_exp_avg_sq1 = (int8_t)q_mapping(exp_avg_sq_qmap, 4, (float)exp_avg_sq1 / absmax_exp_avg_sq);
            local_packed_exp_avg |= ((q_exp_avg1 & mask) << 4);
            local_packed_exp_avg_sq |= ((q_exp_avg_sq1 & mask) << 4);
        }

        exp_avg[global_id] = local_packed_exp_avg;
        exp_avg_sq[global_id] = local_packed_exp_avg_sq;
        if (threadIdx.x == 0) {
            exp_avg_scale[scale_id] = (T)absmax_exp_avg;
            exp_avg_sq_scale[scale_id] = (T)absmax_exp_avg_sq;
        }
        __syncthreads();
}


void fused_adamw_cuda(Tensor& p, Tensor& g, Tensor& exp_avg, Tensor& exp_avg_sq, 
                      float beta1, float beta2, float lr, float weight_decay, float eps, float step) {
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");
    
    const int block_dim = 128;
    int grid_dim = ((total_size + block_dim - 1) / block_dim);
    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "fused_adam_cuda", ([&] {
        adamw_cuda_kernel<scalar_t><<<blocks, block_dim>>>(
            p.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            exp_avg_sq.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
            step,
            total_size
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}


void fused_adamw4bit_cuda(Tensor& p, Tensor& g, Tensor& exp_avg, Tensor& exp_avg_sq,
                          Tensor& exp_avg_scale, Tensor& exp_avg_sq_scale, Tensor& exp_avg_qmap, Tensor& exp_avg_sq_qmap,
                          float beta1, float beta2, float lr, float weight_decay, float eps, float step) {
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");
    
    const int block_dim = 128;
    int grid_dim = ((total_size + block_dim - 1) / block_dim);
    TORCH_CHECK(grid_dim == exp_avg_scale.numel());
    TORCH_CHECK(grid_dim == exp_avg_sq_scale.numel());
    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "fused_adam4bit_cuda", ([&] {
        adamw4bit_cuda_kernel<scalar_t><<<blocks, block_dim/2>>>(
            p.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            exp_avg.data_ptr<int8_t>(),
            exp_avg_sq.data_ptr<int8_t>(),
            exp_avg_scale.data_ptr<scalar_t>(),
            exp_avg_sq_scale.data_ptr<scalar_t>(),
            exp_avg_qmap.data_ptr<float>(),
            exp_avg_sq_qmap.data_ptr<float>(),
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
            step,
            total_size
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}