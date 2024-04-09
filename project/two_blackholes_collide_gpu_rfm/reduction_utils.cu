// #include "reduction_utils.h"
#include "./BHaH_defines.h"
#include "./BHaH_gpu_defines.h"
#include "./BHaH_gpu_function_prototypes.h"
#include <stdio.h>
#include <unistd.h>

template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length) {
    // shared data between all warps
    // Assumes one block = 32 warps = 32 * 32 threads
    // As of today, the standard maximum threads per
    // block is 1024 = 32 * 32
    __shared__ T shared_data[32];

    // largest value for uint
    T REDUCTION_LIMIT = (T) 0xFFFFFFFFU;

    // Global data index - expecting a 1D dataset
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;

    // thread index
    uint tid = threadIdx.x;

    // local thread minimum - set to something large
    T local_min = REDUCTION_LIMIT;
    // printf("init: %f\n", *address_as_ull);

    // warp mask - says all threads are involved in shuffle
    // 0xFFFFFFFFU in binary is 32 1's.
    unsigned mask = 0xFFFFFFFFU;

    // lane = which thread am I in the warp
    uint lane = threadIdx.x % warpSize;
    // warpID = which warp am I in the block
    uint warpID = threadIdx.x / warpSize;

    // Stride through data for each thread
    while(idx < data_length) {
        if(local_min > data[idx]) 
            local_min = data[idx];
        // idx stride
        idx += gridDim.x * blockDim.x;
    }

    // Shuffle down kernel
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
        T shfl = __shfl_down_sync(mask, local_min, offset);
        if(local_min > shfl) 
            local_min = shfl;
    }
    // Shuffle results in lane 0 have the shuffle result
    if(lane == 0) shared_data[warpID] = local_min;
    
    // Make sure all warps in the block are syncronized
    __syncthreads();
    unsigned long long int* address_as_ull;
    // Since there is only 32 partial reductions, we only
    // have one warp worth of work
    if(warpID == 0) {
        // Check to make sure we had 32 blocks of data
        if(tid < blockDim.x / warpSize) {
            local_min = shared_data[lane];
        } else {
            local_min = REDUCTION_LIMIT;
        }
        
        // Shuffle down kernel
        for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
            T shfl = __shfl_down_sync(mask, local_min, offset);
            if(local_min > shfl) 
                local_min = shfl;
        }
        address_as_ull = (unsigned long long int*)&local_min;
        if(tid == 0) {
            atomicMin((unsigned long long int *)min, (unsigned long long int)*address_as_ull);
        }
    }
}

__host__
REAL find_min(REAL * data, uint const data_length) {
    // This can be tested up to 1024
    uint threadCount = 32;
    
    // print_data<<<1,1>>>(data, data_length);
    // Number of blocks of 1024U threads
    uint blockCount = MAX(68, (data_length + threadCount - 1) / SHARED_SIZE_LIMIT);
    if(blockCount < 1) 
        blockCount = 1;
    
    // CUDA atomics other than cas are only
    // compatible with (u)int.  To be generic
    // we use unsigned long long to be able to handle
    // 64 bit floats
    using ull = unsigned long long int;
    ull * h_min = (ull*)malloc(sizeof(ull));
    ull * d_min;
    *h_min = (unsigned long long int)0xFFFFFFFFFFFFFFFFU;
    
    cudaMalloc(&d_min, sizeof(ull));
    cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

    cudaMemcpy(d_min, h_min, sizeof(ull), cudaMemcpyHostToDevice);
    cudaCheckErrors(cudaMemcpy, "cudaCopyTo failure"); // error checking
    
    find_min_cu<<<blockCount, threadCount>>>(data, d_min, data_length);
    cudaCheckErrors(find_min_cu, "cudaKernel - find_min_cu failed"); // error checking
    
    cudaMemcpy(h_min, d_min, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "cudaCopyFrom failure"); // error checking

    cudaFree(d_min);
    cudaCheckErrors(cudaFree, "cudaFree failure"); // error checking

    // Recast back to result pointer type
    REAL * res = (REAL *) h_min;
    return *res;
}

template<class T>
__global__
void reduction_sum_gpu(T * data, T * sum, uint const data_length) {
    // shared data between all warps
    // Assumes one block = 32 warps = 32 * 32 threads
    // As of today, the standard maximum threads per
    // block is 1024 = 32 * 32
    __shared__ T shared_data[32];

    // Global data index - expecting a 1D dataset
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;

    // thread index
    uint tid = threadIdx.x;

    // local thread minimum - set to something large
    T local_sum = (T)0.0;

    // warp mask - says all threads are involved in shuffle
    // 0xFFFFFFFFU in binary is 32 1's.
    unsigned mask = 0xFFFFFFFFU;

    // lane = which thread am I in the warp
    uint lane = threadIdx.x % warpSize;
    // warpID = which warp am I in the block
    uint warpID = threadIdx.x / warpSize;

    // Stride through data for each thread
    while(idx < data_length) {
        local_sum += data[idx];
        // idx stride
        idx += gridDim.x * blockDim.x;
    }

    // Shuffle down kernel
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
        T shfl = __shfl_down_sync(mask, local_sum, offset);
        local_sum += shfl;
    }
    // Shuffle results in lane 0 have the shuffle result
    if(lane == 0) {
        shared_data[warpID] = local_sum;
    }
    
    // Make sure all warps in the block are syncronized
    __syncthreads();
    // Since there is only 32 partial reductions, we only
    // have one warp worth of work
    if(warpID == 0) {
        // Check to make sure we had 32 blocks of data
        if(tid < blockDim.x / warpSize) {
            local_sum = shared_data[lane];
        } else {
            local_sum = (T) 0.0;
        }
        
        // Shuffle down kernel
        for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
            T shfl = __shfl_down_sync(mask, local_sum, offset);
            local_sum += shfl;
        }
        if(tid == 0) {
            atomicAdd(sum, local_sum);
        }
    }
}

__host__
REAL reduction_sum(REAL * data, uint const data_length) {
    // This can be tested up to 1024
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    
    // Number of blocks of 1024U threads
    uint blockCount = (data_length + threadCount - 1) / SHARED_SIZE_LIMIT;
    if(blockCount < 1) 
        blockCount = 1;
    
    // CUDA atomics other than cas are only
    // compatible with (u)int.  To be generic
    // we use unsigned long long to be able to handle
    // 64 bit floats
    using ull = REAL;
    ull * h_sum = (ull*)malloc(sizeof(ull));
    ull * d_sum;
    *h_sum = 0.;
    
    cudaMalloc(&d_sum, sizeof(ull));
    cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

    cudaMemcpy(d_sum, h_sum, sizeof(ull), cudaMemcpyHostToDevice);
    cudaCheckErrors(cudaMemcpy, "cudaCopyTo failure"); // error checking
    
    reduction_sum_gpu<REAL><<<blockCount, threadCount>>>(data, d_sum, data_length);
    cudaCheckErrors(reduction_sum_gpu, "cudaKernel - find_min_cu failed"); // error checking
    
    cudaMemcpy(h_sum, d_sum, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "cudaCopyFrom failure"); // error checking

    cudaFree(d_sum);
    cudaCheckErrors(cudaFree, "cudaFree failure"); // error checking

    // Recast back to result pointer type
    REAL * res = (REAL *) h_sum;
    return *res;
}

__host__
uint reduction_sum(uint * data, uint const data_length) {
    // Number of blocks of 1024U threads
    uint blockCount = data_length / SHARED_SIZE_LIMIT;
    if(blockCount < 1) 
        blockCount = 1;
    
    // This can be tested up to 1024
    uint threadCount = SHARED_SIZE_LIMIT / 2;
    
    // CUDA atomics other than cas are only
    // compatible with (u)int.  To be generic
    // we use unsigned long long to be able to handle
    // 64 bit floats
    using ull = uint;
    ull * h_sum = (ull*)malloc(sizeof(ull));
    ull * d_sum;
    *h_sum = 0;
    
    cudaMalloc(&d_sum, sizeof(ull));
    cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

    cudaMemcpy(d_sum, h_sum, sizeof(ull), cudaMemcpyHostToDevice);
    cudaCheckErrors(cudaMemcpy, "cudaCopyTo failure"); // error checking
    
    reduction_sum_gpu<<<blockCount, threadCount>>>(data, d_sum, data_length);
    cudaCheckErrors(reduction_sum_gpu, "cudaKernel - find_min_cu failed"); // error checking
    
    cudaMemcpy(h_sum, d_sum, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "cudaCopyFrom failure"); // error checking

    cudaFree(d_sum);
    cudaCheckErrors(cudaFree, "cudaFree failure"); // error checking

    // Recast back to result pointer type
    uint * res = (uint *) h_sum;
    return *res;
}

__global__
void print_params() {
  printf("dxx0: %f - dxx1: %f - dxx2: %f\n", d_params.dxx0, d_params.dxx1, d_params.dxx2);
  printf("Nxx0+G: %d - Nxx1+G: %d - Nxx2+G: %d\n", d_params.Nxx_plus_2NGHOSTS0, d_params.Nxx_plus_2NGHOSTS1, d_params.Nxx_plus_2NGHOSTS2);
}