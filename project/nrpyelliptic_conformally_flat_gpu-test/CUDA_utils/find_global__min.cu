#include "../BHaH_defines.h"

__global__ static void find_min__cuda(REAL *data, unsigned long long int *min, uint const data_length) {
  // shared data between all warps
  // Assumes one block = 32 warps = 32 * 32 threads
  // As of today, the standard maximum threads per
  // block is 1024 = 32 * 32
  __shared__ REAL shared_data[32];

  // largest value for uint
  REAL REDUCTION_LIMIT = (REAL)0xFFFFFFFFU;

  // Global data index - expecting a 1D dataset
  uint idx = threadIdx.x + blockDim.x * blockIdx.x;

  // thread index
  uint tid = threadIdx.x;

  // local thread minimum - set to something large
  REAL local_min = REDUCTION_LIMIT;

  // warp mask - says all threads are involved in shuffle
  // 0xFFFFFFFFU in binary is 32 1's.
  unsigned mask = 0xFFFFFFFFU;

  // lane = which thread am I in the warp
  uint lane = threadIdx.x % warpSize;
  // warpID = which warp am I in the block
  uint warpID = threadIdx.x / warpSize;

  // Stride through data for each thread
  while (idx < data_length) {
    if (local_min > data[idx])
      local_min = data[idx];
    // idx stride
    idx += gridDim.x * blockDim.x;
  }

  // Shuffle down kernel
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    REAL shfl = __shfl_down_sync(mask, local_min, offset);
    if (local_min > shfl)
      local_min = shfl;
  }
  // Shuffle results in lane 0 have the shuffle result
  if (lane == 0)
    shared_data[warpID] = local_min;

  // Make sure all warps in the block are synchronized
  __syncthreads();
  unsigned long long int *address_as_ull;
  // Since there is only 32 partial reductions, we only
  // have one warp worth of work
  if (warpID == 0) {
    // Check to make sure we had 32 blocks of data
    if (tid < blockDim.x / warpSize) {
      local_min = shared_data[lane];
    } else {
      local_min = REDUCTION_LIMIT;
    }

    // Shuffle down kernel
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      REAL shfl = __shfl_down_sync(mask, local_min, offset);
      if (local_min > shfl)
        local_min = shfl;
    }
    address_as_ull = (unsigned long long int *)&local_min;
    if (tid == 0) {
      atomicMin((unsigned long long int *)min, (unsigned long long int)*address_as_ull);
    }
  }
}

/*
 * Find array global minimum.
 */
__host__ REAL find_global__min(REAL *data, uint const data_length) {

  // This can be tested up to 1024
  uint threadCount = 32;

  // Number of blocks
  uint blockCount = (data_length + threadCount - 1) / threadCount;

  // CUDA atomics other than cas are only
  // compatible with (u)int.  To be generic
  // we use unsigned long long to be able to handle
  // 64 bit floats
  using ull = unsigned long long int;
  ull *h_min = (ull *)malloc(sizeof(ull));
  ull *d_min;
  *h_min = (unsigned long long int)0xFFFFFFFFU;

  cudaMalloc(&d_min, sizeof(ull));
  cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

  cudaMemcpy(d_min, h_min, sizeof(ull), cudaMemcpyHostToDevice);
  cudaCheckErrors(cudaMemcpy, "cudaCopyTo failure"); // error checking

  find_min__cuda<<<blockCount, threadCount>>>(data, d_min, data_length);
  cudaCheckErrors(find_min_cu, "cudaKernel - find_min_cu failed"); // error checking

  cudaMemcpy(h_min, d_min, sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "cudaCopyFrom failure"); // error checking

  cudaFree(d_min);
  cudaCheckErrors(cudaFree, "cudaFree failure"); // error checking

  // Recast back to result pointer type
  REAL *res = (REAL *)h_min;
  return *res;
}
