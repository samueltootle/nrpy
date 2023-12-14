#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
/*
 * rfm_precompute_defines: reference metric precomputed lookup arrays: defines
 */

__global__
void rfm_precompute_defines__rfm__Spherical_xx0_gpu(REAL *restrict f0_of_xx0, const REAL *restrict xx0) {
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;
  for (int i0 = tid0; i0 < d_params.Nxx_plus_2NGHOSTS0; i0+=stride0) {
    const REAL xx = xx0[i0];
    f0_of_xx0[i0] = xx;
  }
}

__global__
void rfm_precompute_defines__rfm__Spherical_xx1_gpu(REAL *restrict f1_of_xx1, const REAL *restrict xx1) {
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;
  for (int i1 = tid0; i1 < d_params.Nxx_plus_2NGHOSTS1; i1+=stride0) {
    const REAL xx = xx1[i1];
    f1_of_xx1[i1] = sin(xx);
  }
}

__global__
void rfm_precompute_defines__rfm__Spherical_xx1__D1_gpu(REAL *restrict f1_of_xx1__D1, const REAL *restrict xx1) {
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;
  for (int i1 = tid0; i1 < d_params.Nxx_plus_2NGHOSTS1; i1+=stride0) {
    const REAL xx = xx1[i1];
    f1_of_xx1__D1[i1] = cos(xx);
  }
}

__global__
void rfm_precompute_defines__rfm__Spherical_xx1__DD11_gpu(REAL *restrict f1_of_xx1__DD11, REAL *restrict f1_of_xx1) {
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;
  for (int i1 = tid0; i1 < d_params.Nxx_plus_2NGHOSTS1; i1+=stride0) {
    f1_of_xx1__DD11[i1] = -f1_of_xx1[i1];
  }
}

void rfm_precompute_defines__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                            rfm_struct *restrict rfmstruct, REAL * xx[3]) {

  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  
  dim3 block_threads, grid_blocks;
  auto set_grid_block = [&block_threads, &grid_blocks](auto Nx) {
    size_t tx = MIN(Nx, 1024);
    block_threads = dim3(tx, 1, 1);
    grid_blocks = dim3((Nx + tx - 1)/tx, 1, 1);
  };
  set_grid_block(Nxx_plus_2NGHOSTS0);
  rfm_precompute_defines__rfm__Spherical_xx0_gpu<<<grid_blocks, block_threads, 0, stream1>>>(rfmstruct->f0_of_xx0, xx[0]);
  cudaCheckErrors(rfm_precompute_defines__rfm__Spherical_xx0_gpu, "kernel failed");

  set_grid_block(Nxx_plus_2NGHOSTS1);
  rfm_precompute_defines__rfm__Spherical_xx1_gpu<<<grid_blocks, block_threads, 0, stream2>>>(rfmstruct->f1_of_xx1, xx[1]);
  cudaCheckErrors(rfm_precompute_defines__rfm__Spherical_xx1_gpu, "kernel failed");
  rfm_precompute_defines__rfm__Spherical_xx1__D1_gpu<<<grid_blocks, block_threads, 0, stream3>>>(rfmstruct->f1_of_xx1__D1, xx[1]);
  cudaCheckErrors(rfm_precompute_defines__rfm__Spherical_xx1__D1_gpu, "kernel failed");
  rfm_precompute_defines__rfm__Spherical_xx1__DD11_gpu<<<grid_blocks, block_threads, 0, stream2>>>(rfmstruct->f1_of_xx1__DD11, rfmstruct->f1_of_xx1);
  cudaCheckErrors(rfm_precompute_defines__rfm__Spherical_xx1__DD11_gpu, "kernel failed");
}
