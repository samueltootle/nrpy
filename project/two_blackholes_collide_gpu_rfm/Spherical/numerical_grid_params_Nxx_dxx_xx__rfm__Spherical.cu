#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
// #include "../BHaH_gpu_function_prototypes.h"
/*
 * Set up a cell-centered Spherical grid of size grid_physical_size. Set params: Nxx, Nxx_plus_2NGHOSTS, dxx, invdxx, and xx.
 */

__host__
void set_params(REAL convergence_factor, params_struct * params) {
  params->Nxx0 = 320;
  params->Nxx1 = 128;
  params->Nxx2 = 128;

  const REAL grid_physical_size = params->grid_physical_size;
  // Ignoring name for now since I don't see it used anywhere...
  // snprintf(params->CoordSystemName, 50, "Spherical");

  // convergence_factor does not increase resolution across an axis of symmetry:
  if (params->Nxx0 != 2)
    params->Nxx0 *= convergence_factor;
  if (params->Nxx1 != 2)
    params->Nxx1 *= convergence_factor;
  if (params->Nxx2 != 2)
    params->Nxx2 *= convergence_factor;
  
  params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;

  // Set grid size to grid_physical_size (set above, based on params->grid_physical_size):
  params->RMAX = grid_physical_size;

  // Set xxmin, xxmax
  params->xxmin0 = 0;
  params->xxmin1 = 0;
  params->xxmin2 = -M_PI;
  params->xxmax0 = params->RMAX;
  params->xxmax1 = M_PI;
  params->xxmax2 = M_PI;

  params->dxx0 = (params->xxmax0 - params->xxmin0) / ((REAL)params->Nxx0);
  params->dxx1 = (params->xxmax1 - params->xxmin1) / ((REAL)params->Nxx1);
  params->dxx2 = (params->xxmax2 - params->xxmin2) / ((REAL)params->Nxx2);

  params->invdxx0 = ((REAL)params->Nxx0) / (params->xxmax0 - params->xxmin0);
  params->invdxx1 = ((REAL)params->Nxx1) / (params->xxmax1 - params->xxmin1);
  params->invdxx2 = ((REAL)params->Nxx2) / (params->xxmax2 - params->xxmin2);
}

__global__
void initialize_grid_xx0_gpu(REAL *restrict xx0) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const& xxmin0 = d_params.xxmin0;

  REAL const& dxx0 = d_params.dxx0;

  REAL const& Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  constexpr REAL onehalf = 1./2.;

  for (int j = index; j < Nxx_plus_2NGHOSTS0; j+=stride)
    xx0[j] = xxmin0 + ((REAL)(j - NGHOSTS) + onehalf) * dxx0;

}
__global__
void initialize_grid_xx1_gpu(REAL *restrict xx1) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const& xxmin1 = d_params.xxmin1;

  REAL const& dxx1 = d_params.dxx1;

  REAL const& Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  constexpr REAL onehalf = 1./2.;

  for (int j = index; j < Nxx_plus_2NGHOSTS1; j+=stride)
    xx1[j] = xxmin1 + ((REAL)(j - NGHOSTS) + onehalf) * dxx1;

}
__global__
void initialize_grid_xx2_gpu(REAL *restrict xx2) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const& xxmin2 = d_params.xxmin2;

  REAL const& dxx2 = d_params.dxx2;

  REAL const& Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  constexpr REAL onehalf = 1./2.;

  for (int j = index; j < Nxx_plus_2NGHOSTS2; j+=stride)
    xx2[j] = xxmin2 + ((REAL)(j - NGHOSTS) + onehalf) * dxx2;

}

void numerical_grid_params_Nxx_dxx_xx__rfm__Spherical(commondata_struct *restrict commondata, 
                                                      params_struct *restrict params, 
                                                      REAL * xx[3]) {
  set_params(commondata->convergence_factor, params);
  
  // Copy params to __constant__
  set_param_constants(params);
  
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Set up cell-centered Cartesian coordinate grid, centered at the origin.
  cudaMalloc(&xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2);
  cudaCheckErrors(malloc, "Malloc failed")

  dim3 block_threads, grid_blocks;
  auto set_grid_block = [&block_threads, &grid_blocks](auto Nx) {
    size_t threads_in_x_dir = MIN(Nx, 1024);
    block_threads = dim3(threads_in_x_dir, 1, 1);
    grid_blocks = dim3((Nx + threads_in_x_dir - 1)/threads_in_x_dir, 1, 1);
  };
  
  set_grid_block(Nxx_plus_2NGHOSTS0);
  initialize_grid_xx0_gpu<<<grid_blocks, block_threads, 0, streams[0]>>>(xx[0]);
  cudaCheckErrors(initialize_grid_xx0_gpu, "kernel failed");

  set_grid_block(Nxx_plus_2NGHOSTS1);
  initialize_grid_xx1_gpu<<<grid_blocks, block_threads, 0, streams[1]>>>(xx[1]);
  cudaCheckErrors(initialize_grid_xx1_gpu, "kernel failed");

  set_grid_block(Nxx_plus_2NGHOSTS2);
  initialize_grid_xx2_gpu<<<grid_blocks, block_threads, 0, streams[2]>>>(xx[2]);
  cudaCheckErrors(initialize_grid_xx2_gpu, "kernel failed");
}
