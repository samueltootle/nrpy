#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
// #include "../BHaH_gpu_function_prototypes.h"
/*
 * Set up a cell-centered Spherical grid of size grid_physical_size. Set params: Nxx, Nxx_plus_2NGHOSTS, dxx, invdxx, and xx.
 */

__global__
void set_params_gpu(REAL convergence_factor, params_struct * params) {
  params->Nxx0 = 72;
  params->Nxx1 = 12;
  params->Nxx2 = 2;

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
void initialize_grid_gpu(params_struct * params, 
                         REAL *restrict xx0,
                         REAL *restrict xx1,
                         REAL *restrict xx2) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const REAL xxmin0 = params->xxmin0;
  const REAL xxmin1 = params->xxmin1;
  const REAL xxmin2 = params->xxmin2;

  const REAL dxx0 = params->dxx0;
  const REAL dxx1 = params->dxx1;
  const REAL dxx2 = params->dxx2;

  const REAL Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const REAL Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const REAL Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  constexpr REAL onehalf = 1./2.;

  for (int j = index; j < Nxx_plus_2NGHOSTS0; j+=stride)
    xx0[j] = xxmin0 + ((REAL)(j - NGHOSTS) + onehalf) * dxx0;
  for (int j = index; j < Nxx_plus_2NGHOSTS1; j+=stride)
    xx1[j] = xxmin1 + ((REAL)(j - NGHOSTS) + onehalf) * dxx1;
  for (int j = index; j < Nxx_plus_2NGHOSTS2; j+=stride)
    xx2[j] = xxmin2 + ((REAL)(j - NGHOSTS) + onehalf) * dxx2;

}

void numerical_grid_params_Nxx_dxx_xx__rfm__Spherical(commondata_struct *restrict commondata, 
                                                      params_struct *restrict params, 
                                                      REAL * xx[3]) {
  set_params_gpu<<<1,1>>>(commondata->convergence_factor, params);
  cudaCheckErrors(set_params_gpu, "kernel failed")
  
  int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
  cudaMemcpy(&Nxx_plus_2NGHOSTS0, &params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS1, &params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS2, &params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")

  // Set up cell-centered Cartesian coordinate grid, centered at the origin.
  cudaMalloc(&xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2);
  cudaCheckErrors(malloc, "Malloc failed")

  int const N = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  uint threads = (N > 1024) ? 1024 : N;
  dim3 block_threads(threads,1,1);
  dim3 grid_blocks((N + 1024 - 1)/1024);
  initialize_grid_gpu<<<grid_blocks, block_threads>>>(params, xx[0], xx[1], xx[2]);
  cudaCheckErrors(initialize_grid_gpu, "kernel failed");
}
