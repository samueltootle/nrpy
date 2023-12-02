#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Set up cell-centered Cartesian grids.
 */
__global__
void set_params_gpu(REAL convergence_factor, 
                    REAL CFL_FACTOR,
                    params_struct * params, REAL *restrict min_dt) {
  const REAL xxmin0 = params->xxmin0;
  const REAL xxmin1 = params->xxmin1;
  const REAL xxmin2 = params->xxmin2;
  const REAL xxmax0 = params->xxmax0;
  const REAL xxmax1 = params->xxmax1;
  const REAL xxmax2 = params->xxmax2;

  params->Nxx0 *= convergence_factor;
  params->Nxx1 *= convergence_factor;
  params->Nxx2 *= convergence_factor;

  params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;

  params->dxx0 = (xxmax0 - xxmin0) / ((REAL)params->Nxx0);
  params->dxx1 = (xxmax1 - xxmin1) / ((REAL)params->Nxx1);
  params->dxx2 = (xxmax2 - xxmin2) / ((REAL)params->Nxx2);

  params->invdxx0 = ((REAL)params->Nxx0) / (xxmax0 - xxmin0);
  params->invdxx1 = ((REAL)params->Nxx1) / (xxmax1 - xxmin1);
  params->invdxx2 = ((REAL)params->Nxx2) / (xxmax2 - xxmin2);

  *min_dt = MIN(*min_dt, CFL_FACTOR * MIN(params->dxx0, MIN(params->dxx1, params->dxx2))); // CFL condition
}

__global__
void initialize_grid_gpu(params_struct * params, 
                         REAL *restrict xx0,
                         REAL *restrict xx1,
                         REAL *restrict xx2) {

  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  
    
  // params_struct *restrict params = &griddata[grid].params;
  const REAL xxmin0 = params->xxmin0;
  const REAL xxmin1 = params->xxmin1;
  const REAL xxmin2 = params->xxmin2;

  for (int j = index; j < params->Nxx_plus_2NGHOSTS0; j+=stride)
    xx0[j] = xxmin0 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx0;
  for (int j = index; j < params->Nxx_plus_2NGHOSTS1; j+=stride)
    xx1[j] = xxmin1 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx1;
  for (int j = index; j < params->Nxx_plus_2NGHOSTS2; j+=stride)
    xx2[j] = xxmin2 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx2;
}

__host__
void allocate_grid_storage(REAL *xx[3], int const Nxxtot0, int const Nxxtot1, int const Nxxtot2) {
  cudaMalloc(&xx[0], sizeof(REAL) * Nxxtot0);
  cudaCheckErrors(cudaMalloc, "memory failed")
  cudaMalloc(&xx[1], sizeof(REAL) * Nxxtot1);
  cudaCheckErrors(cudaMalloc, "memory failed")
  cudaMalloc(&xx[2], sizeof(REAL) * Nxxtot2);
  cudaCheckErrors(cudaMalloc, "memory failed")
}

__host__
void testcpy(REAL const * const xx, size_t idx) {
  REAL x;
  cudaMemcpy(&x, &xx[idx], sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMalloc, "memory failed")
  printf("var: %1.15f\n", x);
}

void numerical_grids_and_timestep(commondata_struct * commondata, griddata_struct *griddata, bool calling_for_first_time) {

  commondata->dt = 1e30;
  // Initialize timestepping parameters to zero if this is the first time this function is called.
  if (calling_for_first_time) {
    commondata->nn = 0;
    commondata->nn_0 = 0;
    commondata->t_0 = 0.0;
    commondata->time = 0.0;
  }
  REAL* d_min_dt;
  cudaMalloc(&d_min_dt, sizeof(REAL));
  cudaCheckErrors(cudaMalloc, "memory failed")
  cudaMemcpy(d_min_dt, &commondata->dt, sizeof(REAL), cudaMemcpyHostToDevice);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    params_struct restrict* params = griddata[grid].params;
    const REAL convergence_factor = commondata->convergence_factor;
    const REAL CFL_FACTOR = commondata->CFL_FACTOR;
    set_params_gpu<<<1,1>>>(convergence_factor, CFL_FACTOR, params, d_min_dt);
    cudaCheckErrors(set_params_gpu, "kernel failed")
    cudaDeviceSynchronize();
    
    int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
    cudaMemcpy(&Nxx_plus_2NGHOSTS0, &griddata[grid].params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")
    cudaMemcpy(&Nxx_plus_2NGHOSTS1, &griddata[grid].params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")
    cudaMemcpy(&Nxx_plus_2NGHOSTS2, &griddata[grid].params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")
    
    allocate_grid_storage(griddata[grid].xx, Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2);

    int const N = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
    
    dim3 block_threads(1024,1,1);
    dim3 grid_blocks((N + 1024 - 1)/1024);
    initialize_grid_gpu<<<grid_blocks, block_threads>>>(params, 
                                                        griddata[grid].xx[0], 
                                                        griddata[grid].xx[1], 
                                                        griddata[grid].xx[2]);
    cudaCheckErrors(initialize_grid_gpu, "initialize failed");
  }
  cudaMemcpy(&commondata->dt, d_min_dt, sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
}
