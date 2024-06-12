#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
/*
 * GPU Kernel: initialize_grid_xx0_gpu.
 *
 */
__global__ static void initialize_grid_xx0_gpu(REAL *restrict xx0) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const xxmin0 = d_params.xxmin0;

  REAL const dxx0 = d_params.dxx0;

  REAL const Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  constexpr REAL onehalf = 1.0 / 2.0;

  for (int j = index; j < Nxx_plus_2NGHOSTS0; j += stride)
    xx0[j] = xxmin0 + ((REAL)(j - NGHOSTS) + onehalf) * dxx0;
}
/*
 * GPU Kernel: initialize_grid_xx1_gpu.
 *
 */
__global__ static void initialize_grid_xx1_gpu(REAL *restrict xx1) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const xxmin1 = d_params.xxmin1;

  REAL const dxx1 = d_params.dxx1;

  REAL const Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  constexpr REAL onehalf = 1.0 / 2.0;

  for (int j = index; j < Nxx_plus_2NGHOSTS1; j += stride)
    xx1[j] = xxmin1 + ((REAL)(j - NGHOSTS) + onehalf) * dxx1;
}
/*
 * GPU Kernel: initialize_grid_xx2_gpu.
 *
 */
__global__ static void initialize_grid_xx2_gpu(REAL *restrict xx2) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  REAL const xxmin2 = d_params.xxmin2;

  REAL const dxx2 = d_params.dxx2;

  REAL const Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  constexpr REAL onehalf = 1.0 / 2.0;

  for (int j = index; j < Nxx_plus_2NGHOSTS2; j += stride)
    xx2[j] = xxmin2 + ((REAL)(j - NGHOSTS) + onehalf) * dxx2;
}

/*
 *
 * Initializes a cell-centered grid in SinhSymTP coordinates based on physical dimensions (grid_physical_size).
 *
 * Inputs:
 * - Nx[] inputs: Specifies new grid dimensions, if needed.
 * - grid_is_resized: Indicates if the grid has been manually resized, triggering adjustments to grid parameters.
 * - convergence_factor: Multiplier for grid dimensions to refine resolution, applied only if grid hasn't been resized.
 *
 * Parameter outputs:
 * - Nxx: Number of grid points in each direction.
 * - Nxx_plus_2NGHOSTS: Total grid points including ghost zones.
 * - dxx: Grid spacing.
 * - invdxx: Inverse of grid spacing.
 *
 * Grid setup output:
 * - xx: Coordinate values for each (cell-centered) grid point.
 *
 */
void numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3],
                                                      const int Nx[3], const bool grid_is_resized) {
  // Start by setting default values for Nxx.
  params->Nxx0 = 256;
  params->Nxx1 = 256;
  params->Nxx2 = 16;

  // If all components of Nx[] are set to reasonable values (i.e., not -1), then set params->Nxx{} to Nx[].
  if (!(Nx[0] == -1 || Nx[1] == -1 || Nx[2] == -1)) {
    params->Nxx0 = Nx[0];
    params->Nxx1 = Nx[1];
    params->Nxx2 = Nx[2];
  }
  snprintf(params->CoordSystemName, 50, "SinhSymTP");

  if (!grid_is_resized) {
    // convergence_factor does not increase resolution across an axis of symmetry:
    if (params->Nxx0 != 2)
      params->Nxx0 *= commondata->convergence_factor;
    if (params->Nxx1 != 2)
      params->Nxx1 *= commondata->convergence_factor;
    if (params->Nxx2 != 2)
      params->Nxx2 *= commondata->convergence_factor;
  }

  params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;
#include "../set_CodeParameters.h"

  // Set grid size to grid_physical_size (set above, based on params->grid_physical_size):
  {
    // Set grid size to a function of grid_physical_size, just set in set_CodeParameters.h above:
    params->AMAX = grid_physical_size;
  }
  if (!grid_is_resized) {
    // Set xxmin, xxmax
    params->xxmin0 = 0;
    params->xxmin1 = 0;
    params->xxmin2 = -M_PI;
    params->xxmax0 = 1;
    params->xxmax1 = M_PI;
    params->xxmax2 = M_PI;
  }

  params->dxx0 = (params->xxmax0 - params->xxmin0) / ((REAL)params->Nxx0);
  params->dxx1 = (params->xxmax1 - params->xxmin1) / ((REAL)params->Nxx1);
  params->dxx2 = (params->xxmax2 - params->xxmin2) / ((REAL)params->Nxx2);

  params->invdxx0 = ((REAL)params->Nxx0) / (params->xxmax0 - params->xxmin0);
  params->invdxx1 = ((REAL)params->Nxx1) / (params->xxmax1 - params->xxmin1);
  params->invdxx2 = ((REAL)params->Nxx2) / (params->xxmax2 - params->xxmin2);

  // Allocate device storage
  cudaMalloc(&xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2);
  cudaCheckErrors(malloc, "Malloc failed");

  cpyHosttoDevice_params__constant(params);

  dim3 block_threads, grid_blocks;
  auto set_grid_block = [&block_threads, &grid_blocks](auto Nx) {
    size_t threads_in_x_dir = 32;
    block_threads = dim3(threads_in_x_dir, 1, 1);
    grid_blocks = dim3((Nx + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  };

  size_t streamid = params->grid_idx % nstreams;
  set_grid_block(Nxx_plus_2NGHOSTS0);
  initialize_grid_xx0_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(xx[0]);
  cudaCheckErrors(initialize_grid_xx0_gpu, "kernel failed");

  streamid = (params->grid_idx + 1) % nstreams;
  set_grid_block(Nxx_plus_2NGHOSTS1);
  initialize_grid_xx1_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(xx[1]);
  cudaCheckErrors(initialize_grid_xx1_gpu, "kernel failed");

  streamid = (params->grid_idx + 2) % nstreams;
  set_grid_block(Nxx_plus_2NGHOSTS2);
  initialize_grid_xx2_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(xx[2]);
  cudaCheckErrors(initialize_grid_xx2_gpu, "kernel failed");
}
