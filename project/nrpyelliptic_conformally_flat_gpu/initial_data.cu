#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * GPU Kernel: initial_guess_all_points_gpu.
 * GPU Kernel to initialize all grid points.
 */
__global__ static void initial_guess_all_points_gpu(const REAL *restrict x0, const REAL *restrict x1, const REAL *restrict x2,
                                                    REAL *restrict in_gfs) {

  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
  [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
  [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

  const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
    [[maybe_unused]] const REAL xx2 = x2[i2];
    for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
      [[maybe_unused]] const REAL xx1 = x1[i1];
      for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
        [[maybe_unused]] const REAL xx0 = x0[i0];
        initial_guess_single_point(xx0, xx1, xx2, &in_gfs[IDX4(UUGF, i0, i1, i2)], &in_gfs[IDX4(VVGF, i0, i1, i2)]);
      } // END LOOP: for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0)
    } // END LOOP: for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1)
  } // END LOOP: for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2)
}

/*
 * Set initial guess to solutions of hyperbolic relaxation equation at all points.
 */
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
    REAL *restrict x0 = griddata[grid].xx[0];
    REAL *restrict x1 = griddata[grid].xx[1];
    REAL *restrict x2 = griddata[grid].xx[2];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
#include "set_CodeParameters.h"

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = NGHOSTS;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = params->grid_idx % nstreams;
    initial_guess_all_points_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(x0, x1, x2, in_gfs);
    cudaCheckErrors(cudaKernel, "initial_guess_all_points_gpu failure");
  }
}
