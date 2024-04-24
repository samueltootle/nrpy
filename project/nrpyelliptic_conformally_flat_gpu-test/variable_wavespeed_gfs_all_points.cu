#include "BHaH_defines.h"
/*
 * GPU Kernel: variable_wavespeed_gfs_all_points_gpu.
 * GPU Kernel to initialize auxillary grid functions at all grid points.
 */
__global__ static void variable_wavespeed_gfs_all_points_gpu(const REAL *restrict x0, const REAL *restrict x1, const REAL *restrict x2,
                                                             REAL *restrict in_gfs, const REAL dt, const REAL MINIMUM_GLOBAL_WAVESPEED) {
  // Temporary parameters
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const REAL bScale = d_params.bScale;
  const REAL dxx0 = d_params.dxx0;
  const REAL dxx1 = d_params.dxx1;
  const REAL dxx2 = d_params.dxx2;

  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  [[maybe_unused]] const int invdxx0 = d_params.invdxx0;
  [[maybe_unused]] const int invdxx1 = d_params.invdxx1;
  [[maybe_unused]] const int invdxx2 = d_params.invdxx2;

  const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
    [[maybe_unused]] const REAL xx2 = x2[i2];
    for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
      [[maybe_unused]] const REAL xx1 = x1[i1];
      for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
        [[maybe_unused]] const REAL xx0 = x0[i0];
        /*
         *  Original SymPy expressions:
         *  "[const REAL dsmin0 = AMAX*dxx0*(exp(xx0/SINHWAA)/SINHWAA + exp(-xx0/SINHWAA)/SINHWAA)*sqrt(AMAX**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2*sin(xx1)**2)/(sqrt(AMAX**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2)*(exp(1/SINHWAA) - exp(-1/SINHWAA)))]"
         *  "[const REAL dsmin1 = dxx1*sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * bScale**2*sin(xx1)**2)]"
         *  "[const REAL dsmin2 = AMAX*dxx2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))*sin(xx1)/(exp(1/SINHWAA) - exp(-1/SINHWAA))]"
         */
        const REAL tmp1 = sin(xx1);
        const REAL tmp2 = (1.0 / (SINHWAA));
        const REAL tmp3 = exp(tmp2) - exp(-tmp2);
        const REAL tmp5 = exp(tmp2 * xx0);
        const REAL tmp6 = exp(-tmp2 * xx0);
        const REAL tmp10 = AMAX / tmp3;
        const REAL tmp7 = tmp5 - tmp6;
        const REAL tmp8 = ((AMAX) * (AMAX)) * ((tmp7) * (tmp7)) / ((tmp3) * (tmp3));
        const REAL tmp9 = sqrt(((bScale) * (bScale)) * ((tmp1) * (tmp1)) + tmp8);
        const REAL dsmin0 = dxx0 * tmp10 * tmp9 * (tmp2 * tmp5 + tmp2 * tmp6) / sqrt(((bScale) * (bScale)) + tmp8);
        const REAL dsmin1 = dxx1 * tmp9;
        const REAL dsmin2 = dxx2 * tmp1 * tmp10 * tmp7;

        // Set local wavespeed
        in_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)] = MINIMUM_GLOBAL_WAVESPEED * MIN(dsmin0, MIN(dsmin1, dsmin2)) / dt;

      } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
    } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
  } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
}

/*
 * Compute variable wavespeed for all grids based on local grid spacing.
 */
void variable_wavespeed_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
    REAL *restrict x0 = griddata[grid].xx[0];
    REAL *restrict x1 = griddata[grid].xx[1];
    REAL *restrict x2 = griddata[grid].xx[2];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.auxevol_gfs;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = NGHOSTS;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = params->grid_idx % nstreams;
    variable_wavespeed_gfs_all_points_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(x0, x1, x2, in_gfs, dt,
                                                                                                         MINIMUM_GLOBAL_WAVESPEED);
    cudaCheckErrors(cudaKernel, "variable_wavespeed_gfs_all_points_gpu failure");
  } // END LOOP for(int grid=0; grid<commondata->NUMGRIDS; grid++)
}
