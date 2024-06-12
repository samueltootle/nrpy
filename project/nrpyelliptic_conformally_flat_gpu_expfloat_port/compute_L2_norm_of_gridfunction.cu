#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * GPU Kernel: compute_L2_norm_of_gridfunction_gpu.
 * GPU Kernel to compute L2 quantities pointwise (not summed).
 */
__global__ static void compute_L2_norm_of_gridfunction_gpu(const REAL *restrict x0, const REAL *restrict x1, const REAL *restrict x2,
                                                           const REAL *restrict in_gfs, REAL *restrict aux_gfs, const REAL integration_radius,
                                                           const int gf_index) {
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

  [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
  [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
  [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

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
         *  "[const REAL r = sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2*sin(xx1)**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * (AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2)*cos(xx1)**2)]"
         *  "[const REAL sqrtdetgamma = AMAX**4*(exp(xx0/SINHWAA)/SINHWAA + exp(-xx0/SINHWAA)/SINHWAA)**2*(AMAX**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2*sin(xx1)**2)**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2*sin(xx1)**2/((AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * bScale**2)*(exp(1/SINHWAA) - exp(-1/SINHWAA))**4)]"
         */
        const REAL tmp0 = ((sin(xx1)) * (sin(xx1)));
        const REAL tmp1 = (1.0 / (SINHWAA));
        const REAL tmp2 = exp(tmp1) - exp(-tmp1);
        const REAL tmp4 = exp(tmp1 * xx0);
        const REAL tmp5 = exp(-tmp1 * xx0);
        const REAL tmp6 = ((tmp4 - tmp5) * (tmp4 - tmp5));
        const REAL tmp7 = ((AMAX) * (AMAX)) * tmp6 / ((tmp2) * (tmp2));
        const REAL tmp9 = ((bScale) * (bScale)) + tmp7;
        const REAL r = sqrt(tmp0 * tmp7 + tmp9 * ((cos(xx1)) * (cos(xx1))));
        const REAL sqrtdetgamma = ((AMAX) * (AMAX) * (AMAX) * (AMAX)) * tmp0 * tmp6 *
                                  ((((bScale) * (bScale)) * tmp0 + tmp7) * (((bScale) * (bScale)) * tmp0 + tmp7)) *
                                  ((tmp1 * tmp4 + tmp1 * tmp5) * (tmp1 * tmp4 + tmp1 * tmp5)) / (((tmp2) * (tmp2) * (tmp2) * (tmp2)) * tmp9);

        if (r < integration_radius) {
          const REAL gf_of_x = in_gfs[IDX4(gf_index, i0, i1, i2)];
          const REAL dV = sqrtdetgamma * dxx0 * dxx1 * dxx2;
          aux_gfs[IDX4(L2_SQUARED_DVGF, i0, i1, i2)] = gf_of_x * gf_of_x * dV;
          aux_gfs[IDX4(L2_DVGF, i0, i1, i2)] = dV;
        } // END if(r < integration_radius)

      } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
    } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
  } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
}

/*
 * Compute l2-norm of a gridfunction assuming a single grid.
 */
REAL compute_L2_norm_of_gridfunction(commondata_struct *restrict commondata, griddata_struct *restrict griddata, const REAL integration_radius,
                                     const int gf_index, const REAL *restrict in_gf) {

  // Unpack grid parameters assuming a single grid
  const int grid = 0;
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  REAL *restrict x0 = griddata[grid].xx[0];
  REAL *restrict x1 = griddata[grid].xx[1];
  REAL *restrict x2 = griddata[grid].xx[2];
  REAL *restrict in_gfs = griddata[grid].gridfuncs.diagnostic_output_gfs;
  REAL *restrict aux_gfs = griddata[grid].gridfuncs.diagnostic_output_gfs2;

  // Since we're performing sums, make sure arrays are zero'd
  cudaMemset(aux_gfs, 0, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);

  // Set summation variables to compute l2-norm

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = NGHOSTS;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir, (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                       (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  compute_L2_norm_of_gridfunction_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(x0, x1, x2, in_gfs, aux_gfs, integration_radius,
                                                                                                     gf_index);
  cudaCheckErrors(cudaKernel, "compute_L2_norm_of_gridfunction_gpu failure");

  REAL squared_sum = find_global__sum(&aux_gfs[IDX4(L2_SQUARED_DVGF, 0, 0, 0)], Nxx_plus_2NGHOSTS_tot);
  REAL volume_sum = find_global__sum(&aux_gfs[IDX4(L2_DVGF, 0, 0, 0)], Nxx_plus_2NGHOSTS_tot);
  // Compute and output the log of the l2-norm.
  return log10(1e-16 + sqrt(squared_sum / volume_sum)); // 1e-16 + ... avoids log10(0)
}
