#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
/*
 * GPU Kernel: compute_ds_min__gpu.
 * GPU Kernel to compute local ds_min per grid point.
 */
__global__ static void compute_ds_min__gpu(const REAL *restrict x0, const REAL *restrict x1, const REAL *restrict x2, REAL *restrict ds_min) {
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

  for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
    [[maybe_unused]] const REAL xx2 = x2[i2];
    for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
      [[maybe_unused]] const REAL xx1 = x1[i1];
      for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
        [[maybe_unused]] const REAL xx0 = x0[i0];
        REAL dsmin0, dsmin1, dsmin2;
        /*
         *  Original SymPy expressions:
         *  "[dsmin0 = Abs(AMAX*dxx0*(exp(xx0/SINHWAA)/SINHWAA + exp(-xx0/SINHWAA)/SINHWAA)*sqrt(AMAX**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2*sin(xx1)**2)/(exp(1/SINHWAA) -
         * exp(-1/SINHWAA)))/Abs(sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2))]"
         *  "[dsmin1 = Abs(dxx1*sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * bScale**2*sin(xx1)**2))]"
         *  "[dsmin2 = Abs(AMAX*dxx2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))*sin(xx1)/(exp(1/SINHWAA) - exp(-1/SINHWAA)))]"
         */
        const REAL tmp1 = (1.0 / (SINHWAA));
        const REAL tmp8 = sin(xx1);
        const REAL tmp2 = exp(tmp1) - exp(-tmp1);
        const REAL tmp4 = exp(tmp1 * xx0);
        const REAL tmp5 = exp(-tmp1 * xx0);
        const REAL tmp10 = AMAX / tmp2;
        const REAL tmp6 = tmp4 - tmp5;
        const REAL tmp7 = ((AMAX) * (AMAX)) * ((tmp6) * (tmp6)) / ((tmp2) * (tmp2));
        const REAL tmp9 = sqrt(((bScale) * (bScale)) * ((tmp8) * (tmp8)) + tmp7);
        dsmin0 = fabs(dxx0 * tmp10 * tmp9 * (tmp1 * tmp4 + tmp1 * tmp5)) / fabs(sqrt(((bScale) * (bScale)) + tmp7));
        dsmin1 = fabs(dxx1 * tmp9);
        dsmin2 = fabs(dxx2 * tmp10 * tmp6 * tmp8);

        int idx = IDX3(i0, i1, i2);
        ds_min[idx] = MIN(dsmin0, MIN(dsmin1, dsmin2));

      } // END LOOP: for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0)
    } // END LOOP: for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1)
  } // END LOOP: for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2)
}

/*
 * Output minimum gridspacing ds_min on a SinhSymTP numerical grid.
 */
void cfl_limited_timestep__rfm__SinhSymTP(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                          bc_struct *restrict bcstruct) {
#include "../set_CodeParameters.h"

  const int Nxx_tot = (Nxx_plus_2NGHOSTS0) * (Nxx_plus_2NGHOSTS1) * (Nxx_plus_2NGHOSTS2);
  REAL *ds_min;
  REAL *restrict x0 = xx[0];
  REAL *restrict x1 = xx[1];
  REAL *restrict x2 = xx[2];

  // We only loop over a single GF array length
  cudaMalloc(&ds_min, sizeof(REAL) * Nxx_tot);
  cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

  const size_t threads_in_x_dir = 64;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

  dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir, (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                       (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  compute_ds_min__gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(x0, x1, x2, ds_min);
  cudaCheckErrors(cudaKernel, "compute_ds_min__gpu failure");

  REAL ds_min__global = find_global__min(ds_min, Nxx_tot);

  commondata->dt = MIN(commondata->dt, ds_min__global * commondata->CFL_FACTOR);
  cudaFree(ds_min);
}
