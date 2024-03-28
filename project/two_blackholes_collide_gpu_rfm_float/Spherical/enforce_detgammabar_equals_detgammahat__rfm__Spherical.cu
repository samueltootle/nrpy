#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Enforce det(gammabar) = det(gammahat) constraint. Required for strong hyperbolicity.
 */
__global__
void enforce_detgammabar_equals_detgammahat__rfm__Spherical_gpu(const REAL *restrict _f0_of_xx0, const REAL *restrict _f1_of_xx1, 
  const REAL *restrict _f1_of_xx1__D1, const REAL *restrict _f1_of_xx1__DD11, REAL *restrict in_gfs) {
  __attribute_maybe_unused__ int const & Nxx0 = d_params.Nxx0;
  __attribute_maybe_unused__ int const & Nxx1 = d_params.Nxx1;
  __attribute_maybe_unused__ int const & Nxx2 = d_params.Nxx2;

  __attribute_maybe_unused__ int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  __attribute_maybe_unused__ int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  __attribute_maybe_unused__ int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  __attribute_maybe_unused__ REAL const & invdxx0 = d_params.invdxx0;
  __attribute_maybe_unused__ REAL const & invdxx1 = d_params.invdxx1;
  __attribute_maybe_unused__ REAL const & invdxx2 = d_params.invdxx2;

  // Global data index - expecting a 1D dataset
  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
  const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
  // Thread strides
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;
  
  for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2+=stride2) {
    for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1+=stride1) {
      const REAL f1_of_xx1 = _f1_of_xx1[i1];
      const REAL f1_of_xx1__D1 = _f1_of_xx1__D1[i1];
      const REAL f1_of_xx1__DD11 = _f1_of_xx1__DD11[i1];

      for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0+=stride0) {
        const REAL f0_of_xx0 = _f0_of_xx0[i0];
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = hDD00 + 1;
        const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp2 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp4 = ((f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp5 = FDPart3tmp4 * hDD11 + FDPart3tmp4;
        const REAL FDPart3tmp6 = FDPart3tmp2 * FDPart3tmp4;
        const REAL FDPart3tmp7 = FDPart3tmp6 * hDD22 + FDPart3tmp6;
        const REAL FDPart3tmp8 = cbrtf(fabsf(FDPart3tmp1 * FDPart3tmp2) /
                                       (-FDPart3tmp0 * FDPart3tmp1 * FDPart3tmp2 * ((hDD12) * (hDD12)) + FDPart3tmp0 * FDPart3tmp5 * FDPart3tmp7 +
                                        2 * FDPart3tmp1 * FDPart3tmp2 * hDD01 * hDD02 * hDD12 - FDPart3tmp4 * FDPart3tmp7 * ((hDD01) * (hDD01)) -
                                        FDPart3tmp5 * FDPart3tmp6 * ((hDD02) * (hDD02))));
        in_gfs[IDX4(HDD00GF, i0, i1, i2)] = FDPart3tmp0 * FDPart3tmp8 - 1;
        in_gfs[IDX4(HDD01GF, i0, i1, i2)] = FDPart3tmp8 * hDD01;
        in_gfs[IDX4(HDD02GF, i0, i1, i2)] = FDPart3tmp8 * hDD02;
        in_gfs[IDX4(HDD11GF, i0, i1, i2)] = FDPart3tmp8 * (hDD11 + 1) - 1;
        in_gfs[IDX4(HDD12GF, i0, i1, i2)] = FDPart3tmp8 * hDD12;
        in_gfs[IDX4(HDD22GF, i0, i1, i2)] = FDPart3tmp8 * (hDD22 + 1) - 1;

      } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
    }   // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
  }     // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
}

void enforce_detgammabar_equals_detgammahat__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                            const rfm_struct *restrict rfmstruct, REAL *restrict in_gfs) {
#include "../set_CodeParameters.h"
  int threads_in_x_dir = 32; //MIN(1024, params->Nxx0 / 32);
  int threads_in_y_dir = 32; //MIN(1024 / threads_in_x_dir, params->Nxx1);
  int threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(
    (Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
    (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
    (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir
  );
  // enforce_detgammabar_equals_detgammahat__rfm__Spherical_gpu<<<1,1>>>(
  enforce_detgammabar_equals_detgammahat__rfm__Spherical_gpu<<<grid_blocks, block_threads>>>(
    rfmstruct->f0_of_xx0, rfmstruct->f1_of_xx1, 
    rfmstruct->f1_of_xx1__D1, rfmstruct->f1_of_xx1__DD11, in_gfs
  );    
}