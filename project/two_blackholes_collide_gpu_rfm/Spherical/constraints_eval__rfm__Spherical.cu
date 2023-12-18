#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Finite difference function for operator dD0, with FD accuracy order 4.
 */
__device__ REAL fd_function_dD0_fdorder4(const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1, const REAL FDPROTO_i0p2,
                                     const REAL invdxx0) {

  const REAL FD_result = invdxx0 * (FDPart1_Rational_1_12 * (FDPROTO_i0m2 - FDPROTO_i0p2) + FDPart1_Rational_2_3 * (-FDPROTO_i0m1 + FDPROTO_i0p1));

  return FD_result;
}
/*
 * Finite difference function for operator dD1, with FD accuracy order 4.
 */
__device__ REAL fd_function_dD1_fdorder4(const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1, const REAL FDPROTO_i1p2,
                                     const REAL invdxx1) {

  const REAL FD_result = invdxx1 * (FDPart1_Rational_1_12 * (FDPROTO_i1m2 - FDPROTO_i1p2) + FDPart1_Rational_2_3 * (-FDPROTO_i1m1 + FDPROTO_i1p1));

  return FD_result;
}
/*
 * Finite difference function for operator dD2, with FD accuracy order 4.
 */
__device__ REAL fd_function_dD2_fdorder4(const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1, const REAL FDPROTO_i2p2,
                                     const REAL invdxx2) {

  const REAL FD_result = invdxx2 * (FDPart1_Rational_1_12 * (FDPROTO_i2m2 - FDPROTO_i2p2) + FDPart1_Rational_2_3 * (-FDPROTO_i2m1 + FDPROTO_i2p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD00, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD00_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1,
                                       const REAL FDPROTO_i0p2, const REAL invdxx0) {

  const REAL FD_result = ((invdxx0) * (invdxx0)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i0m2 - FDPROTO_i0p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i0m1 + FDPROTO_i0p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD01, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD01_fdorder4(const REAL FDPROTO_i0m1_i1m1, const REAL FDPROTO_i0m1_i1m2, const REAL FDPROTO_i0m1_i1p1,
                                       const REAL FDPROTO_i0m1_i1p2, const REAL FDPROTO_i0m2_i1m1, const REAL FDPROTO_i0m2_i1m2,
                                       const REAL FDPROTO_i0m2_i1p1, const REAL FDPROTO_i0m2_i1p2, const REAL FDPROTO_i0p1_i1m1,
                                       const REAL FDPROTO_i0p1_i1m2, const REAL FDPROTO_i0p1_i1p1, const REAL FDPROTO_i0p1_i1p2,
                                       const REAL FDPROTO_i0p2_i1m1, const REAL FDPROTO_i0p2_i1m2, const REAL FDPROTO_i0p2_i1p1,
                                       const REAL FDPROTO_i0p2_i1p2, const REAL invdxx0, const REAL invdxx1) {

  const REAL FD_result = invdxx0 * invdxx1 *
                         (FDPart1_Rational_1_144 * (FDPROTO_i0m2_i1m2 - FDPROTO_i0m2_i1p2 - FDPROTO_i0p2_i1m2 + FDPROTO_i0p2_i1p2) +
                          FDPart1_Rational_1_18 * (-FDPROTO_i0m1_i1m2 + FDPROTO_i0m1_i1p2 - FDPROTO_i0m2_i1m1 + FDPROTO_i0m2_i1p1 +
                                                   FDPROTO_i0p1_i1m2 - FDPROTO_i0p1_i1p2 + FDPROTO_i0p2_i1m1 - FDPROTO_i0p2_i1p1) +
                          FDPart1_Rational_4_9 * (FDPROTO_i0m1_i1m1 - FDPROTO_i0m1_i1p1 - FDPROTO_i0p1_i1m1 + FDPROTO_i0p1_i1p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD02, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD02_fdorder4(const REAL FDPROTO_i0m1_i2m1, const REAL FDPROTO_i0m1_i2m2, const REAL FDPROTO_i0m1_i2p1,
                                       const REAL FDPROTO_i0m1_i2p2, const REAL FDPROTO_i0m2_i2m1, const REAL FDPROTO_i0m2_i2m2,
                                       const REAL FDPROTO_i0m2_i2p1, const REAL FDPROTO_i0m2_i2p2, const REAL FDPROTO_i0p1_i2m1,
                                       const REAL FDPROTO_i0p1_i2m2, const REAL FDPROTO_i0p1_i2p1, const REAL FDPROTO_i0p1_i2p2,
                                       const REAL FDPROTO_i0p2_i2m1, const REAL FDPROTO_i0p2_i2m2, const REAL FDPROTO_i0p2_i2p1,
                                       const REAL FDPROTO_i0p2_i2p2, const REAL invdxx0, const REAL invdxx2) {

  const REAL FD_result = invdxx0 * invdxx2 *
                         (FDPart1_Rational_1_144 * (FDPROTO_i0m2_i2m2 - FDPROTO_i0m2_i2p2 - FDPROTO_i0p2_i2m2 + FDPROTO_i0p2_i2p2) +
                          FDPart1_Rational_1_18 * (-FDPROTO_i0m1_i2m2 + FDPROTO_i0m1_i2p2 - FDPROTO_i0m2_i2m1 + FDPROTO_i0m2_i2p1 +
                                                   FDPROTO_i0p1_i2m2 - FDPROTO_i0p1_i2p2 + FDPROTO_i0p2_i2m1 - FDPROTO_i0p2_i2p1) +
                          FDPart1_Rational_4_9 * (FDPROTO_i0m1_i2m1 - FDPROTO_i0m1_i2p1 - FDPROTO_i0p1_i2m1 + FDPROTO_i0p1_i2p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD11, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD11_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1,
                                       const REAL FDPROTO_i1p2, const REAL invdxx1) {

  const REAL FD_result = ((invdxx1) * (invdxx1)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i1m2 - FDPROTO_i1p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i1m1 + FDPROTO_i1p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD12, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD12_fdorder4(const REAL FDPROTO_i1m1_i2m1, const REAL FDPROTO_i1m1_i2m2, const REAL FDPROTO_i1m1_i2p1,
                                       const REAL FDPROTO_i1m1_i2p2, const REAL FDPROTO_i1m2_i2m1, const REAL FDPROTO_i1m2_i2m2,
                                       const REAL FDPROTO_i1m2_i2p1, const REAL FDPROTO_i1m2_i2p2, const REAL FDPROTO_i1p1_i2m1,
                                       const REAL FDPROTO_i1p1_i2m2, const REAL FDPROTO_i1p1_i2p1, const REAL FDPROTO_i1p1_i2p2,
                                       const REAL FDPROTO_i1p2_i2m1, const REAL FDPROTO_i1p2_i2m2, const REAL FDPROTO_i1p2_i2p1,
                                       const REAL FDPROTO_i1p2_i2p2, const REAL invdxx1, const REAL invdxx2) {

  const REAL FD_result = invdxx1 * invdxx2 *
                         (FDPart1_Rational_1_144 * (FDPROTO_i1m2_i2m2 - FDPROTO_i1m2_i2p2 - FDPROTO_i1p2_i2m2 + FDPROTO_i1p2_i2p2) +
                          FDPart1_Rational_1_18 * (-FDPROTO_i1m1_i2m2 + FDPROTO_i1m1_i2p2 - FDPROTO_i1m2_i2m1 + FDPROTO_i1m2_i2p1 +
                                                   FDPROTO_i1p1_i2m2 - FDPROTO_i1p1_i2p2 + FDPROTO_i1p2_i2m1 - FDPROTO_i1p2_i2p1) +
                          FDPart1_Rational_4_9 * (FDPROTO_i1m1_i2m1 - FDPROTO_i1m1_i2p1 - FDPROTO_i1p1_i2m1 + FDPROTO_i1p1_i2p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD22, with FD accuracy order 4.
 */
__device__ REAL fd_function_dDD22_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1,
                                       const REAL FDPROTO_i2p2, const REAL invdxx2) {

  const REAL FD_result = ((invdxx2) * (invdxx2)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i2m2 - FDPROTO_i2p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i2m1 + FDPROTO_i2p1));

  return FD_result;
}

/*
 * Evaluate BSSN constraints.
 */
__global__
void constraints_eval__rfm__Spherical_gpu(const REAL *restrict _f0_of_xx0, const REAL *restrict _f1_of_xx1, 
  const REAL *restrict _f1_of_xx1__D1, const REAL *restrict _f1_of_xx1__DD11, const REAL *restrict in_gfs, 
    const REAL *restrict auxevol_gfs, REAL *restrict diagnostic_output_gfs) {
  int const & Nxx0 = d_params.Nxx0;
  int const & Nxx1 = d_params.Nxx1;
  int const & Nxx2 = d_params.Nxx2;

  int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  REAL const & invdxx0 = d_params.invdxx0;
  REAL const & invdxx1 = d_params.invdxx1;
  REAL const & invdxx2 = d_params.invdxx2;

  // Global data index - expecting a 1D dataset
  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
  const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
  // Thread strides
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;
  
  for(size_t i2 = tid2+NGHOSTS; i2 < Nxx2+NGHOSTS; i2 += stride2) {
    for(size_t i1 = tid1+NGHOSTS; i1 < Nxx1+NGHOSTS; i1 += stride1) {
      const REAL f1_of_xx1 = _f1_of_xx1[i1];
      const REAL f1_of_xx1__D1 = _f1_of_xx1__D1[i1];
      __attribute_maybe_unused__ const REAL f1_of_xx1__DD11 = _f1_of_xx1__DD11[i1];

      for(size_t i0 = tid0+NGHOSTS; i0 < Nxx0+NGHOSTS; i0 += stride0) {
        const REAL f0_of_xx0 = _f0_of_xx0[i0];
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL RbarDD00 = auxevol_gfs[IDX4(RBARDD00GF, i0, i1, i2)];
        const REAL RbarDD01 = auxevol_gfs[IDX4(RBARDD01GF, i0, i1, i2)];
        const REAL RbarDD02 = auxevol_gfs[IDX4(RBARDD02GF, i0, i1, i2)];
        const REAL RbarDD11 = auxevol_gfs[IDX4(RBARDD11GF, i0, i1, i2)];
        const REAL RbarDD12 = auxevol_gfs[IDX4(RBARDD12GF, i0, i1, i2)];
        const REAL RbarDD22 = auxevol_gfs[IDX4(RBARDD22GF, i0, i1, i2)];
        const REAL aDD00_i2m2 = in_gfs[IDX4(ADD00GF, i0, i1, i2 - 2)];
        const REAL aDD00_i2m1 = in_gfs[IDX4(ADD00GF, i0, i1, i2 - 1)];
        const REAL aDD00_i1m2 = in_gfs[IDX4(ADD00GF, i0, i1 - 2, i2)];
        const REAL aDD00_i1m1 = in_gfs[IDX4(ADD00GF, i0, i1 - 1, i2)];
        const REAL aDD00_i0m2 = in_gfs[IDX4(ADD00GF, i0 - 2, i1, i2)];
        const REAL aDD00_i0m1 = in_gfs[IDX4(ADD00GF, i0 - 1, i1, i2)];
        const REAL aDD00 = in_gfs[IDX4(ADD00GF, i0, i1, i2)];
        const REAL aDD00_i0p1 = in_gfs[IDX4(ADD00GF, i0 + 1, i1, i2)];
        const REAL aDD00_i0p2 = in_gfs[IDX4(ADD00GF, i0 + 2, i1, i2)];
        const REAL aDD00_i1p1 = in_gfs[IDX4(ADD00GF, i0, i1 + 1, i2)];
        const REAL aDD00_i1p2 = in_gfs[IDX4(ADD00GF, i0, i1 + 2, i2)];
        const REAL aDD00_i2p1 = in_gfs[IDX4(ADD00GF, i0, i1, i2 + 1)];
        const REAL aDD00_i2p2 = in_gfs[IDX4(ADD00GF, i0, i1, i2 + 2)];
        const REAL aDD01_i2m2 = in_gfs[IDX4(ADD01GF, i0, i1, i2 - 2)];
        const REAL aDD01_i2m1 = in_gfs[IDX4(ADD01GF, i0, i1, i2 - 1)];
        const REAL aDD01_i1m2 = in_gfs[IDX4(ADD01GF, i0, i1 - 2, i2)];
        const REAL aDD01_i1m1 = in_gfs[IDX4(ADD01GF, i0, i1 - 1, i2)];
        const REAL aDD01_i0m2 = in_gfs[IDX4(ADD01GF, i0 - 2, i1, i2)];
        const REAL aDD01_i0m1 = in_gfs[IDX4(ADD01GF, i0 - 1, i1, i2)];
        const REAL aDD01 = in_gfs[IDX4(ADD01GF, i0, i1, i2)];
        const REAL aDD01_i0p1 = in_gfs[IDX4(ADD01GF, i0 + 1, i1, i2)];
        const REAL aDD01_i0p2 = in_gfs[IDX4(ADD01GF, i0 + 2, i1, i2)];
        const REAL aDD01_i1p1 = in_gfs[IDX4(ADD01GF, i0, i1 + 1, i2)];
        const REAL aDD01_i1p2 = in_gfs[IDX4(ADD01GF, i0, i1 + 2, i2)];
        const REAL aDD01_i2p1 = in_gfs[IDX4(ADD01GF, i0, i1, i2 + 1)];
        const REAL aDD01_i2p2 = in_gfs[IDX4(ADD01GF, i0, i1, i2 + 2)];
        const REAL aDD02_i2m2 = in_gfs[IDX4(ADD02GF, i0, i1, i2 - 2)];
        const REAL aDD02_i2m1 = in_gfs[IDX4(ADD02GF, i0, i1, i2 - 1)];
        const REAL aDD02_i1m2 = in_gfs[IDX4(ADD02GF, i0, i1 - 2, i2)];
        const REAL aDD02_i1m1 = in_gfs[IDX4(ADD02GF, i0, i1 - 1, i2)];
        const REAL aDD02_i0m2 = in_gfs[IDX4(ADD02GF, i0 - 2, i1, i2)];
        const REAL aDD02_i0m1 = in_gfs[IDX4(ADD02GF, i0 - 1, i1, i2)];
        const REAL aDD02 = in_gfs[IDX4(ADD02GF, i0, i1, i2)];
        const REAL aDD02_i0p1 = in_gfs[IDX4(ADD02GF, i0 + 1, i1, i2)];
        const REAL aDD02_i0p2 = in_gfs[IDX4(ADD02GF, i0 + 2, i1, i2)];
        const REAL aDD02_i1p1 = in_gfs[IDX4(ADD02GF, i0, i1 + 1, i2)];
        const REAL aDD02_i1p2 = in_gfs[IDX4(ADD02GF, i0, i1 + 2, i2)];
        const REAL aDD02_i2p1 = in_gfs[IDX4(ADD02GF, i0, i1, i2 + 1)];
        const REAL aDD02_i2p2 = in_gfs[IDX4(ADD02GF, i0, i1, i2 + 2)];
        const REAL aDD11_i2m2 = in_gfs[IDX4(ADD11GF, i0, i1, i2 - 2)];
        const REAL aDD11_i2m1 = in_gfs[IDX4(ADD11GF, i0, i1, i2 - 1)];
        const REAL aDD11_i1m2 = in_gfs[IDX4(ADD11GF, i0, i1 - 2, i2)];
        const REAL aDD11_i1m1 = in_gfs[IDX4(ADD11GF, i0, i1 - 1, i2)];
        const REAL aDD11_i0m2 = in_gfs[IDX4(ADD11GF, i0 - 2, i1, i2)];
        const REAL aDD11_i0m1 = in_gfs[IDX4(ADD11GF, i0 - 1, i1, i2)];
        const REAL aDD11 = in_gfs[IDX4(ADD11GF, i0, i1, i2)];
        const REAL aDD11_i0p1 = in_gfs[IDX4(ADD11GF, i0 + 1, i1, i2)];
        const REAL aDD11_i0p2 = in_gfs[IDX4(ADD11GF, i0 + 2, i1, i2)];
        const REAL aDD11_i1p1 = in_gfs[IDX4(ADD11GF, i0, i1 + 1, i2)];
        const REAL aDD11_i1p2 = in_gfs[IDX4(ADD11GF, i0, i1 + 2, i2)];
        const REAL aDD11_i2p1 = in_gfs[IDX4(ADD11GF, i0, i1, i2 + 1)];
        const REAL aDD11_i2p2 = in_gfs[IDX4(ADD11GF, i0, i1, i2 + 2)];
        const REAL aDD12_i2m2 = in_gfs[IDX4(ADD12GF, i0, i1, i2 - 2)];
        const REAL aDD12_i2m1 = in_gfs[IDX4(ADD12GF, i0, i1, i2 - 1)];
        const REAL aDD12_i1m2 = in_gfs[IDX4(ADD12GF, i0, i1 - 2, i2)];
        const REAL aDD12_i1m1 = in_gfs[IDX4(ADD12GF, i0, i1 - 1, i2)];
        const REAL aDD12_i0m2 = in_gfs[IDX4(ADD12GF, i0 - 2, i1, i2)];
        const REAL aDD12_i0m1 = in_gfs[IDX4(ADD12GF, i0 - 1, i1, i2)];
        const REAL aDD12 = in_gfs[IDX4(ADD12GF, i0, i1, i2)];
        const REAL aDD12_i0p1 = in_gfs[IDX4(ADD12GF, i0 + 1, i1, i2)];
        const REAL aDD12_i0p2 = in_gfs[IDX4(ADD12GF, i0 + 2, i1, i2)];
        const REAL aDD12_i1p1 = in_gfs[IDX4(ADD12GF, i0, i1 + 1, i2)];
        const REAL aDD12_i1p2 = in_gfs[IDX4(ADD12GF, i0, i1 + 2, i2)];
        const REAL aDD12_i2p1 = in_gfs[IDX4(ADD12GF, i0, i1, i2 + 1)];
        const REAL aDD12_i2p2 = in_gfs[IDX4(ADD12GF, i0, i1, i2 + 2)];
        const REAL aDD22_i2m2 = in_gfs[IDX4(ADD22GF, i0, i1, i2 - 2)];
        const REAL aDD22_i2m1 = in_gfs[IDX4(ADD22GF, i0, i1, i2 - 1)];
        const REAL aDD22_i1m2 = in_gfs[IDX4(ADD22GF, i0, i1 - 2, i2)];
        const REAL aDD22_i1m1 = in_gfs[IDX4(ADD22GF, i0, i1 - 1, i2)];
        const REAL aDD22_i0m2 = in_gfs[IDX4(ADD22GF, i0 - 2, i1, i2)];
        const REAL aDD22_i0m1 = in_gfs[IDX4(ADD22GF, i0 - 1, i1, i2)];
        const REAL aDD22 = in_gfs[IDX4(ADD22GF, i0, i1, i2)];
        const REAL aDD22_i0p1 = in_gfs[IDX4(ADD22GF, i0 + 1, i1, i2)];
        const REAL aDD22_i0p2 = in_gfs[IDX4(ADD22GF, i0 + 2, i1, i2)];
        const REAL aDD22_i1p1 = in_gfs[IDX4(ADD22GF, i0, i1 + 1, i2)];
        const REAL aDD22_i1p2 = in_gfs[IDX4(ADD22GF, i0, i1 + 2, i2)];
        const REAL aDD22_i2p1 = in_gfs[IDX4(ADD22GF, i0, i1, i2 + 1)];
        const REAL aDD22_i2p2 = in_gfs[IDX4(ADD22GF, i0, i1, i2 + 2)];
        const REAL cf_i1m2_i2m2 = in_gfs[IDX4(CFGF, i0, i1 - 2, i2 - 2)];
        const REAL cf_i1m1_i2m2 = in_gfs[IDX4(CFGF, i0, i1 - 1, i2 - 2)];
        const REAL cf_i0m2_i2m2 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2 - 2)];
        const REAL cf_i0m1_i2m2 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2 - 2)];
        const REAL cf_i2m2 = in_gfs[IDX4(CFGF, i0, i1, i2 - 2)];
        const REAL cf_i0p1_i2m2 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2 - 2)];
        const REAL cf_i0p2_i2m2 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2 - 2)];
        const REAL cf_i1p1_i2m2 = in_gfs[IDX4(CFGF, i0, i1 + 1, i2 - 2)];
        const REAL cf_i1p2_i2m2 = in_gfs[IDX4(CFGF, i0, i1 + 2, i2 - 2)];
        const REAL cf_i1m2_i2m1 = in_gfs[IDX4(CFGF, i0, i1 - 2, i2 - 1)];
        const REAL cf_i1m1_i2m1 = in_gfs[IDX4(CFGF, i0, i1 - 1, i2 - 1)];
        const REAL cf_i0m2_i2m1 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2 - 1)];
        const REAL cf_i0m1_i2m1 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2 - 1)];
        const REAL cf_i2m1 = in_gfs[IDX4(CFGF, i0, i1, i2 - 1)];
        const REAL cf_i0p1_i2m1 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2 - 1)];
        const REAL cf_i0p2_i2m1 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2 - 1)];
        const REAL cf_i1p1_i2m1 = in_gfs[IDX4(CFGF, i0, i1 + 1, i2 - 1)];
        const REAL cf_i1p2_i2m1 = in_gfs[IDX4(CFGF, i0, i1 + 2, i2 - 1)];
        const REAL cf_i0m2_i1m2 = in_gfs[IDX4(CFGF, i0 - 2, i1 - 2, i2)];
        const REAL cf_i0m1_i1m2 = in_gfs[IDX4(CFGF, i0 - 1, i1 - 2, i2)];
        const REAL cf_i1m2 = in_gfs[IDX4(CFGF, i0, i1 - 2, i2)];
        const REAL cf_i0p1_i1m2 = in_gfs[IDX4(CFGF, i0 + 1, i1 - 2, i2)];
        const REAL cf_i0p2_i1m2 = in_gfs[IDX4(CFGF, i0 + 2, i1 - 2, i2)];
        const REAL cf_i0m2_i1m1 = in_gfs[IDX4(CFGF, i0 - 2, i1 - 1, i2)];
        const REAL cf_i0m1_i1m1 = in_gfs[IDX4(CFGF, i0 - 1, i1 - 1, i2)];
        const REAL cf_i1m1 = in_gfs[IDX4(CFGF, i0, i1 - 1, i2)];
        const REAL cf_i0p1_i1m1 = in_gfs[IDX4(CFGF, i0 + 1, i1 - 1, i2)];
        const REAL cf_i0p2_i1m1 = in_gfs[IDX4(CFGF, i0 + 2, i1 - 1, i2)];
        const REAL cf_i0m2 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2)];
        const REAL cf_i0m1 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2)];
        const REAL cf = in_gfs[IDX4(CFGF, i0, i1, i2)];
        const REAL cf_i0p1 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2)];
        const REAL cf_i0p2 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2)];
        const REAL cf_i0m2_i1p1 = in_gfs[IDX4(CFGF, i0 - 2, i1 + 1, i2)];
        const REAL cf_i0m1_i1p1 = in_gfs[IDX4(CFGF, i0 - 1, i1 + 1, i2)];
        const REAL cf_i1p1 = in_gfs[IDX4(CFGF, i0, i1 + 1, i2)];
        const REAL cf_i0p1_i1p1 = in_gfs[IDX4(CFGF, i0 + 1, i1 + 1, i2)];
        const REAL cf_i0p2_i1p1 = in_gfs[IDX4(CFGF, i0 + 2, i1 + 1, i2)];
        const REAL cf_i0m2_i1p2 = in_gfs[IDX4(CFGF, i0 - 2, i1 + 2, i2)];
        const REAL cf_i0m1_i1p2 = in_gfs[IDX4(CFGF, i0 - 1, i1 + 2, i2)];
        const REAL cf_i1p2 = in_gfs[IDX4(CFGF, i0, i1 + 2, i2)];
        const REAL cf_i0p1_i1p2 = in_gfs[IDX4(CFGF, i0 + 1, i1 + 2, i2)];
        const REAL cf_i0p2_i1p2 = in_gfs[IDX4(CFGF, i0 + 2, i1 + 2, i2)];
        const REAL cf_i1m2_i2p1 = in_gfs[IDX4(CFGF, i0, i1 - 2, i2 + 1)];
        const REAL cf_i1m1_i2p1 = in_gfs[IDX4(CFGF, i0, i1 - 1, i2 + 1)];
        const REAL cf_i0m2_i2p1 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2 + 1)];
        const REAL cf_i0m1_i2p1 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2 + 1)];
        const REAL cf_i2p1 = in_gfs[IDX4(CFGF, i0, i1, i2 + 1)];
        const REAL cf_i0p1_i2p1 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2 + 1)];
        const REAL cf_i0p2_i2p1 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2 + 1)];
        const REAL cf_i1p1_i2p1 = in_gfs[IDX4(CFGF, i0, i1 + 1, i2 + 1)];
        const REAL cf_i1p2_i2p1 = in_gfs[IDX4(CFGF, i0, i1 + 2, i2 + 1)];
        const REAL cf_i1m2_i2p2 = in_gfs[IDX4(CFGF, i0, i1 - 2, i2 + 2)];
        const REAL cf_i1m1_i2p2 = in_gfs[IDX4(CFGF, i0, i1 - 1, i2 + 2)];
        const REAL cf_i0m2_i2p2 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2 + 2)];
        const REAL cf_i0m1_i2p2 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2 + 2)];
        const REAL cf_i2p2 = in_gfs[IDX4(CFGF, i0, i1, i2 + 2)];
        const REAL cf_i0p1_i2p2 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2 + 2)];
        const REAL cf_i0p2_i2p2 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2 + 2)];
        const REAL cf_i1p1_i2p2 = in_gfs[IDX4(CFGF, i0, i1 + 1, i2 + 2)];
        const REAL cf_i1p2_i2p2 = in_gfs[IDX4(CFGF, i0, i1 + 2, i2 + 2)];
        const REAL hDD00_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 2)];
        const REAL hDD00_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 1)];
        const REAL hDD00_i1m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2)];
        const REAL hDD00_i1m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2)];
        const REAL hDD00_i0m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2)];
        const REAL hDD00_i0m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2)];
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD00_i0p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2)];
        const REAL hDD00_i0p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2)];
        const REAL hDD00_i1p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2)];
        const REAL hDD00_i1p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2)];
        const REAL hDD00_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 1)];
        const REAL hDD00_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 2)];
        const REAL hDD01_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 2)];
        const REAL hDD01_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 1)];
        const REAL hDD01_i1m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2)];
        const REAL hDD01_i1m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2)];
        const REAL hDD01_i0m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2)];
        const REAL hDD01_i0m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD01_i0p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2)];
        const REAL hDD01_i0p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2)];
        const REAL hDD01_i1p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2)];
        const REAL hDD01_i1p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2)];
        const REAL hDD01_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 1)];
        const REAL hDD01_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 2)];
        const REAL hDD02_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 2)];
        const REAL hDD02_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 1)];
        const REAL hDD02_i1m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2)];
        const REAL hDD02_i1m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2)];
        const REAL hDD02_i0m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2)];
        const REAL hDD02_i0m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD02_i0p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2)];
        const REAL hDD02_i0p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2)];
        const REAL hDD02_i1p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2)];
        const REAL hDD02_i1p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2)];
        const REAL hDD02_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 1)];
        const REAL hDD02_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 2)];
        const REAL hDD11_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 2)];
        const REAL hDD11_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 1)];
        const REAL hDD11_i1m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2)];
        const REAL hDD11_i1m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2)];
        const REAL hDD11_i0m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2)];
        const REAL hDD11_i0m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD11_i0p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2)];
        const REAL hDD11_i0p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2)];
        const REAL hDD11_i1p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2)];
        const REAL hDD11_i1p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2)];
        const REAL hDD11_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 1)];
        const REAL hDD11_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 2)];
        const REAL hDD12_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 2)];
        const REAL hDD12_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 1)];
        const REAL hDD12_i1m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2)];
        const REAL hDD12_i1m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2)];
        const REAL hDD12_i0m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2)];
        const REAL hDD12_i0m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD12_i0p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2)];
        const REAL hDD12_i0p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2)];
        const REAL hDD12_i1p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2)];
        const REAL hDD12_i1p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2)];
        const REAL hDD12_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 1)];
        const REAL hDD12_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 2)];
        const REAL hDD22_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 2)];
        const REAL hDD22_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 1)];
        const REAL hDD22_i1m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2)];
        const REAL hDD22_i1m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2)];
        const REAL hDD22_i0m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2)];
        const REAL hDD22_i0m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];
        const REAL hDD22_i0p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2)];
        const REAL hDD22_i0p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2)];
        const REAL hDD22_i1p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2)];
        const REAL hDD22_i1p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2)];
        const REAL hDD22_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 1)];
        const REAL hDD22_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 2)];
        const REAL trK_i2m2 = in_gfs[IDX4(TRKGF, i0, i1, i2 - 2)];
        const REAL trK_i2m1 = in_gfs[IDX4(TRKGF, i0, i1, i2 - 1)];
        const REAL trK_i1m2 = in_gfs[IDX4(TRKGF, i0, i1 - 2, i2)];
        const REAL trK_i1m1 = in_gfs[IDX4(TRKGF, i0, i1 - 1, i2)];
        const REAL trK_i0m2 = in_gfs[IDX4(TRKGF, i0 - 2, i1, i2)];
        const REAL trK_i0m1 = in_gfs[IDX4(TRKGF, i0 - 1, i1, i2)];
        const REAL trK = in_gfs[IDX4(TRKGF, i0, i1, i2)];
        const REAL trK_i0p1 = in_gfs[IDX4(TRKGF, i0 + 1, i1, i2)];
        const REAL trK_i0p2 = in_gfs[IDX4(TRKGF, i0 + 2, i1, i2)];
        const REAL trK_i1p1 = in_gfs[IDX4(TRKGF, i0, i1 + 1, i2)];
        const REAL trK_i1p2 = in_gfs[IDX4(TRKGF, i0, i1 + 2, i2)];
        const REAL trK_i2p1 = in_gfs[IDX4(TRKGF, i0, i1, i2 + 1)];
        const REAL trK_i2p2 = in_gfs[IDX4(TRKGF, i0, i1, i2 + 2)];
        const REAL aDD_dD000 = fd_function_dD0_fdorder4(aDD00_i0m1, aDD00_i0m2, aDD00_i0p1, aDD00_i0p2, invdxx0);
        const REAL aDD_dD001 = fd_function_dD1_fdorder4(aDD00_i1m1, aDD00_i1m2, aDD00_i1p1, aDD00_i1p2, invdxx1);
        const REAL aDD_dD002 = fd_function_dD2_fdorder4(aDD00_i2m1, aDD00_i2m2, aDD00_i2p1, aDD00_i2p2, invdxx2);
        const REAL aDD_dD010 = fd_function_dD0_fdorder4(aDD01_i0m1, aDD01_i0m2, aDD01_i0p1, aDD01_i0p2, invdxx0);
        const REAL aDD_dD011 = fd_function_dD1_fdorder4(aDD01_i1m1, aDD01_i1m2, aDD01_i1p1, aDD01_i1p2, invdxx1);
        const REAL aDD_dD012 = fd_function_dD2_fdorder4(aDD01_i2m1, aDD01_i2m2, aDD01_i2p1, aDD01_i2p2, invdxx2);
        const REAL aDD_dD020 = fd_function_dD0_fdorder4(aDD02_i0m1, aDD02_i0m2, aDD02_i0p1, aDD02_i0p2, invdxx0);
        const REAL aDD_dD021 = fd_function_dD1_fdorder4(aDD02_i1m1, aDD02_i1m2, aDD02_i1p1, aDD02_i1p2, invdxx1);
        const REAL aDD_dD022 = fd_function_dD2_fdorder4(aDD02_i2m1, aDD02_i2m2, aDD02_i2p1, aDD02_i2p2, invdxx2);
        const REAL aDD_dD110 = fd_function_dD0_fdorder4(aDD11_i0m1, aDD11_i0m2, aDD11_i0p1, aDD11_i0p2, invdxx0);
        const REAL aDD_dD111 = fd_function_dD1_fdorder4(aDD11_i1m1, aDD11_i1m2, aDD11_i1p1, aDD11_i1p2, invdxx1);
        const REAL aDD_dD112 = fd_function_dD2_fdorder4(aDD11_i2m1, aDD11_i2m2, aDD11_i2p1, aDD11_i2p2, invdxx2);
        const REAL aDD_dD120 = fd_function_dD0_fdorder4(aDD12_i0m1, aDD12_i0m2, aDD12_i0p1, aDD12_i0p2, invdxx0);
        const REAL aDD_dD121 = fd_function_dD1_fdorder4(aDD12_i1m1, aDD12_i1m2, aDD12_i1p1, aDD12_i1p2, invdxx1);
        const REAL aDD_dD122 = fd_function_dD2_fdorder4(aDD12_i2m1, aDD12_i2m2, aDD12_i2p1, aDD12_i2p2, invdxx2);
        const REAL aDD_dD220 = fd_function_dD0_fdorder4(aDD22_i0m1, aDD22_i0m2, aDD22_i0p1, aDD22_i0p2, invdxx0);
        const REAL aDD_dD221 = fd_function_dD1_fdorder4(aDD22_i1m1, aDD22_i1m2, aDD22_i1p1, aDD22_i1p2, invdxx1);
        const REAL aDD_dD222 = fd_function_dD2_fdorder4(aDD22_i2m1, aDD22_i2m2, aDD22_i2p1, aDD22_i2p2, invdxx2);
        const REAL cf_dD0 = fd_function_dD0_fdorder4(cf_i0m1, cf_i0m2, cf_i0p1, cf_i0p2, invdxx0);
        const REAL cf_dD1 = fd_function_dD1_fdorder4(cf_i1m1, cf_i1m2, cf_i1p1, cf_i1p2, invdxx1);
        const REAL cf_dD2 = fd_function_dD2_fdorder4(cf_i2m1, cf_i2m2, cf_i2p1, cf_i2p2, invdxx2);
        const REAL cf_dDD00 = fd_function_dDD00_fdorder4(cf, cf_i0m1, cf_i0m2, cf_i0p1, cf_i0p2, invdxx0);
        const REAL cf_dDD01 = fd_function_dDD01_fdorder4(cf_i0m1_i1m1, cf_i0m1_i1m2, cf_i0m1_i1p1, cf_i0m1_i1p2, cf_i0m2_i1m1, cf_i0m2_i1m2,
                                                         cf_i0m2_i1p1, cf_i0m2_i1p2, cf_i0p1_i1m1, cf_i0p1_i1m2, cf_i0p1_i1p1, cf_i0p1_i1p2,
                                                         cf_i0p2_i1m1, cf_i0p2_i1m2, cf_i0p2_i1p1, cf_i0p2_i1p2, invdxx0, invdxx1);
        const REAL cf_dDD02 = fd_function_dDD02_fdorder4(cf_i0m1_i2m1, cf_i0m1_i2m2, cf_i0m1_i2p1, cf_i0m1_i2p2, cf_i0m2_i2m1, cf_i0m2_i2m2,
                                                         cf_i0m2_i2p1, cf_i0m2_i2p2, cf_i0p1_i2m1, cf_i0p1_i2m2, cf_i0p1_i2p1, cf_i0p1_i2p2,
                                                         cf_i0p2_i2m1, cf_i0p2_i2m2, cf_i0p2_i2p1, cf_i0p2_i2p2, invdxx0, invdxx2);
        const REAL cf_dDD11 = fd_function_dDD11_fdorder4(cf, cf_i1m1, cf_i1m2, cf_i1p1, cf_i1p2, invdxx1);
        const REAL cf_dDD12 = fd_function_dDD12_fdorder4(cf_i1m1_i2m1, cf_i1m1_i2m2, cf_i1m1_i2p1, cf_i1m1_i2p2, cf_i1m2_i2m1, cf_i1m2_i2m2,
                                                         cf_i1m2_i2p1, cf_i1m2_i2p2, cf_i1p1_i2m1, cf_i1p1_i2m2, cf_i1p1_i2p1, cf_i1p1_i2p2,
                                                         cf_i1p2_i2m1, cf_i1p2_i2m2, cf_i1p2_i2p1, cf_i1p2_i2p2, invdxx1, invdxx2);
        const REAL cf_dDD22 = fd_function_dDD22_fdorder4(cf, cf_i2m1, cf_i2m2, cf_i2p1, cf_i2p2, invdxx2);
        const REAL hDD_dD000 = fd_function_dD0_fdorder4(hDD00_i0m1, hDD00_i0m2, hDD00_i0p1, hDD00_i0p2, invdxx0);
        const REAL hDD_dD001 = fd_function_dD1_fdorder4(hDD00_i1m1, hDD00_i1m2, hDD00_i1p1, hDD00_i1p2, invdxx1);
        const REAL hDD_dD002 = fd_function_dD2_fdorder4(hDD00_i2m1, hDD00_i2m2, hDD00_i2p1, hDD00_i2p2, invdxx2);
        const REAL hDD_dD010 = fd_function_dD0_fdorder4(hDD01_i0m1, hDD01_i0m2, hDD01_i0p1, hDD01_i0p2, invdxx0);
        const REAL hDD_dD011 = fd_function_dD1_fdorder4(hDD01_i1m1, hDD01_i1m2, hDD01_i1p1, hDD01_i1p2, invdxx1);
        const REAL hDD_dD012 = fd_function_dD2_fdorder4(hDD01_i2m1, hDD01_i2m2, hDD01_i2p1, hDD01_i2p2, invdxx2);
        const REAL hDD_dD020 = fd_function_dD0_fdorder4(hDD02_i0m1, hDD02_i0m2, hDD02_i0p1, hDD02_i0p2, invdxx0);
        const REAL hDD_dD021 = fd_function_dD1_fdorder4(hDD02_i1m1, hDD02_i1m2, hDD02_i1p1, hDD02_i1p2, invdxx1);
        const REAL hDD_dD022 = fd_function_dD2_fdorder4(hDD02_i2m1, hDD02_i2m2, hDD02_i2p1, hDD02_i2p2, invdxx2);
        const REAL hDD_dD110 = fd_function_dD0_fdorder4(hDD11_i0m1, hDD11_i0m2, hDD11_i0p1, hDD11_i0p2, invdxx0);
        const REAL hDD_dD111 = fd_function_dD1_fdorder4(hDD11_i1m1, hDD11_i1m2, hDD11_i1p1, hDD11_i1p2, invdxx1);
        const REAL hDD_dD112 = fd_function_dD2_fdorder4(hDD11_i2m1, hDD11_i2m2, hDD11_i2p1, hDD11_i2p2, invdxx2);
        const REAL hDD_dD120 = fd_function_dD0_fdorder4(hDD12_i0m1, hDD12_i0m2, hDD12_i0p1, hDD12_i0p2, invdxx0);
        const REAL hDD_dD121 = fd_function_dD1_fdorder4(hDD12_i1m1, hDD12_i1m2, hDD12_i1p1, hDD12_i1p2, invdxx1);
        const REAL hDD_dD122 = fd_function_dD2_fdorder4(hDD12_i2m1, hDD12_i2m2, hDD12_i2p1, hDD12_i2p2, invdxx2);
        const REAL hDD_dD220 = fd_function_dD0_fdorder4(hDD22_i0m1, hDD22_i0m2, hDD22_i0p1, hDD22_i0p2, invdxx0);
        const REAL hDD_dD221 = fd_function_dD1_fdorder4(hDD22_i1m1, hDD22_i1m2, hDD22_i1p1, hDD22_i1p2, invdxx1);
        const REAL hDD_dD222 = fd_function_dD2_fdorder4(hDD22_i2m1, hDD22_i2m2, hDD22_i2p1, hDD22_i2p2, invdxx2);
        const REAL trK_dD0 = fd_function_dD0_fdorder4(trK_i0m1, trK_i0m2, trK_i0p1, trK_i0p2, invdxx0);
        const REAL trK_dD1 = fd_function_dD1_fdorder4(trK_i1m1, trK_i1m2, trK_i1p1, trK_i1p2, invdxx1);
        const REAL trK_dD2 = fd_function_dD2_fdorder4(trK_i2m1, trK_i2m2, trK_i2p1, trK_i2p2, invdxx2);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = ((f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp2 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp3 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp4 = hDD00 + 1;
        const REAL FDPart3tmp18 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0)) * hDD12;
        const REAL FDPart3tmp19 = f0_of_xx0 * f1_of_xx1;
        const REAL FDPart3tmp28 = 2 * f0_of_xx0;
        const REAL FDPart3tmp36 = aDD02 * f1_of_xx1;
        const REAL FDPart3tmp67 = aDD01 * f0_of_xx0;
        const REAL FDPart3tmp111 = (1.0 / (cf));
        const REAL FDPart3tmp134 = f0_of_xx0 * hDD_dD012;
        const REAL FDPart3tmp177 = ((cf) * (cf) * (cf) * (cf));
        const REAL FDPart3tmp1 = FDPart3tmp0 * aDD11;
        const REAL FDPart3tmp5 = FDPart3tmp2 * FDPart3tmp3 * ((hDD12) * (hDD12));
        const REAL FDPart3tmp6 = FDPart3tmp0 * hDD11 + FDPart3tmp0;
        const REAL FDPart3tmp7 = FDPart3tmp0 * FDPart3tmp3;
        const REAL FDPart3tmp10 = FDPart3tmp0 * ((hDD01) * (hDD01));
        const REAL FDPart3tmp13 = FDPart3tmp0 * f1_of_xx1;
        const REAL FDPart3tmp29 = FDPart3tmp28 * aDD01;
        const REAL FDPart3tmp37 = FDPart3tmp28 * FDPart3tmp36;
        const REAL FDPart3tmp64 = FDPart3tmp36 * f0_of_xx0;
        const REAL FDPart3tmp101 = (1.0 / ((cf) * (cf)));
        const REAL FDPart3tmp114 = FDPart3tmp28 * hDD_dD010 + 2 * hDD01 - hDD_dD001;
        const REAL FDPart3tmp117 = FDPart3tmp28 * f1_of_xx1;
        const REAL FDPart3tmp126 = (0.5) * FDPart3tmp111 * cf_dD1;
        const REAL FDPart3tmp132 = FDPart3tmp0 * hDD_dD110 + FDPart3tmp28 * hDD11 + FDPart3tmp28;
        const REAL FDPart3tmp137 = f0_of_xx0 * f1_of_xx1__D1 * hDD02;
        const REAL FDPart3tmp148 = FDPart3tmp0 * hDD_dD111;
        const REAL FDPart3tmp150 = FDPart3tmp0 * hDD_dD112;
        const REAL FDPart3tmp157 = FDPart3tmp28 * FDPart3tmp3;
        const REAL FDPart3tmp181 = 12 * FDPart3tmp67;
        const REAL FDPart3tmp8 = FDPart3tmp7 * ((hDD02) * (hDD02));
        const REAL FDPart3tmp9 = FDPart3tmp7 * hDD22 + FDPart3tmp7;
        const REAL FDPart3tmp20 = FDPart3tmp18 * f1_of_xx1 * hDD01 - FDPart3tmp19 * FDPart3tmp6 * hDD02;
        const REAL FDPart3tmp24 = -FDPart3tmp10 + FDPart3tmp4 * FDPart3tmp6;
        const REAL FDPart3tmp26 = FDPart3tmp7 * aDD22;
        const REAL FDPart3tmp32 = 2 * FDPart3tmp13;
        const REAL FDPart3tmp62 = FDPart3tmp13 * aDD12;
        const REAL FDPart3tmp106 = 2 * FDPart3tmp101;
        const REAL FDPart3tmp118 = FDPart3tmp117 * hDD_dD020 + 2 * f1_of_xx1 * hDD02 - hDD_dD002;
        const REAL FDPart3tmp122 = (0.5) * FDPart3tmp111 * cf_dD2;
        const REAL FDPart3tmp129 = (0.5) * FDPart3tmp111 * cf_dD0;
        const REAL FDPart3tmp138 = FDPart3tmp137 + FDPart3tmp19 * hDD_dD021;
        const REAL FDPart3tmp149 = -FDPart3tmp132 + 2 * f0_of_xx0 * hDD_dD011;
        const REAL FDPart3tmp158 = FDPart3tmp157 * hDD22 + FDPart3tmp157 + FDPart3tmp7 * hDD_dD220;
        const REAL FDPart3tmp171 = FDPart3tmp7 * hDD_dD222;
        const REAL FDPart3tmp11 = -FDPart3tmp10 * FDPart3tmp9 + 2 * FDPart3tmp2 * FDPart3tmp3 * hDD01 * hDD02 * hDD12 - FDPart3tmp4 * FDPart3tmp5 +
                                  FDPart3tmp4 * FDPart3tmp6 * FDPart3tmp9 - FDPart3tmp6 * FDPart3tmp8;
        const REAL FDPart3tmp15 = -FDPart3tmp13 * FDPart3tmp4 * hDD12 + FDPart3tmp13 * hDD01 * hDD02;
        const REAL FDPart3tmp33 = FDPart3tmp32 * aDD12;
        const REAL FDPart3tmp40 = FDPart3tmp18 * FDPart3tmp3 * hDD02 - FDPart3tmp9 * f0_of_xx0 * hDD01;
        const REAL FDPart3tmp43 = FDPart3tmp4 * FDPart3tmp9 - FDPart3tmp8;
        const REAL FDPart3tmp54 = -FDPart3tmp5 + FDPart3tmp6 * FDPart3tmp9;
        const REAL FDPart3tmp141 = FDPart3tmp117 * hDD12 + FDPart3tmp13 * hDD_dD120;
        const REAL FDPart3tmp152 = 2 * FDPart3tmp0 * f1_of_xx1__D1 * hDD12 - FDPart3tmp150 + FDPart3tmp32 * hDD_dD121;
        const REAL FDPart3tmp163 = FDPart3tmp32 * f1_of_xx1__D1;
        const REAL FDPart3tmp165 = -FDPart3tmp117 * hDD12 - FDPart3tmp13 * hDD_dD120 + FDPart3tmp134 + FDPart3tmp138;
        const REAL FDPart3tmp173 = -FDPart3tmp158 + 2 * f0_of_xx0 * f1_of_xx1 * hDD_dD022;
        const REAL FDPart3tmp182 = 12 * FDPart3tmp62;
        const REAL FDPart3tmp12 = (1.0 / ((FDPart3tmp11) * (FDPart3tmp11)));
        const REAL FDPart3tmp100 = (1.0 / (FDPart3tmp11));
        const REAL FDPart3tmp142 = -FDPart3tmp134 + FDPart3tmp138 + FDPart3tmp141;
        const REAL FDPart3tmp156 = FDPart3tmp134 - FDPart3tmp137 + FDPart3tmp141 - FDPart3tmp19 * hDD_dD021;
        const REAL FDPart3tmp164 = FDPart3tmp163 * hDD22 + FDPart3tmp163 + FDPart3tmp7 * hDD_dD221;
        const REAL FDPart3tmp16 = FDPart3tmp12 * ((FDPart3tmp15) * (FDPart3tmp15));
        const REAL FDPart3tmp22 = FDPart3tmp12 * aDD00;
        const REAL FDPart3tmp27 = FDPart3tmp12 * ((FDPart3tmp24) * (FDPart3tmp24)) * FDPart3tmp26;
        const REAL FDPart3tmp30 = FDPart3tmp12 * FDPart3tmp15;
        const REAL FDPart3tmp34 = FDPart3tmp12 * FDPart3tmp24;
        const REAL FDPart3tmp44 = FDPart3tmp12 * ((FDPart3tmp43) * (FDPart3tmp43));
        const REAL FDPart3tmp47 = FDPart3tmp12 * FDPart3tmp43;
        const REAL FDPart3tmp50 = FDPart3tmp12 * ((FDPart3tmp20) * (FDPart3tmp20));
        const REAL FDPart3tmp52 = FDPart3tmp12 * ((FDPart3tmp40) * (FDPart3tmp40));
        const REAL FDPart3tmp58 = FDPart3tmp12 * FDPart3tmp20 * FDPart3tmp40;
        const REAL FDPart3tmp102 = 4 * FDPart3tmp100 * FDPart3tmp101;
        const REAL FDPart3tmp105 = FDPart3tmp100 * FDPart3tmp24;
        const REAL FDPart3tmp108 = FDPart3tmp100 * FDPart3tmp43;
        const REAL FDPart3tmp110 = FDPart3tmp100 * FDPart3tmp54;
        const REAL FDPart3tmp112 = (0.5) * FDPart3tmp100;
        const REAL FDPart3tmp131 = (0.5) * FDPart3tmp100 * hDD_dD001;
        const REAL FDPart3tmp147 = 16 * FDPart3tmp100;
        const REAL FDPart3tmp172 = 2 * FDPart3tmp0 * f1_of_xx1 * hDD_dD122 - FDPart3tmp164;
        const REAL FDPart3tmp178 = (2.0 / 3.0) * FDPart3tmp100;
        const REAL FDPart3tmp23 = ((FDPart3tmp20) * (FDPart3tmp20)) * FDPart3tmp22;
        const REAL FDPart3tmp31 = FDPart3tmp20 * FDPart3tmp30;
        const REAL FDPart3tmp35 = FDPart3tmp15 * FDPart3tmp34;
        const REAL FDPart3tmp42 = FDPart3tmp22 * ((FDPart3tmp40) * (FDPart3tmp40));
        const REAL FDPart3tmp46 = FDPart3tmp30 * FDPart3tmp40;
        const REAL FDPart3tmp48 = FDPart3tmp15 * FDPart3tmp47;
        const REAL FDPart3tmp49 = FDPart3tmp40 * FDPart3tmp47;
        const REAL FDPart3tmp56 = FDPart3tmp22 * ((FDPart3tmp54) * (FDPart3tmp54));
        const REAL FDPart3tmp60 = FDPart3tmp12 * FDPart3tmp20 * FDPart3tmp54;
        const REAL FDPart3tmp61 = FDPart3tmp12 * FDPart3tmp40 * FDPart3tmp54;
        const REAL FDPart3tmp69 = FDPart3tmp20 * FDPart3tmp22 * FDPart3tmp40;
        const REAL FDPart3tmp71 = FDPart3tmp34 * FDPart3tmp40;
        const REAL FDPart3tmp73 = FDPart3tmp20 * FDPart3tmp47;
        const REAL FDPart3tmp75 = FDPart3tmp34 * FDPart3tmp43;
        const REAL FDPart3tmp83 = FDPart3tmp30 * FDPart3tmp54;
        const REAL FDPart3tmp113 = FDPart3tmp112 * FDPart3tmp20;
        const REAL FDPart3tmp115 = FDPart3tmp112 * FDPart3tmp15;
        const REAL FDPart3tmp119 = (0.5) * FDPart3tmp105;
        const REAL FDPart3tmp123 = FDPart3tmp112 * FDPart3tmp40;
        const REAL FDPart3tmp124 = (0.5) * FDPart3tmp108;
        const REAL FDPart3tmp127 = (0.5) * FDPart3tmp110;
        const REAL FDPart3tmp79 = FDPart3tmp20 * FDPart3tmp26 * FDPart3tmp34;
        const REAL FDPart3tmp86 = FDPart3tmp20 * FDPart3tmp22 * FDPart3tmp54;
        const REAL FDPart3tmp88 = FDPart3tmp34 * FDPart3tmp54 * FDPart3tmp64;
        const REAL FDPart3tmp96 = FDPart3tmp22 * FDPart3tmp40 * FDPart3tmp54;
        const REAL FDPart3tmp98 = FDPart3tmp47 * FDPart3tmp54 * FDPart3tmp67;
        const REAL FDPart3tmp120 = -FDPart3tmp113 * hDD_dD000 - FDPart3tmp114 * FDPart3tmp115 - FDPart3tmp118 * FDPart3tmp119;
        const REAL FDPart3tmp125 = -FDPart3tmp114 * FDPart3tmp124 - FDPart3tmp115 * FDPart3tmp118 - FDPart3tmp123 * hDD_dD000;
        const REAL FDPart3tmp128 = -FDPart3tmp113 * FDPart3tmp118 - FDPart3tmp114 * FDPart3tmp123 - FDPart3tmp127 * hDD_dD000;
        const REAL FDPart3tmp143 = -FDPart3tmp112 * FDPart3tmp132 * FDPart3tmp15 - FDPart3tmp119 * FDPart3tmp142 - FDPart3tmp131 * FDPart3tmp20;
        const REAL FDPart3tmp145 = -FDPart3tmp112 * FDPart3tmp142 * FDPart3tmp15 - FDPart3tmp124 * FDPart3tmp132 - FDPart3tmp131 * FDPart3tmp40;
        const REAL FDPart3tmp146 =
            -0.5 * FDPart3tmp110 * hDD_dD001 - FDPart3tmp112 * FDPart3tmp132 * FDPart3tmp40 - FDPart3tmp112 * FDPart3tmp142 * FDPart3tmp20;
        const REAL FDPart3tmp153 = -FDPart3tmp113 * FDPart3tmp149 - FDPart3tmp115 * FDPart3tmp148 - FDPart3tmp119 * FDPart3tmp152;
        const REAL FDPart3tmp154 = -FDPart3tmp115 * FDPart3tmp152 - FDPart3tmp123 * FDPart3tmp149 - FDPart3tmp124 * FDPart3tmp148;
        const REAL FDPart3tmp155 = -FDPart3tmp113 * FDPart3tmp152 - FDPart3tmp123 * FDPart3tmp148 - FDPart3tmp127 * FDPart3tmp149;
        const REAL FDPart3tmp159 = -FDPart3tmp113 * hDD_dD002 - FDPart3tmp115 * FDPart3tmp156 - FDPart3tmp119 * FDPart3tmp158;
        const REAL FDPart3tmp161 = -FDPart3tmp112 * FDPart3tmp15 * FDPart3tmp158 - FDPart3tmp123 * hDD_dD002 - FDPart3tmp124 * FDPart3tmp156;
        const REAL FDPart3tmp162 = -FDPart3tmp112 * FDPart3tmp158 * FDPart3tmp20 - FDPart3tmp123 * FDPart3tmp156 - FDPart3tmp127 * hDD_dD002;
        const REAL FDPart3tmp167 = -FDPart3tmp112 * FDPart3tmp165 * FDPart3tmp20 - FDPart3tmp115 * FDPart3tmp150 - FDPart3tmp119 * FDPart3tmp164;
        const REAL FDPart3tmp169 =
            -FDPart3tmp112 * FDPart3tmp15 * FDPart3tmp164 - FDPart3tmp112 * FDPart3tmp165 * FDPart3tmp40 - FDPart3tmp124 * FDPart3tmp150;
        const REAL FDPart3tmp170 = -FDPart3tmp112 * FDPart3tmp164 * FDPart3tmp20 - FDPart3tmp123 * FDPart3tmp150 - FDPart3tmp127 * FDPart3tmp165;
        const REAL FDPart3tmp174 = -FDPart3tmp113 * FDPart3tmp173 - FDPart3tmp115 * FDPart3tmp172 - FDPart3tmp119 * FDPart3tmp171;
        const REAL FDPart3tmp175 = -FDPart3tmp115 * FDPart3tmp171 - FDPart3tmp123 * FDPart3tmp173 - FDPart3tmp124 * FDPart3tmp172;
        const REAL FDPart3tmp176 = -FDPart3tmp113 * FDPart3tmp171 - FDPart3tmp123 * FDPart3tmp172 - FDPart3tmp127 * FDPart3tmp173;
        const REAL FDPart3tmp183 = 6 * FDPart3tmp1 * FDPart3tmp48 + 6 * FDPart3tmp16 * FDPart3tmp62 + 6 * FDPart3tmp26 * FDPart3tmp35 +
                                   6 * FDPart3tmp31 * FDPart3tmp64 + 6 * FDPart3tmp46 * FDPart3tmp67 + 6 * FDPart3tmp62 * FDPart3tmp75 +
                                   6 * FDPart3tmp64 * FDPart3tmp71 + 6 * FDPart3tmp67 * FDPart3tmp73 + 6 * FDPart3tmp69;
        const REAL FDPart3tmp184 = 6 * FDPart3tmp1 * FDPart3tmp46 + 6 * FDPart3tmp31 * FDPart3tmp62 + 6 * FDPart3tmp50 * FDPart3tmp64 +
                                   6 * FDPart3tmp58 * FDPart3tmp67 + 6 * FDPart3tmp62 * FDPart3tmp71 + 6 * FDPart3tmp67 * FDPart3tmp83 +
                                   6 * FDPart3tmp79 + 6 * FDPart3tmp86 + 6 * FDPart3tmp88;
        const REAL FDPart3tmp185 =
            FDPart3tmp12 * (FDPart3tmp120 * FDPart3tmp37 + FDPart3tmp125 * FDPart3tmp29 + 2 * FDPart3tmp128 * aDD00 + aDD_dD000);
        const REAL FDPart3tmp195 =
            FDPart3tmp0 * aDD_dD111 + 2 * FDPart3tmp1 * FDPart3tmp154 + FDPart3tmp153 * FDPart3tmp33 + FDPart3tmp155 * FDPart3tmp29;
        const REAL FDPart3tmp202 =
            FDPart3tmp12 * (2 * FDPart3tmp174 * FDPart3tmp26 + FDPart3tmp175 * FDPart3tmp33 + FDPart3tmp176 * FDPart3tmp37 + FDPart3tmp7 * aDD_dD222);
        const REAL FDPart3tmp208 = FDPart3tmp1 * FDPart3tmp161 + FDPart3tmp159 * FDPart3tmp62 + FDPart3tmp162 * FDPart3tmp67;
        const REAL FDPart3tmp209 = FDPart3tmp143 * FDPart3tmp26 + FDPart3tmp145 * FDPart3tmp62 + FDPart3tmp146 * FDPart3tmp64;
        const REAL FDPart3tmp211 = FDPart3tmp167 * FDPart3tmp64 + FDPart3tmp169 * FDPart3tmp67 + FDPart3tmp170 * aDD00;
        const REAL FDPart3tmp220 = 6 * FDPart3tmp1 * FDPart3tmp49 + 6 * FDPart3tmp26 * FDPart3tmp31 + 6 * FDPart3tmp46 * FDPart3tmp62 +
                                   6 * FDPart3tmp52 * FDPart3tmp67 + 6 * FDPart3tmp58 * FDPart3tmp64 + 6 * FDPart3tmp62 * FDPart3tmp73 +
                                   6 * FDPart3tmp64 * FDPart3tmp83 + 6 * FDPart3tmp96 + 6 * FDPart3tmp98;
        const REAL FDPart3tmp188 = FDPart3tmp143 * FDPart3tmp37 + FDPart3tmp145 * FDPart3tmp29 + 2 * FDPart3tmp146 * aDD00 + aDD_dD001;
        const REAL FDPart3tmp190 = FDPart3tmp159 * FDPart3tmp37 + FDPart3tmp161 * FDPart3tmp29 + 2 * FDPart3tmp162 * aDD00 + aDD_dD002;
        const REAL FDPart3tmp192 = FDPart3tmp0 * aDD_dD110 + 2 * FDPart3tmp1 * FDPart3tmp145 + FDPart3tmp143 * FDPart3tmp33 +
                                   FDPart3tmp146 * FDPart3tmp29 + FDPart3tmp28 * aDD11;
        const REAL FDPart3tmp194 =
            FDPart3tmp0 * aDD_dD112 + 2 * FDPart3tmp1 * FDPart3tmp169 + FDPart3tmp167 * FDPart3tmp33 + FDPart3tmp170 * FDPart3tmp29;
        const REAL FDPart3tmp197 = FDPart3tmp157 * aDD22 + 2 * FDPart3tmp159 * FDPart3tmp26 + FDPart3tmp161 * FDPart3tmp33 +
                                   FDPart3tmp162 * FDPart3tmp37 + FDPart3tmp7 * aDD_dD220;
        const REAL FDPart3tmp200 = FDPart3tmp163 * aDD22 + 2 * FDPart3tmp167 * FDPart3tmp26 + FDPart3tmp169 * FDPart3tmp33 +
                                   FDPart3tmp170 * FDPart3tmp37 + FDPart3tmp7 * aDD_dD221;
        const REAL FDPart3tmp203 = FDPart3tmp1 * FDPart3tmp125 + FDPart3tmp120 * FDPart3tmp62 + FDPart3tmp128 * FDPart3tmp67 +
                                   FDPart3tmp143 * FDPart3tmp64 + FDPart3tmp145 * FDPart3tmp67 + FDPart3tmp146 * aDD00 + aDD01 +
                                   aDD_dD010 * f0_of_xx0;
        const REAL FDPart3tmp205 = FDPart3tmp120 * FDPart3tmp26 + FDPart3tmp125 * FDPart3tmp62 + FDPart3tmp128 * FDPart3tmp64 +
                                   FDPart3tmp159 * FDPart3tmp64 + FDPart3tmp161 * FDPart3tmp67 + FDPart3tmp162 * aDD00 + FDPart3tmp19 * aDD_dD020 +
                                   FDPart3tmp36;
        const REAL FDPart3tmp207 = FDPart3tmp1 * FDPart3tmp145 + FDPart3tmp143 * FDPart3tmp62 + FDPart3tmp146 * FDPart3tmp67 +
                                   FDPart3tmp153 * FDPart3tmp64 + FDPart3tmp154 * FDPart3tmp67 + FDPart3tmp155 * aDD00 + aDD_dD011 * f0_of_xx0;
        const REAL FDPart3tmp210 = FDPart3tmp117 * aDD12 + FDPart3tmp13 * aDD_dD120 + FDPart3tmp208 + FDPart3tmp209;
        const REAL FDPart3tmp212 = FDPart3tmp19 * aDD_dD021 + FDPart3tmp209 + FDPart3tmp211 + aDD02 * f0_of_xx0 * f1_of_xx1__D1;
        const REAL FDPart3tmp213 = FDPart3tmp0 * aDD12 * f1_of_xx1__D1 + FDPart3tmp1 * FDPart3tmp169 + FDPart3tmp13 * aDD_dD121 +
                                   FDPart3tmp153 * FDPart3tmp26 + FDPart3tmp154 * FDPart3tmp62 + FDPart3tmp155 * FDPart3tmp64 +
                                   FDPart3tmp167 * FDPart3tmp62 + FDPart3tmp170 * FDPart3tmp67;
        const REAL FDPart3tmp214 = FDPart3tmp208 + FDPart3tmp211 + aDD_dD012 * f0_of_xx0;
        const REAL FDPart3tmp215 = FDPart3tmp159 * FDPart3tmp26 + FDPart3tmp161 * FDPart3tmp62 + FDPart3tmp162 * FDPart3tmp64 +
                                   FDPart3tmp174 * FDPart3tmp64 + FDPart3tmp175 * FDPart3tmp67 + FDPart3tmp176 * aDD00 + FDPart3tmp19 * aDD_dD022;
        const REAL FDPart3tmp217 = FDPart3tmp1 * FDPart3tmp175 + FDPart3tmp13 * aDD_dD122 + FDPart3tmp167 * FDPart3tmp26 +
                                   FDPart3tmp169 * FDPart3tmp62 + FDPart3tmp170 * FDPart3tmp64 + FDPart3tmp174 * FDPart3tmp62 +
                                   FDPart3tmp176 * FDPart3tmp67;
        const REAL FDPart3tmp204 = FDPart3tmp203 * FDPart3tmp54;
        const REAL FDPart3tmp206 = FDPart3tmp205 * FDPart3tmp54;
        const REAL FDPart3tmp218 =
            -2.0 / 3.0 * FDPart3tmp105 * trK_dD2 -
            FDPart3tmp122 * (6 * FDPart3tmp1 * FDPart3tmp16 + FDPart3tmp181 * FDPart3tmp31 + FDPart3tmp182 * FDPart3tmp35 +
                             12 * FDPart3tmp20 * FDPart3tmp34 * FDPart3tmp64 + 6 * FDPart3tmp23 + 6 * FDPart3tmp27) -
            FDPart3tmp126 * FDPart3tmp183 - FDPart3tmp129 * FDPart3tmp184 - FDPart3tmp15 * FDPart3tmp178 * trK_dD1 +
            FDPart3tmp15 * FDPart3tmp200 * FDPart3tmp34 + FDPart3tmp16 * FDPart3tmp194 + FDPart3tmp16 * FDPart3tmp213 -
            FDPart3tmp178 * FDPart3tmp20 * trK_dD0 + FDPart3tmp185 * FDPart3tmp20 * FDPart3tmp54 + FDPart3tmp188 * FDPart3tmp58 +
            FDPart3tmp190 * FDPart3tmp50 + FDPart3tmp192 * FDPart3tmp46 + FDPart3tmp195 * FDPart3tmp48 + FDPart3tmp197 * FDPart3tmp20 * FDPart3tmp34 +
            2 * FDPart3tmp20 * FDPart3tmp215 * FDPart3tmp34 + FDPart3tmp202 * ((FDPart3tmp24) * (FDPart3tmp24)) + FDPart3tmp203 * FDPart3tmp58 +
            FDPart3tmp204 * FDPart3tmp30 + FDPart3tmp205 * FDPart3tmp50 + FDPart3tmp206 * FDPart3tmp34 + FDPart3tmp207 * FDPart3tmp46 +
            FDPart3tmp207 * FDPart3tmp73 + FDPart3tmp210 * FDPart3tmp31 + FDPart3tmp210 * FDPart3tmp71 + FDPart3tmp212 * FDPart3tmp31 +
            FDPart3tmp212 * FDPart3tmp71 + FDPart3tmp213 * FDPart3tmp75 + 2 * FDPart3tmp214 * FDPart3tmp31 + 2 * FDPart3tmp217 * FDPart3tmp35;
        const REAL FDPart3tmp222 =
            -2.0 / 3.0 * FDPart3tmp108 * trK_dD1 - FDPart3tmp122 * FDPart3tmp183 -
            FDPart3tmp126 * (6 * FDPart3tmp1 * FDPart3tmp44 + 6 * FDPart3tmp16 * FDPart3tmp26 + FDPart3tmp181 * FDPart3tmp49 +
                             FDPart3tmp182 * FDPart3tmp48 + 6 * FDPart3tmp42 + 12 * FDPart3tmp46 * FDPart3tmp64) -
            FDPart3tmp129 * FDPart3tmp220 - FDPart3tmp15 * FDPart3tmp178 * trK_dD2 + FDPart3tmp15 * FDPart3tmp202 * FDPart3tmp24 +
            FDPart3tmp16 * FDPart3tmp200 + FDPart3tmp16 * FDPart3tmp217 - FDPart3tmp178 * FDPart3tmp40 * trK_dD0 +
            FDPart3tmp185 * FDPart3tmp40 * FDPart3tmp54 + FDPart3tmp188 * FDPart3tmp52 + FDPart3tmp190 * FDPart3tmp58 + FDPart3tmp192 * FDPart3tmp49 +
            FDPart3tmp194 * FDPart3tmp48 + FDPart3tmp195 * FDPart3tmp44 + FDPart3tmp197 * FDPart3tmp20 * FDPart3tmp30 + FDPart3tmp203 * FDPart3tmp52 +
            FDPart3tmp204 * FDPart3tmp47 + FDPart3tmp205 * FDPart3tmp58 + FDPart3tmp206 * FDPart3tmp30 + 2 * FDPart3tmp207 * FDPart3tmp49 +
            FDPart3tmp210 * FDPart3tmp46 + FDPart3tmp210 * FDPart3tmp73 + 2 * FDPart3tmp212 * FDPart3tmp46 + 2 * FDPart3tmp213 * FDPart3tmp48 +
            FDPart3tmp214 * FDPart3tmp46 + FDPart3tmp214 * FDPart3tmp73 + FDPart3tmp215 * FDPart3tmp31 + FDPart3tmp215 * FDPart3tmp71 +
            FDPart3tmp217 * FDPart3tmp75;
        const REAL FDPart3tmp224 =
            -2.0 / 3.0 * FDPart3tmp110 * trK_dD0 + FDPart3tmp12 * FDPart3tmp15 * FDPart3tmp20 * FDPart3tmp200 +
            2 * FDPart3tmp12 * FDPart3tmp20 * FDPart3tmp206 + 2 * FDPart3tmp12 * FDPart3tmp204 * FDPart3tmp40 - FDPart3tmp122 * FDPart3tmp184 -
            FDPart3tmp126 * FDPart3tmp220 -
            FDPart3tmp129 * (6 * FDPart3tmp1 * FDPart3tmp52 + FDPart3tmp181 * FDPart3tmp61 + FDPart3tmp182 * FDPart3tmp58 +
                             6 * FDPart3tmp26 * FDPart3tmp50 + 6 * FDPart3tmp56 + 12 * FDPart3tmp60 * FDPart3tmp64) -
            FDPart3tmp178 * FDPart3tmp20 * trK_dD2 - FDPart3tmp178 * FDPart3tmp40 * trK_dD1 + FDPart3tmp185 * ((FDPart3tmp54) * (FDPart3tmp54)) +
            FDPart3tmp188 * FDPart3tmp61 + FDPart3tmp190 * FDPart3tmp60 + FDPart3tmp192 * FDPart3tmp52 + FDPart3tmp194 * FDPart3tmp46 +
            FDPart3tmp195 * FDPart3tmp49 + FDPart3tmp197 * FDPart3tmp50 + FDPart3tmp20 * FDPart3tmp202 * FDPart3tmp24 +
            FDPart3tmp207 * FDPart3tmp47 * FDPart3tmp54 + FDPart3tmp207 * FDPart3tmp52 + 2 * FDPart3tmp210 * FDPart3tmp58 +
            FDPart3tmp212 * FDPart3tmp58 + FDPart3tmp212 * FDPart3tmp83 + FDPart3tmp213 * FDPart3tmp46 + FDPart3tmp213 * FDPart3tmp73 +
            FDPart3tmp214 * FDPart3tmp58 + FDPart3tmp214 * FDPart3tmp83 + FDPart3tmp215 * FDPart3tmp34 * FDPart3tmp54 + FDPart3tmp215 * FDPart3tmp50 +
            FDPart3tmp217 * FDPart3tmp31 + FDPart3tmp217 * FDPart3tmp71;
        diagnostic_output_gfs[IDX4(HGF, i0, i1, i2)] =
            -FDPart3tmp1 * (FDPart3tmp1 * FDPart3tmp44 + FDPart3tmp16 * FDPart3tmp26 + FDPart3tmp29 * FDPart3tmp49 + FDPart3tmp33 * FDPart3tmp48 +
                            FDPart3tmp37 * FDPart3tmp46 + FDPart3tmp42) -
            FDPart3tmp26 * (FDPart3tmp1 * FDPart3tmp16 + FDPart3tmp20 * FDPart3tmp34 * FDPart3tmp37 + FDPart3tmp23 + FDPart3tmp27 +
                            FDPart3tmp29 * FDPart3tmp31 + FDPart3tmp33 * FDPart3tmp35) -
            FDPart3tmp29 * (FDPart3tmp1 * FDPart3tmp49 + FDPart3tmp26 * FDPart3tmp31 + FDPart3tmp46 * FDPart3tmp62 + FDPart3tmp52 * FDPart3tmp67 +
                            FDPart3tmp58 * FDPart3tmp64 + FDPart3tmp62 * FDPart3tmp73 + FDPart3tmp64 * FDPart3tmp83 + FDPart3tmp96 + FDPart3tmp98) -
            FDPart3tmp33 * (FDPart3tmp1 * FDPart3tmp48 + FDPart3tmp16 * FDPart3tmp62 + FDPart3tmp26 * FDPart3tmp35 + FDPart3tmp31 * FDPart3tmp64 +
                            FDPart3tmp46 * FDPart3tmp67 + FDPart3tmp62 * FDPart3tmp75 + FDPart3tmp64 * FDPart3tmp71 + FDPart3tmp67 * FDPart3tmp73 +
                            FDPart3tmp69) -
            FDPart3tmp37 * (FDPart3tmp1 * FDPart3tmp46 + FDPart3tmp31 * FDPart3tmp62 + FDPart3tmp50 * FDPart3tmp64 + FDPart3tmp58 * FDPart3tmp67 +
                            FDPart3tmp62 * FDPart3tmp71 + FDPart3tmp67 * FDPart3tmp83 + FDPart3tmp79 + FDPart3tmp86 + FDPart3tmp88) -
            aDD00 * (FDPart3tmp1 * FDPart3tmp52 + FDPart3tmp26 * FDPart3tmp50 + FDPart3tmp29 * FDPart3tmp61 + FDPart3tmp33 * FDPart3tmp58 +
                     FDPart3tmp37 * FDPart3tmp60 + FDPart3tmp56) +
            ((cf) * (cf)) *
                (2 * FDPart3tmp100 * FDPart3tmp15 * RbarDD12 + 2 * FDPart3tmp100 * FDPart3tmp20 * RbarDD02 + FDPart3tmp100 * FDPart3tmp24 * RbarDD22 +
                 2 * FDPart3tmp100 * FDPart3tmp40 * RbarDD01 + FDPart3tmp100 * FDPart3tmp43 * RbarDD11 + FDPart3tmp100 * FDPart3tmp54 * RbarDD00 -
                 FDPart3tmp102 * FDPart3tmp15 * cf_dD1 * cf_dD2 - FDPart3tmp102 * FDPart3tmp20 * cf_dD0 * cf_dD2 -
                 FDPart3tmp102 * FDPart3tmp40 * cf_dD0 * cf_dD1 - FDPart3tmp105 * FDPart3tmp106 * ((cf_dD2) * (cf_dD2)) -
                 8 * FDPart3tmp105 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * ((cf_dD2) * (cf_dD2)) - cf_dDD22) - FDPart3tmp122 * FDPart3tmp174 -
                      FDPart3tmp126 * FDPart3tmp175 - FDPart3tmp129 * FDPart3tmp176) -
                 FDPart3tmp106 * FDPart3tmp108 * ((cf_dD1) * (cf_dD1)) - FDPart3tmp106 * FDPart3tmp110 * ((cf_dD0) * (cf_dD0)) -
                 8 * FDPart3tmp108 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * ((cf_dD1) * (cf_dD1)) - cf_dDD11) - FDPart3tmp122 * FDPart3tmp153 -
                      FDPart3tmp126 * FDPart3tmp154 - FDPart3tmp129 * FDPart3tmp155) -
                 8 * FDPart3tmp110 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * ((cf_dD0) * (cf_dD0)) - cf_dDD00) - FDPart3tmp120 * FDPart3tmp122 -
                      FDPart3tmp125 * FDPart3tmp126 - FDPart3tmp128 * FDPart3tmp129) -
                 FDPart3tmp147 * FDPart3tmp15 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * cf_dD1 * cf_dD2 - cf_dDD12) - FDPart3tmp122 * FDPart3tmp167 -
                      FDPart3tmp126 * FDPart3tmp169 - FDPart3tmp129 * FDPart3tmp170) -
                 FDPart3tmp147 * FDPart3tmp20 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * cf_dD0 * cf_dD2 - cf_dDD02) - FDPart3tmp122 * FDPart3tmp159 -
                      FDPart3tmp126 * FDPart3tmp161 - FDPart3tmp129 * FDPart3tmp162) -
                 FDPart3tmp147 * FDPart3tmp40 *
                     ((0.5) * FDPart3tmp111 * (FDPart3tmp111 * cf_dD0 * cf_dD1 - cf_dDD01) - FDPart3tmp122 * FDPart3tmp143 -
                      FDPart3tmp126 * FDPart3tmp145 - FDPart3tmp129 * FDPart3tmp146)) +
            (2.0 / 3.0) * ((trK) * (trK));
        diagnostic_output_gfs[IDX4(MSQUAREDGF, i0, i1, i2)] = 2 * FDPart3tmp13 * FDPart3tmp177 * FDPart3tmp218 * FDPart3tmp222 * hDD12 +
                                                              FDPart3tmp177 * ((FDPart3tmp218) * (FDPart3tmp218)) * FDPart3tmp9 +
                                                              FDPart3tmp177 * FDPart3tmp218 * FDPart3tmp224 * FDPart3tmp28 * f1_of_xx1 * hDD02 +
                                                              FDPart3tmp177 * ((FDPart3tmp222) * (FDPart3tmp222)) * FDPart3tmp6 +
                                                              FDPart3tmp177 * FDPart3tmp222 * FDPart3tmp224 * FDPart3tmp28 * hDD01 +
                                                              FDPart3tmp177 * ((FDPart3tmp224) * (FDPart3tmp224)) * FDPart3tmp4;

      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    }   // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  }     // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}

void constraints_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                      const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, const REAL *restrict auxevol_gfs,
                                      REAL *restrict diagnostic_output_gfs) {
#include "../set_CodeParameters.h"
  int threads_in_x_dir = MIN(1024, params->Nxx0 / 32);
  int threads_in_y_dir = MIN(1024 / threads_in_x_dir, params->Nxx1);
  int threads_in_z_dir = 1;
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

  // Assumes the grids are small enough such that Nxx0 < 1024, therefore we only
  // need tiles to cover y and z
  dim3 grid_blocks(params->Nxx1 / threads_in_y_dir, params->Nxx2, 1);
  constraints_eval__rfm__Spherical_gpu<<<grid_blocks, block_threads>>>(rfmstruct->f0_of_xx0, rfmstruct->f1_of_xx1, 
    rfmstruct->f1_of_xx1__D1, rfmstruct->f1_of_xx1__DD11, in_gfs, auxevol_gfs, diagnostic_output_gfs);
}
