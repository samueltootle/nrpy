#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Finite difference function for operator dD0, with FD accuracy order 4.
 */
static __device__ REAL fd_function_dD0_fdorder4(const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1, const REAL FDPROTO_i0p2,
                                     const REAL invdxx0) {

  const REAL FD_result = invdxx0 * (FDPart1_Rational_1_12 * (FDPROTO_i0m2 - FDPROTO_i0p2) + FDPart1_Rational_2_3 * (-FDPROTO_i0m1 + FDPROTO_i0p1));

  return FD_result;
}
/*
 * Finite difference function for operator dD1, with FD accuracy order 4.
 */
__device__ static REAL fd_function_dD1_fdorder4(const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1, const REAL FDPROTO_i1p2,
                                     const REAL invdxx1) {

  const REAL FD_result = invdxx1 * (FDPart1_Rational_1_12 * (FDPROTO_i1m2 - FDPROTO_i1p2) + FDPart1_Rational_2_3 * (-FDPROTO_i1m1 + FDPROTO_i1p1));

  return FD_result;
}
/*
 * Finite difference function for operator dD2, with FD accuracy order 4.
 */
__device__ static REAL fd_function_dD2_fdorder4(const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1, const REAL FDPROTO_i2p2,
                                     const REAL invdxx2) {

  const REAL FD_result = invdxx2 * (FDPart1_Rational_1_12 * (FDPROTO_i2m2 - FDPROTO_i2p2) + FDPart1_Rational_2_3 * (-FDPROTO_i2m1 + FDPROTO_i2p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD00, with FD accuracy order 4.
 */
__device__ static REAL fd_function_dDD00_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1,
                                       const REAL FDPROTO_i0p2, const REAL invdxx0) {

  const REAL FD_result = ((invdxx0) * (invdxx0)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i0m2 - FDPROTO_i0p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i0m1 + FDPROTO_i0p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD01, with FD accuracy order 4.
 */
__device__ static REAL fd_function_dDD01_fdorder4(const REAL FDPROTO_i0m1_i1m1, const REAL FDPROTO_i0m1_i1m2, const REAL FDPROTO_i0m1_i1p1,
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
__device__ static REAL fd_function_dDD02_fdorder4(const REAL FDPROTO_i0m1_i2m1, const REAL FDPROTO_i0m1_i2m2, const REAL FDPROTO_i0m1_i2p1,
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
__device__ static REAL fd_function_dDD11_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1,
                                       const REAL FDPROTO_i1p2, const REAL invdxx1) {

  const REAL FD_result = ((invdxx1) * (invdxx1)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i1m2 - FDPROTO_i1p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i1m1 + FDPROTO_i1p1));

  return FD_result;
}
/*
 * Finite difference function for operator dDD12, with FD accuracy order 4.
 */
__device__ static REAL fd_function_dDD12_fdorder4(const REAL FDPROTO_i1m1_i2m1, const REAL FDPROTO_i1m1_i2m2, const REAL FDPROTO_i1m1_i2p1,
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
__device__ static REAL fd_function_dDD22_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1,
                                       const REAL FDPROTO_i2p2, const REAL invdxx2) {

  const REAL FD_result = ((invdxx2) * (invdxx2)) * (-FDPROTO * FDPart1_Rational_5_2 + FDPart1_Rational_1_12 * (-FDPROTO_i2m2 - FDPROTO_i2p2) +
                                                    FDPart1_Rational_4_3 * (FDPROTO_i2m1 + FDPROTO_i2p1));

  return FD_result;
}
/*
 * Set Ricci tensor.
 */
__global__
void Ricci_eval__rfm__Spherical_gpu(const REAL *restrict _f0_of_xx0, const REAL *restrict _f1_of_xx1, 
  const REAL *restrict _f1_of_xx1__D1, const REAL *restrict _f1_of_xx1__DD11, const REAL *restrict in_gfs, REAL *restrict auxevol_gfs) {
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
      const REAL f1_of_xx1__DD11 = _f1_of_xx1__DD11[i1];

      for(size_t i0 = tid0+NGHOSTS; i0 < Nxx0+NGHOSTS; i0 += stride0) {
        const REAL f0_of_xx0 = _f0_of_xx0[i0];
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL hDD00_i1m2_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD00_i1m1_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD00_i0m2_i2m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD00_i0m1_i2m2 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD00_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 2)];
        const REAL hDD00_i0p1_i2m2 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD00_i0p2_i2m2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD00_i1p1_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD00_i1p2_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD00_i1m2_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD00_i1m1_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD00_i0m2_i2m1 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD00_i0m1_i2m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD00_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 1)];
        const REAL hDD00_i0p1_i2m1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD00_i0p2_i2m1 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD00_i1p1_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD00_i1p2_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD00_i0m2_i1m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD00_i0m1_i1m2 = in_gfs[IDX4(HDD00GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD00_i1m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2)];
        const REAL hDD00_i0p1_i1m2 = in_gfs[IDX4(HDD00GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD00_i0p2_i1m2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD00_i0m2_i1m1 = in_gfs[IDX4(HDD00GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD00_i0m1_i1m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD00_i1m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2)];
        const REAL hDD00_i0p1_i1m1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD00_i0p2_i1m1 = in_gfs[IDX4(HDD00GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD00_i0m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2)];
        const REAL hDD00_i0m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2)];
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD00_i0p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2)];
        const REAL hDD00_i0p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2)];
        const REAL hDD00_i0m2_i1p1 = in_gfs[IDX4(HDD00GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD00_i0m1_i1p1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD00_i1p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2)];
        const REAL hDD00_i0p1_i1p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD00_i0p2_i1p1 = in_gfs[IDX4(HDD00GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD00_i0m2_i1p2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD00_i0m1_i1p2 = in_gfs[IDX4(HDD00GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD00_i1p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2)];
        const REAL hDD00_i0p1_i1p2 = in_gfs[IDX4(HDD00GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD00_i0p2_i1p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD00_i1m2_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD00_i1m1_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD00_i0m2_i2p1 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD00_i0m1_i2p1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD00_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 1)];
        const REAL hDD00_i0p1_i2p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD00_i0p2_i2p1 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD00_i1p1_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD00_i1p2_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD00_i1m2_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD00_i1m1_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD00_i0m2_i2p2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD00_i0m1_i2p2 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD00_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 2)];
        const REAL hDD00_i0p1_i2p2 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD00_i0p2_i2p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD00_i1p1_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD00_i1p2_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2 + 2)];
        const REAL hDD01_i1m2_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD01_i1m1_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD01_i0m2_i2m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD01_i0m1_i2m2 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD01_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 2)];
        const REAL hDD01_i0p1_i2m2 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD01_i0p2_i2m2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD01_i1p1_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD01_i1p2_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD01_i1m2_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD01_i1m1_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD01_i0m2_i2m1 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD01_i0m1_i2m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD01_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 1)];
        const REAL hDD01_i0p1_i2m1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD01_i0p2_i2m1 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD01_i1p1_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD01_i1p2_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD01_i0m2_i1m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD01_i0m1_i1m2 = in_gfs[IDX4(HDD01GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD01_i1m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2)];
        const REAL hDD01_i0p1_i1m2 = in_gfs[IDX4(HDD01GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD01_i0p2_i1m2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD01_i0m2_i1m1 = in_gfs[IDX4(HDD01GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD01_i0m1_i1m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD01_i1m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2)];
        const REAL hDD01_i0p1_i1m1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD01_i0p2_i1m1 = in_gfs[IDX4(HDD01GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD01_i0m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2)];
        const REAL hDD01_i0m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD01_i0p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2)];
        const REAL hDD01_i0p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2)];
        const REAL hDD01_i0m2_i1p1 = in_gfs[IDX4(HDD01GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD01_i0m1_i1p1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD01_i1p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2)];
        const REAL hDD01_i0p1_i1p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD01_i0p2_i1p1 = in_gfs[IDX4(HDD01GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD01_i0m2_i1p2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD01_i0m1_i1p2 = in_gfs[IDX4(HDD01GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD01_i1p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2)];
        const REAL hDD01_i0p1_i1p2 = in_gfs[IDX4(HDD01GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD01_i0p2_i1p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD01_i1m2_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD01_i1m1_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD01_i0m2_i2p1 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD01_i0m1_i2p1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD01_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 1)];
        const REAL hDD01_i0p1_i2p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD01_i0p2_i2p1 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD01_i1p1_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD01_i1p2_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD01_i1m2_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD01_i1m1_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD01_i0m2_i2p2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD01_i0m1_i2p2 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD01_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 2)];
        const REAL hDD01_i0p1_i2p2 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD01_i0p2_i2p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD01_i1p1_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD01_i1p2_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2 + 2)];
        const REAL hDD02_i1m2_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD02_i1m1_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD02_i0m2_i2m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD02_i0m1_i2m2 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD02_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 2)];
        const REAL hDD02_i0p1_i2m2 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD02_i0p2_i2m2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD02_i1p1_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD02_i1p2_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD02_i1m2_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD02_i1m1_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD02_i0m2_i2m1 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD02_i0m1_i2m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD02_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 1)];
        const REAL hDD02_i0p1_i2m1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD02_i0p2_i2m1 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD02_i1p1_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD02_i1p2_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD02_i0m2_i1m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD02_i0m1_i1m2 = in_gfs[IDX4(HDD02GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD02_i1m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2)];
        const REAL hDD02_i0p1_i1m2 = in_gfs[IDX4(HDD02GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD02_i0p2_i1m2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD02_i0m2_i1m1 = in_gfs[IDX4(HDD02GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD02_i0m1_i1m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD02_i1m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2)];
        const REAL hDD02_i0p1_i1m1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD02_i0p2_i1m1 = in_gfs[IDX4(HDD02GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD02_i0m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2)];
        const REAL hDD02_i0m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD02_i0p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2)];
        const REAL hDD02_i0p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2)];
        const REAL hDD02_i0m2_i1p1 = in_gfs[IDX4(HDD02GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD02_i0m1_i1p1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD02_i1p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2)];
        const REAL hDD02_i0p1_i1p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD02_i0p2_i1p1 = in_gfs[IDX4(HDD02GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD02_i0m2_i1p2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD02_i0m1_i1p2 = in_gfs[IDX4(HDD02GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD02_i1p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2)];
        const REAL hDD02_i0p1_i1p2 = in_gfs[IDX4(HDD02GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD02_i0p2_i1p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD02_i1m2_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD02_i1m1_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD02_i0m2_i2p1 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD02_i0m1_i2p1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD02_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 1)];
        const REAL hDD02_i0p1_i2p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD02_i0p2_i2p1 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD02_i1p1_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD02_i1p2_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD02_i1m2_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD02_i1m1_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD02_i0m2_i2p2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD02_i0m1_i2p2 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD02_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 2)];
        const REAL hDD02_i0p1_i2p2 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD02_i0p2_i2p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD02_i1p1_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD02_i1p2_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2 + 2)];
        const REAL hDD11_i1m2_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD11_i1m1_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD11_i0m2_i2m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD11_i0m1_i2m2 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD11_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 2)];
        const REAL hDD11_i0p1_i2m2 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD11_i0p2_i2m2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD11_i1p1_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD11_i1p2_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD11_i1m2_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD11_i1m1_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD11_i0m2_i2m1 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD11_i0m1_i2m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD11_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 1)];
        const REAL hDD11_i0p1_i2m1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD11_i0p2_i2m1 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD11_i1p1_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD11_i1p2_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD11_i0m2_i1m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD11_i0m1_i1m2 = in_gfs[IDX4(HDD11GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD11_i1m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2)];
        const REAL hDD11_i0p1_i1m2 = in_gfs[IDX4(HDD11GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD11_i0p2_i1m2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD11_i0m2_i1m1 = in_gfs[IDX4(HDD11GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD11_i0m1_i1m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD11_i1m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2)];
        const REAL hDD11_i0p1_i1m1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD11_i0p2_i1m1 = in_gfs[IDX4(HDD11GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD11_i0m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2)];
        const REAL hDD11_i0m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD11_i0p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2)];
        const REAL hDD11_i0p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2)];
        const REAL hDD11_i0m2_i1p1 = in_gfs[IDX4(HDD11GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD11_i0m1_i1p1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD11_i1p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2)];
        const REAL hDD11_i0p1_i1p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD11_i0p2_i1p1 = in_gfs[IDX4(HDD11GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD11_i0m2_i1p2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD11_i0m1_i1p2 = in_gfs[IDX4(HDD11GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD11_i1p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2)];
        const REAL hDD11_i0p1_i1p2 = in_gfs[IDX4(HDD11GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD11_i0p2_i1p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD11_i1m2_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD11_i1m1_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD11_i0m2_i2p1 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD11_i0m1_i2p1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD11_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 1)];
        const REAL hDD11_i0p1_i2p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD11_i0p2_i2p1 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD11_i1p1_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD11_i1p2_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD11_i1m2_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD11_i1m1_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD11_i0m2_i2p2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD11_i0m1_i2p2 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD11_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 2)];
        const REAL hDD11_i0p1_i2p2 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD11_i0p2_i2p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD11_i1p1_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD11_i1p2_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2 + 2)];
        const REAL hDD12_i1m2_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD12_i1m1_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD12_i0m2_i2m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD12_i0m1_i2m2 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD12_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 2)];
        const REAL hDD12_i0p1_i2m2 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD12_i0p2_i2m2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD12_i1p1_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD12_i1p2_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD12_i1m2_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD12_i1m1_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD12_i0m2_i2m1 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD12_i0m1_i2m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD12_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 1)];
        const REAL hDD12_i0p1_i2m1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD12_i0p2_i2m1 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD12_i1p1_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD12_i1p2_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD12_i0m2_i1m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD12_i0m1_i1m2 = in_gfs[IDX4(HDD12GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD12_i1m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2)];
        const REAL hDD12_i0p1_i1m2 = in_gfs[IDX4(HDD12GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD12_i0p2_i1m2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD12_i0m2_i1m1 = in_gfs[IDX4(HDD12GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD12_i0m1_i1m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD12_i1m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2)];
        const REAL hDD12_i0p1_i1m1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD12_i0p2_i1m1 = in_gfs[IDX4(HDD12GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD12_i0m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2)];
        const REAL hDD12_i0m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD12_i0p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2)];
        const REAL hDD12_i0p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2)];
        const REAL hDD12_i0m2_i1p1 = in_gfs[IDX4(HDD12GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD12_i0m1_i1p1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD12_i1p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2)];
        const REAL hDD12_i0p1_i1p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD12_i0p2_i1p1 = in_gfs[IDX4(HDD12GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD12_i0m2_i1p2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD12_i0m1_i1p2 = in_gfs[IDX4(HDD12GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD12_i1p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2)];
        const REAL hDD12_i0p1_i1p2 = in_gfs[IDX4(HDD12GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD12_i0p2_i1p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD12_i1m2_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD12_i1m1_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD12_i0m2_i2p1 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD12_i0m1_i2p1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD12_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 1)];
        const REAL hDD12_i0p1_i2p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD12_i0p2_i2p1 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD12_i1p1_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD12_i1p2_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD12_i1m2_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD12_i1m1_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD12_i0m2_i2p2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD12_i0m1_i2p2 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD12_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 2)];
        const REAL hDD12_i0p1_i2p2 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD12_i0p2_i2p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD12_i1p1_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD12_i1p2_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2 + 2)];
        const REAL hDD22_i1m2_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2 - 2)];
        const REAL hDD22_i1m1_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2 - 2)];
        const REAL hDD22_i0m2_i2m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2 - 2)];
        const REAL hDD22_i0m1_i2m2 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2 - 2)];
        const REAL hDD22_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 2)];
        const REAL hDD22_i0p1_i2m2 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2 - 2)];
        const REAL hDD22_i0p2_i2m2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2 - 2)];
        const REAL hDD22_i1p1_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2 - 2)];
        const REAL hDD22_i1p2_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2 - 2)];
        const REAL hDD22_i1m2_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2 - 1)];
        const REAL hDD22_i1m1_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2 - 1)];
        const REAL hDD22_i0m2_i2m1 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2 - 1)];
        const REAL hDD22_i0m1_i2m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2 - 1)];
        const REAL hDD22_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 1)];
        const REAL hDD22_i0p1_i2m1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2 - 1)];
        const REAL hDD22_i0p2_i2m1 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2 - 1)];
        const REAL hDD22_i1p1_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2 - 1)];
        const REAL hDD22_i1p2_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2 - 1)];
        const REAL hDD22_i0m2_i1m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1 - 2, i2)];
        const REAL hDD22_i0m1_i1m2 = in_gfs[IDX4(HDD22GF, i0 - 1, i1 - 2, i2)];
        const REAL hDD22_i1m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2)];
        const REAL hDD22_i0p1_i1m2 = in_gfs[IDX4(HDD22GF, i0 + 1, i1 - 2, i2)];
        const REAL hDD22_i0p2_i1m2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1 - 2, i2)];
        const REAL hDD22_i0m2_i1m1 = in_gfs[IDX4(HDD22GF, i0 - 2, i1 - 1, i2)];
        const REAL hDD22_i0m1_i1m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1 - 1, i2)];
        const REAL hDD22_i1m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2)];
        const REAL hDD22_i0p1_i1m1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1 - 1, i2)];
        const REAL hDD22_i0p2_i1m1 = in_gfs[IDX4(HDD22GF, i0 + 2, i1 - 1, i2)];
        const REAL hDD22_i0m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2)];
        const REAL hDD22_i0m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];
        const REAL hDD22_i0p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2)];
        const REAL hDD22_i0p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2)];
        const REAL hDD22_i0m2_i1p1 = in_gfs[IDX4(HDD22GF, i0 - 2, i1 + 1, i2)];
        const REAL hDD22_i0m1_i1p1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1 + 1, i2)];
        const REAL hDD22_i1p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2)];
        const REAL hDD22_i0p1_i1p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1 + 1, i2)];
        const REAL hDD22_i0p2_i1p1 = in_gfs[IDX4(HDD22GF, i0 + 2, i1 + 1, i2)];
        const REAL hDD22_i0m2_i1p2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1 + 2, i2)];
        const REAL hDD22_i0m1_i1p2 = in_gfs[IDX4(HDD22GF, i0 - 1, i1 + 2, i2)];
        const REAL hDD22_i1p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2)];
        const REAL hDD22_i0p1_i1p2 = in_gfs[IDX4(HDD22GF, i0 + 1, i1 + 2, i2)];
        const REAL hDD22_i0p2_i1p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1 + 2, i2)];
        const REAL hDD22_i1m2_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2 + 1)];
        const REAL hDD22_i1m1_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2 + 1)];
        const REAL hDD22_i0m2_i2p1 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2 + 1)];
        const REAL hDD22_i0m1_i2p1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2 + 1)];
        const REAL hDD22_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 1)];
        const REAL hDD22_i0p1_i2p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2 + 1)];
        const REAL hDD22_i0p2_i2p1 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2 + 1)];
        const REAL hDD22_i1p1_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2 + 1)];
        const REAL hDD22_i1p2_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2 + 1)];
        const REAL hDD22_i1m2_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2 + 2)];
        const REAL hDD22_i1m1_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2 + 2)];
        const REAL hDD22_i0m2_i2p2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2 + 2)];
        const REAL hDD22_i0m1_i2p2 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2 + 2)];
        const REAL hDD22_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 2)];
        const REAL hDD22_i0p1_i2p2 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2 + 2)];
        const REAL hDD22_i0p2_i2p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2 + 2)];
        const REAL hDD22_i1p1_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2 + 2)];
        const REAL hDD22_i1p2_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2 + 2)];
        const REAL lambdaU0_i2m2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 - 2)];
        const REAL lambdaU0_i2m1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 - 1)];
        const REAL lambdaU0_i1m2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 - 2, i2)];
        const REAL lambdaU0_i1m1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 - 1, i2)];
        const REAL lambdaU0_i0m2 = in_gfs[IDX4(LAMBDAU0GF, i0 - 2, i1, i2)];
        const REAL lambdaU0_i0m1 = in_gfs[IDX4(LAMBDAU0GF, i0 - 1, i1, i2)];
        const REAL lambdaU0 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2)];
        const REAL lambdaU0_i0p1 = in_gfs[IDX4(LAMBDAU0GF, i0 + 1, i1, i2)];
        const REAL lambdaU0_i0p2 = in_gfs[IDX4(LAMBDAU0GF, i0 + 2, i1, i2)];
        const REAL lambdaU0_i1p1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 + 1, i2)];
        const REAL lambdaU0_i1p2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 + 2, i2)];
        const REAL lambdaU0_i2p1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 + 1)];
        const REAL lambdaU0_i2p2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 + 2)];
        const REAL lambdaU1_i2m2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 - 2)];
        const REAL lambdaU1_i2m1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 - 1)];
        const REAL lambdaU1_i1m2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 - 2, i2)];
        const REAL lambdaU1_i1m1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 - 1, i2)];
        const REAL lambdaU1_i0m2 = in_gfs[IDX4(LAMBDAU1GF, i0 - 2, i1, i2)];
        const REAL lambdaU1_i0m1 = in_gfs[IDX4(LAMBDAU1GF, i0 - 1, i1, i2)];
        const REAL lambdaU1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2)];
        const REAL lambdaU1_i0p1 = in_gfs[IDX4(LAMBDAU1GF, i0 + 1, i1, i2)];
        const REAL lambdaU1_i0p2 = in_gfs[IDX4(LAMBDAU1GF, i0 + 2, i1, i2)];
        const REAL lambdaU1_i1p1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 + 1, i2)];
        const REAL lambdaU1_i1p2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 + 2, i2)];
        const REAL lambdaU1_i2p1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 + 1)];
        const REAL lambdaU1_i2p2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 + 2)];
        const REAL lambdaU2_i2m2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 - 2)];
        const REAL lambdaU2_i2m1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 - 1)];
        const REAL lambdaU2_i1m2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 - 2, i2)];
        const REAL lambdaU2_i1m1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 - 1, i2)];
        const REAL lambdaU2_i0m2 = in_gfs[IDX4(LAMBDAU2GF, i0 - 2, i1, i2)];
        const REAL lambdaU2_i0m1 = in_gfs[IDX4(LAMBDAU2GF, i0 - 1, i1, i2)];
        const REAL lambdaU2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2)];
        const REAL lambdaU2_i0p1 = in_gfs[IDX4(LAMBDAU2GF, i0 + 1, i1, i2)];
        const REAL lambdaU2_i0p2 = in_gfs[IDX4(LAMBDAU2GF, i0 + 2, i1, i2)];
        const REAL lambdaU2_i1p1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 + 1, i2)];
        const REAL lambdaU2_i1p2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 + 2, i2)];
        const REAL lambdaU2_i2p1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 + 1)];
        const REAL lambdaU2_i2p2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 + 2)];
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
        const REAL hDD_dDD0000 = fd_function_dDD00_fdorder4(hDD00, hDD00_i0m1, hDD00_i0m2, hDD00_i0p1, hDD00_i0p2, invdxx0);
        const REAL hDD_dDD0001 =
            fd_function_dDD01_fdorder4(hDD00_i0m1_i1m1, hDD00_i0m1_i1m2, hDD00_i0m1_i1p1, hDD00_i0m1_i1p2, hDD00_i0m2_i1m1, hDD00_i0m2_i1m2,
                                       hDD00_i0m2_i1p1, hDD00_i0m2_i1p2, hDD00_i0p1_i1m1, hDD00_i0p1_i1m2, hDD00_i0p1_i1p1, hDD00_i0p1_i1p2,
                                       hDD00_i0p2_i1m1, hDD00_i0p2_i1m2, hDD00_i0p2_i1p1, hDD00_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD0002 =
            fd_function_dDD02_fdorder4(hDD00_i0m1_i2m1, hDD00_i0m1_i2m2, hDD00_i0m1_i2p1, hDD00_i0m1_i2p2, hDD00_i0m2_i2m1, hDD00_i0m2_i2m2,
                                       hDD00_i0m2_i2p1, hDD00_i0m2_i2p2, hDD00_i0p1_i2m1, hDD00_i0p1_i2m2, hDD00_i0p1_i2p1, hDD00_i0p1_i2p2,
                                       hDD00_i0p2_i2m1, hDD00_i0p2_i2m2, hDD00_i0p2_i2p1, hDD00_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD0011 = fd_function_dDD11_fdorder4(hDD00, hDD00_i1m1, hDD00_i1m2, hDD00_i1p1, hDD00_i1p2, invdxx1);
        const REAL hDD_dDD0012 =
            fd_function_dDD12_fdorder4(hDD00_i1m1_i2m1, hDD00_i1m1_i2m2, hDD00_i1m1_i2p1, hDD00_i1m1_i2p2, hDD00_i1m2_i2m1, hDD00_i1m2_i2m2,
                                       hDD00_i1m2_i2p1, hDD00_i1m2_i2p2, hDD00_i1p1_i2m1, hDD00_i1p1_i2m2, hDD00_i1p1_i2p1, hDD00_i1p1_i2p2,
                                       hDD00_i1p2_i2m1, hDD00_i1p2_i2m2, hDD00_i1p2_i2p1, hDD00_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD0022 = fd_function_dDD22_fdorder4(hDD00, hDD00_i2m1, hDD00_i2m2, hDD00_i2p1, hDD00_i2p2, invdxx2);
        const REAL hDD_dDD0100 = fd_function_dDD00_fdorder4(hDD01, hDD01_i0m1, hDD01_i0m2, hDD01_i0p1, hDD01_i0p2, invdxx0);
        const REAL hDD_dDD0101 =
            fd_function_dDD01_fdorder4(hDD01_i0m1_i1m1, hDD01_i0m1_i1m2, hDD01_i0m1_i1p1, hDD01_i0m1_i1p2, hDD01_i0m2_i1m1, hDD01_i0m2_i1m2,
                                       hDD01_i0m2_i1p1, hDD01_i0m2_i1p2, hDD01_i0p1_i1m1, hDD01_i0p1_i1m2, hDD01_i0p1_i1p1, hDD01_i0p1_i1p2,
                                       hDD01_i0p2_i1m1, hDD01_i0p2_i1m2, hDD01_i0p2_i1p1, hDD01_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD0102 =
            fd_function_dDD02_fdorder4(hDD01_i0m1_i2m1, hDD01_i0m1_i2m2, hDD01_i0m1_i2p1, hDD01_i0m1_i2p2, hDD01_i0m2_i2m1, hDD01_i0m2_i2m2,
                                       hDD01_i0m2_i2p1, hDD01_i0m2_i2p2, hDD01_i0p1_i2m1, hDD01_i0p1_i2m2, hDD01_i0p1_i2p1, hDD01_i0p1_i2p2,
                                       hDD01_i0p2_i2m1, hDD01_i0p2_i2m2, hDD01_i0p2_i2p1, hDD01_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD0111 = fd_function_dDD11_fdorder4(hDD01, hDD01_i1m1, hDD01_i1m2, hDD01_i1p1, hDD01_i1p2, invdxx1);
        const REAL hDD_dDD0112 =
            fd_function_dDD12_fdorder4(hDD01_i1m1_i2m1, hDD01_i1m1_i2m2, hDD01_i1m1_i2p1, hDD01_i1m1_i2p2, hDD01_i1m2_i2m1, hDD01_i1m2_i2m2,
                                       hDD01_i1m2_i2p1, hDD01_i1m2_i2p2, hDD01_i1p1_i2m1, hDD01_i1p1_i2m2, hDD01_i1p1_i2p1, hDD01_i1p1_i2p2,
                                       hDD01_i1p2_i2m1, hDD01_i1p2_i2m2, hDD01_i1p2_i2p1, hDD01_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD0122 = fd_function_dDD22_fdorder4(hDD01, hDD01_i2m1, hDD01_i2m2, hDD01_i2p1, hDD01_i2p2, invdxx2);
        const REAL hDD_dDD0200 = fd_function_dDD00_fdorder4(hDD02, hDD02_i0m1, hDD02_i0m2, hDD02_i0p1, hDD02_i0p2, invdxx0);
        const REAL hDD_dDD0201 =
            fd_function_dDD01_fdorder4(hDD02_i0m1_i1m1, hDD02_i0m1_i1m2, hDD02_i0m1_i1p1, hDD02_i0m1_i1p2, hDD02_i0m2_i1m1, hDD02_i0m2_i1m2,
                                       hDD02_i0m2_i1p1, hDD02_i0m2_i1p2, hDD02_i0p1_i1m1, hDD02_i0p1_i1m2, hDD02_i0p1_i1p1, hDD02_i0p1_i1p2,
                                       hDD02_i0p2_i1m1, hDD02_i0p2_i1m2, hDD02_i0p2_i1p1, hDD02_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD0202 =
            fd_function_dDD02_fdorder4(hDD02_i0m1_i2m1, hDD02_i0m1_i2m2, hDD02_i0m1_i2p1, hDD02_i0m1_i2p2, hDD02_i0m2_i2m1, hDD02_i0m2_i2m2,
                                       hDD02_i0m2_i2p1, hDD02_i0m2_i2p2, hDD02_i0p1_i2m1, hDD02_i0p1_i2m2, hDD02_i0p1_i2p1, hDD02_i0p1_i2p2,
                                       hDD02_i0p2_i2m1, hDD02_i0p2_i2m2, hDD02_i0p2_i2p1, hDD02_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD0211 = fd_function_dDD11_fdorder4(hDD02, hDD02_i1m1, hDD02_i1m2, hDD02_i1p1, hDD02_i1p2, invdxx1);
        const REAL hDD_dDD0212 =
            fd_function_dDD12_fdorder4(hDD02_i1m1_i2m1, hDD02_i1m1_i2m2, hDD02_i1m1_i2p1, hDD02_i1m1_i2p2, hDD02_i1m2_i2m1, hDD02_i1m2_i2m2,
                                       hDD02_i1m2_i2p1, hDD02_i1m2_i2p2, hDD02_i1p1_i2m1, hDD02_i1p1_i2m2, hDD02_i1p1_i2p1, hDD02_i1p1_i2p2,
                                       hDD02_i1p2_i2m1, hDD02_i1p2_i2m2, hDD02_i1p2_i2p1, hDD02_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD0222 = fd_function_dDD22_fdorder4(hDD02, hDD02_i2m1, hDD02_i2m2, hDD02_i2p1, hDD02_i2p2, invdxx2);
        const REAL hDD_dDD1100 = fd_function_dDD00_fdorder4(hDD11, hDD11_i0m1, hDD11_i0m2, hDD11_i0p1, hDD11_i0p2, invdxx0);
        const REAL hDD_dDD1101 =
            fd_function_dDD01_fdorder4(hDD11_i0m1_i1m1, hDD11_i0m1_i1m2, hDD11_i0m1_i1p1, hDD11_i0m1_i1p2, hDD11_i0m2_i1m1, hDD11_i0m2_i1m2,
                                       hDD11_i0m2_i1p1, hDD11_i0m2_i1p2, hDD11_i0p1_i1m1, hDD11_i0p1_i1m2, hDD11_i0p1_i1p1, hDD11_i0p1_i1p2,
                                       hDD11_i0p2_i1m1, hDD11_i0p2_i1m2, hDD11_i0p2_i1p1, hDD11_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD1102 =
            fd_function_dDD02_fdorder4(hDD11_i0m1_i2m1, hDD11_i0m1_i2m2, hDD11_i0m1_i2p1, hDD11_i0m1_i2p2, hDD11_i0m2_i2m1, hDD11_i0m2_i2m2,
                                       hDD11_i0m2_i2p1, hDD11_i0m2_i2p2, hDD11_i0p1_i2m1, hDD11_i0p1_i2m2, hDD11_i0p1_i2p1, hDD11_i0p1_i2p2,
                                       hDD11_i0p2_i2m1, hDD11_i0p2_i2m2, hDD11_i0p2_i2p1, hDD11_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD1111 = fd_function_dDD11_fdorder4(hDD11, hDD11_i1m1, hDD11_i1m2, hDD11_i1p1, hDD11_i1p2, invdxx1);
        const REAL hDD_dDD1112 =
            fd_function_dDD12_fdorder4(hDD11_i1m1_i2m1, hDD11_i1m1_i2m2, hDD11_i1m1_i2p1, hDD11_i1m1_i2p2, hDD11_i1m2_i2m1, hDD11_i1m2_i2m2,
                                       hDD11_i1m2_i2p1, hDD11_i1m2_i2p2, hDD11_i1p1_i2m1, hDD11_i1p1_i2m2, hDD11_i1p1_i2p1, hDD11_i1p1_i2p2,
                                       hDD11_i1p2_i2m1, hDD11_i1p2_i2m2, hDD11_i1p2_i2p1, hDD11_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD1122 = fd_function_dDD22_fdorder4(hDD11, hDD11_i2m1, hDD11_i2m2, hDD11_i2p1, hDD11_i2p2, invdxx2);
        const REAL hDD_dDD1200 = fd_function_dDD00_fdorder4(hDD12, hDD12_i0m1, hDD12_i0m2, hDD12_i0p1, hDD12_i0p2, invdxx0);
        const REAL hDD_dDD1201 =
            fd_function_dDD01_fdorder4(hDD12_i0m1_i1m1, hDD12_i0m1_i1m2, hDD12_i0m1_i1p1, hDD12_i0m1_i1p2, hDD12_i0m2_i1m1, hDD12_i0m2_i1m2,
                                       hDD12_i0m2_i1p1, hDD12_i0m2_i1p2, hDD12_i0p1_i1m1, hDD12_i0p1_i1m2, hDD12_i0p1_i1p1, hDD12_i0p1_i1p2,
                                       hDD12_i0p2_i1m1, hDD12_i0p2_i1m2, hDD12_i0p2_i1p1, hDD12_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD1202 =
            fd_function_dDD02_fdorder4(hDD12_i0m1_i2m1, hDD12_i0m1_i2m2, hDD12_i0m1_i2p1, hDD12_i0m1_i2p2, hDD12_i0m2_i2m1, hDD12_i0m2_i2m2,
                                       hDD12_i0m2_i2p1, hDD12_i0m2_i2p2, hDD12_i0p1_i2m1, hDD12_i0p1_i2m2, hDD12_i0p1_i2p1, hDD12_i0p1_i2p2,
                                       hDD12_i0p2_i2m1, hDD12_i0p2_i2m2, hDD12_i0p2_i2p1, hDD12_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD1211 = fd_function_dDD11_fdorder4(hDD12, hDD12_i1m1, hDD12_i1m2, hDD12_i1p1, hDD12_i1p2, invdxx1);
        const REAL hDD_dDD1212 =
            fd_function_dDD12_fdorder4(hDD12_i1m1_i2m1, hDD12_i1m1_i2m2, hDD12_i1m1_i2p1, hDD12_i1m1_i2p2, hDD12_i1m2_i2m1, hDD12_i1m2_i2m2,
                                       hDD12_i1m2_i2p1, hDD12_i1m2_i2p2, hDD12_i1p1_i2m1, hDD12_i1p1_i2m2, hDD12_i1p1_i2p1, hDD12_i1p1_i2p2,
                                       hDD12_i1p2_i2m1, hDD12_i1p2_i2m2, hDD12_i1p2_i2p1, hDD12_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD1222 = fd_function_dDD22_fdorder4(hDD12, hDD12_i2m1, hDD12_i2m2, hDD12_i2p1, hDD12_i2p2, invdxx2);
        const REAL hDD_dDD2200 = fd_function_dDD00_fdorder4(hDD22, hDD22_i0m1, hDD22_i0m2, hDD22_i0p1, hDD22_i0p2, invdxx0);
        const REAL hDD_dDD2201 =
            fd_function_dDD01_fdorder4(hDD22_i0m1_i1m1, hDD22_i0m1_i1m2, hDD22_i0m1_i1p1, hDD22_i0m1_i1p2, hDD22_i0m2_i1m1, hDD22_i0m2_i1m2,
                                       hDD22_i0m2_i1p1, hDD22_i0m2_i1p2, hDD22_i0p1_i1m1, hDD22_i0p1_i1m2, hDD22_i0p1_i1p1, hDD22_i0p1_i1p2,
                                       hDD22_i0p2_i1m1, hDD22_i0p2_i1m2, hDD22_i0p2_i1p1, hDD22_i0p2_i1p2, invdxx0, invdxx1);
        const REAL hDD_dDD2202 =
            fd_function_dDD02_fdorder4(hDD22_i0m1_i2m1, hDD22_i0m1_i2m2, hDD22_i0m1_i2p1, hDD22_i0m1_i2p2, hDD22_i0m2_i2m1, hDD22_i0m2_i2m2,
                                       hDD22_i0m2_i2p1, hDD22_i0m2_i2p2, hDD22_i0p1_i2m1, hDD22_i0p1_i2m2, hDD22_i0p1_i2p1, hDD22_i0p1_i2p2,
                                       hDD22_i0p2_i2m1, hDD22_i0p2_i2m2, hDD22_i0p2_i2p1, hDD22_i0p2_i2p2, invdxx0, invdxx2);
        const REAL hDD_dDD2211 = fd_function_dDD11_fdorder4(hDD22, hDD22_i1m1, hDD22_i1m2, hDD22_i1p1, hDD22_i1p2, invdxx1);
        const REAL hDD_dDD2212 =
            fd_function_dDD12_fdorder4(hDD22_i1m1_i2m1, hDD22_i1m1_i2m2, hDD22_i1m1_i2p1, hDD22_i1m1_i2p2, hDD22_i1m2_i2m1, hDD22_i1m2_i2m2,
                                       hDD22_i1m2_i2p1, hDD22_i1m2_i2p2, hDD22_i1p1_i2m1, hDD22_i1p1_i2m2, hDD22_i1p1_i2p1, hDD22_i1p1_i2p2,
                                       hDD22_i1p2_i2m1, hDD22_i1p2_i2m2, hDD22_i1p2_i2p1, hDD22_i1p2_i2p2, invdxx1, invdxx2);
        const REAL hDD_dDD2222 = fd_function_dDD22_fdorder4(hDD22, hDD22_i2m1, hDD22_i2m2, hDD22_i2p1, hDD22_i2p2, invdxx2);
        const REAL lambdaU_dD00 = fd_function_dD0_fdorder4(lambdaU0_i0m1, lambdaU0_i0m2, lambdaU0_i0p1, lambdaU0_i0p2, invdxx0);
        const REAL lambdaU_dD01 = fd_function_dD1_fdorder4(lambdaU0_i1m1, lambdaU0_i1m2, lambdaU0_i1p1, lambdaU0_i1p2, invdxx1);
        const REAL lambdaU_dD02 = fd_function_dD2_fdorder4(lambdaU0_i2m1, lambdaU0_i2m2, lambdaU0_i2p1, lambdaU0_i2p2, invdxx2);
        const REAL lambdaU_dD10 = fd_function_dD0_fdorder4(lambdaU1_i0m1, lambdaU1_i0m2, lambdaU1_i0p1, lambdaU1_i0p2, invdxx0);
        const REAL lambdaU_dD11 = fd_function_dD1_fdorder4(lambdaU1_i1m1, lambdaU1_i1m2, lambdaU1_i1p1, lambdaU1_i1p2, invdxx1);
        const REAL lambdaU_dD12 = fd_function_dD2_fdorder4(lambdaU1_i2m1, lambdaU1_i2m2, lambdaU1_i2p1, lambdaU1_i2p2, invdxx2);
        const REAL lambdaU_dD20 = fd_function_dD0_fdorder4(lambdaU2_i0m1, lambdaU2_i0m2, lambdaU2_i0p1, lambdaU2_i0p2, invdxx0);
        const REAL lambdaU_dD21 = fd_function_dD1_fdorder4(lambdaU2_i1m1, lambdaU2_i1m2, lambdaU2_i1p1, lambdaU2_i1p2, invdxx1);
        const REAL lambdaU_dD22 = fd_function_dD2_fdorder4(lambdaU2_i2m1, lambdaU2_i2m2, lambdaU2_i2p1, lambdaU2_i2p2, invdxx2);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = hDD00 + 1;
        const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp2 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp4 = ((f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp15 = f1_of_xx1 * hDD_dD020;
        const REAL FDPart3tmp16 = (1.0 / (f0_of_xx0));
        const REAL FDPart3tmp17 = f1_of_xx1 * hDD02;
        const REAL FDPart3tmp22 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp23 = f1_of_xx1 * hDD12;
        const REAL FDPart3tmp28 = 2 * hDD01;
        const REAL FDPart3tmp32 = f0_of_xx0 * hDD01;
        const REAL FDPart3tmp43 = (1.0 / (f1_of_xx1));
        const REAL FDPart3tmp63 = f1_of_xx1__D1 * hDD02;
        const REAL FDPart3tmp67 = f0_of_xx0 * hDD_dD012;
        const REAL FDPart3tmp72 = f1_of_xx1 * hDD_dD022;
        const REAL FDPart3tmp73 = f1_of_xx1 * f1_of_xx1__D1;
        const REAL FDPart3tmp81 = 2 * f0_of_xx0;
        const REAL FDPart3tmp83 = (1.0F / 2.0F) * hDD_dD000;
        const REAL FDPart3tmp202 = -lambdaU1 + lambdaU_dD01;
        const REAL FDPart3tmp207 = f0_of_xx0 * f1_of_xx1;
        const REAL FDPart3tmp227 = ((f1_of_xx1__D1) * (f1_of_xx1__D1));
        const REAL FDPart3tmp280 = -f1_of_xx1 * lambdaU2 + lambdaU_dD02;
        const REAL FDPart3tmp288 = ((f1_of_xx1) * (f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp3 = FDPart3tmp1 * FDPart3tmp2 * ((hDD12) * (hDD12));
        const REAL FDPart3tmp5 = FDPart3tmp4 * hDD11;
        const REAL FDPart3tmp7 = FDPart3tmp2 * FDPart3tmp4;
        const REAL FDPart3tmp11 = FDPart3tmp4 * ((hDD01) * (hDD01));
        const REAL FDPart3tmp24 = FDPart3tmp17 * f0_of_xx0;
        const REAL FDPart3tmp29 = FDPart3tmp28 - hDD_dD001;
        const REAL FDPart3tmp36 = FDPart3tmp23 * f0_of_xx0;
        const REAL FDPart3tmp38 = f0_of_xx0 * f1_of_xx1 * hDD_dD021;
        const REAL FDPart3tmp44 = FDPart3tmp43 * f1_of_xx1__D1;
        const REAL FDPart3tmp47 = FDPart3tmp23 * FDPart3tmp4;
        const REAL FDPart3tmp50 = FDPart3tmp15 * f0_of_xx0 + FDPart3tmp17;
        const REAL FDPart3tmp55 = f0_of_xx0 * hDD00 - f0_of_xx0 * hDD11 + f0_of_xx0 * hDD_dD011;
        const REAL FDPart3tmp61 = f0_of_xx0 * hDD_dD010 + hDD01;
        const REAL FDPart3tmp64 = FDPart3tmp63 * f0_of_xx0;
        const REAL FDPart3tmp74 = FDPart3tmp2 * f0_of_xx0;
        const REAL FDPart3tmp95 = FDPart3tmp4 * f1_of_xx1;
        const REAL FDPart3tmp102 = FDPart3tmp2 * FDPart3tmp81;
        const REAL FDPart3tmp103 = FDPart3tmp2 * hDD22;
        const REAL FDPart3tmp139 = FDPart3tmp4 * hDD_dD111;
        const REAL FDPart3tmp142 = FDPart3tmp4 * hDD_dD112;
        const REAL FDPart3tmp153 = 2 * FDPart3tmp4;
        const REAL FDPart3tmp190 = FDPart3tmp17 * FDPart3tmp81;
        const REAL FDPart3tmp191 = FDPart3tmp28 * f0_of_xx0;
        const REAL FDPart3tmp199 = (1.0F / 2.0F) * FDPart3tmp32;
        const REAL FDPart3tmp210 = f0_of_xx0 * f1_of_xx1__D1 * hDD_dD020;
        const REAL FDPart3tmp228 = FDPart3tmp43 * f1_of_xx1__DD11 - FDPart3tmp227 / FDPart3tmp2;
        const REAL FDPart3tmp230 = FDPart3tmp4 * FDPart3tmp73;
        const REAL FDPart3tmp233 = f0_of_xx0 * f1_of_xx1__D1 * hDD_dD022;
        const REAL FDPart3tmp281 = -FDPart3tmp16 * f1_of_xx1__D1 * lambdaU2 + FDPart3tmp16 * lambdaU_dD12;
        const REAL FDPart3tmp289 = FDPart3tmp288 * FDPart3tmp4;
        const REAL FDPart3tmp297 = FDPart3tmp227 + f1_of_xx1 * f1_of_xx1__DD11;
        const REAL FDPart3tmp322 = FDPart3tmp4 * f1_of_xx1__D1;
        const REAL FDPart3tmp354 = FDPart3tmp81 * f1_of_xx1;
        const REAL FDPart3tmp379 = 2 * FDPart3tmp73;
        const REAL FDPart3tmp6 = FDPart3tmp4 + FDPart3tmp5;
        const REAL FDPart3tmp8 = FDPart3tmp7 * ((hDD02) * (hDD02));
        const REAL FDPart3tmp9 = FDPart3tmp7 * hDD22;
        const REAL FDPart3tmp19 = 2 * FDPart3tmp17 - hDD_dD002;
        const REAL FDPart3tmp65 = FDPart3tmp38 + FDPart3tmp64;
        const REAL FDPart3tmp69 = -FDPart3tmp36 - FDPart3tmp64 + FDPart3tmp67;
        const REAL FDPart3tmp75 = FDPart3tmp32 * FDPart3tmp73 + FDPart3tmp72 * f0_of_xx0 + FDPart3tmp74 * hDD00 - FDPart3tmp74 * hDD22;
        const REAL FDPart3tmp84 = (1.0F / 2.0F) * FDPart3tmp29 + (1.0F / 2.0F) * FDPart3tmp81 * hDD_dD010;
        const REAL FDPart3tmp92 = FDPart3tmp4 * hDD_dD110 + FDPart3tmp81 * hDD11;
        const REAL FDPart3tmp104 = FDPart3tmp103 * FDPart3tmp81 + FDPart3tmp7 * hDD_dD220;
        const REAL FDPart3tmp141 = FDPart3tmp95 * hDD_dD121;
        const REAL FDPart3tmp144 = FDPart3tmp4 * f1_of_xx1__D1 * hDD12;
        const REAL FDPart3tmp154 = FDPart3tmp153 * FDPart3tmp73;
        const REAL FDPart3tmp175 = FDPart3tmp7 * hDD_dD222;
        const REAL FDPart3tmp206 = FDPart3tmp16 * lambdaU0 + FDPart3tmp16 * lambdaU_dD11;
        const REAL FDPart3tmp214 = FDPart3tmp139 + FDPart3tmp28 * FDPart3tmp4;
        const REAL FDPart3tmp231 = -FDPart3tmp230 * hDD22 + FDPart3tmp5 * FDPart3tmp73 + FDPart3tmp7 * hDD01 + FDPart3tmp95 * hDD_dD122;
        const REAL FDPart3tmp279 = (1.0F / 2.0F) * FDPart3tmp24;
        const REAL FDPart3tmp284 = FDPart3tmp16 * FDPart3tmp43 * lambdaU_dD22 + FDPart3tmp16 * FDPart3tmp44 * lambdaU1 + FDPart3tmp16 * lambdaU0;
        const REAL FDPart3tmp286 = FDPart3tmp74 * hDD_dD221;
        const REAL FDPart3tmp292 = 2 * FDPart3tmp7 * f1_of_xx1__D1;
        const REAL FDPart3tmp335 = -FDPart3tmp228 * FDPart3tmp47;
        const REAL FDPart3tmp352 = FDPart3tmp153 * FDPart3tmp23;
        const REAL FDPart3tmp356 = FDPart3tmp354 * hDD_dD121 + FDPart3tmp81 * f1_of_xx1__D1 * hDD12 + FDPart3tmp95 * hDD_dDD1201;
        const REAL FDPart3tmp376 = FDPart3tmp7 * hDD_dDD2201 + 4 * FDPart3tmp73 * f0_of_xx0 * hDD22;
        const REAL FDPart3tmp10 = FDPart3tmp7 + FDPart3tmp9;
        const REAL FDPart3tmp31 = -FDPart3tmp16 * FDPart3tmp29;
        const REAL FDPart3tmp40 = -FDPart3tmp36 + FDPart3tmp38;
        const REAL FDPart3tmp57 = -2 * FDPart3tmp16 * FDPart3tmp55 + f0_of_xx0 * hDD_dD000;
        const REAL FDPart3tmp82 = (1.0F / 2.0F) * FDPart3tmp15 * FDPart3tmp81 + (1.0F / 2.0F) * FDPart3tmp19;
        const REAL FDPart3tmp93 = FDPart3tmp81 + FDPart3tmp92;
        const REAL FDPart3tmp97 = FDPart3tmp23 * FDPart3tmp81 + FDPart3tmp95 * hDD_dD120;
        const REAL FDPart3tmp105 = FDPart3tmp102 + FDPart3tmp104;
        const REAL FDPart3tmp145 = -FDPart3tmp142 + 2 * FDPart3tmp144;
        const REAL FDPart3tmp155 = FDPart3tmp154 * hDD22 + FDPart3tmp7 * hDD_dD221;
        const REAL FDPart3tmp157 = -FDPart3tmp23 * FDPart3tmp81 + FDPart3tmp65 + FDPart3tmp67 - FDPart3tmp95 * hDD_dD120;
        const REAL FDPart3tmp219 = FDPart3tmp44 * FDPart3tmp69 - f0_of_xx0 * hDD_dDD0112;
        const REAL FDPart3tmp220 = FDPart3tmp141 + FDPart3tmp17 * FDPart3tmp4;
        const REAL FDPart3tmp225 = FDPart3tmp141 + FDPart3tmp144;
        const REAL FDPart3tmp232 = FDPart3tmp16 * FDPart3tmp231;
        const REAL FDPart3tmp293 = FDPart3tmp175 + 2 * FDPart3tmp289 * hDD02 + FDPart3tmp292 * hDD12;
        const REAL FDPart3tmp363 = -FDPart3tmp228 * FDPart3tmp9;
        const REAL FDPart3tmp12 =
            (1.0 / (FDPart3tmp0 * FDPart3tmp10 * FDPart3tmp6 - FDPart3tmp0 * FDPart3tmp3 + 2 * FDPart3tmp1 * FDPart3tmp2 * hDD01 * hDD02 * hDD12 -
                    FDPart3tmp10 * FDPart3tmp11 - FDPart3tmp6 * FDPart3tmp8));
        const REAL FDPart3tmp21 = -FDPart3tmp16 * FDPart3tmp19;
        const REAL FDPart3tmp45 = -FDPart3tmp19 * FDPart3tmp44 - hDD_dDD0012;
        const REAL FDPart3tmp71 = 2 * FDPart3tmp16 * FDPart3tmp69;
        const REAL FDPart3tmp78 = -2 * FDPart3tmp16 * FDPart3tmp75 + FDPart3tmp74 * hDD_dD000;
        const REAL FDPart3tmp98 = FDPart3tmp65 - FDPart3tmp67 + FDPart3tmp97;
        const REAL FDPart3tmp106 = -FDPart3tmp38 - FDPart3tmp64 + FDPart3tmp67 + FDPart3tmp97;
        const REAL FDPart3tmp140 = -FDPart3tmp93 + 2 * f0_of_xx0 * hDD_dD011;
        const REAL FDPart3tmp146 = 2 * FDPart3tmp141 + FDPart3tmp145;
        const REAL FDPart3tmp156 = FDPart3tmp154 + FDPart3tmp155;
        const REAL FDPart3tmp177 = -FDPart3tmp105 + 2 * f0_of_xx0 * f1_of_xx1 * hDD_dD022;
        const REAL FDPart3tmp216 = -FDPart3tmp16 * FDPart3tmp214;
        const REAL FDPart3tmp221 = FDPart3tmp16 * FDPart3tmp220;
        const REAL FDPart3tmp235 = FDPart3tmp207 * hDD_dD122 + FDPart3tmp44 * FDPart3tmp75 - FDPart3tmp55 * f1_of_xx1 * f1_of_xx1__D1;
        const REAL FDPart3tmp283 = FDPart3tmp10 * FDPart3tmp16 * FDPart3tmp43;
        const REAL FDPart3tmp332 = -2 * FDPart3tmp231 * FDPart3tmp44;
        const REAL FDPart3tmp13 = FDPart3tmp12 * (FDPart3tmp10 * FDPart3tmp6 - FDPart3tmp3);
        const REAL FDPart3tmp25 = FDPart3tmp12 * (FDPart3tmp22 * FDPart3tmp23 * hDD01 - FDPart3tmp24 * FDPart3tmp6);
        const REAL FDPart3tmp33 = FDPart3tmp12 * (-FDPart3tmp10 * FDPart3tmp32 + FDPart3tmp2 * FDPart3tmp22 * hDD02 * hDD12);
        const REAL FDPart3tmp42 = 2 * FDPart3tmp16 * FDPart3tmp40;
        const REAL FDPart3tmp48 = FDPart3tmp12 * (-FDPart3tmp0 * FDPart3tmp47 + FDPart3tmp17 * FDPart3tmp4 * hDD01);
        const REAL FDPart3tmp58 = FDPart3tmp12 * (FDPart3tmp0 * FDPart3tmp10 - FDPart3tmp8);
        const REAL FDPart3tmp79 = FDPart3tmp12 * (FDPart3tmp0 * FDPart3tmp6 - FDPart3tmp11);
        const REAL FDPart3tmp176 = -FDPart3tmp156 + 2 * FDPart3tmp4 * f1_of_xx1 * hDD_dD122;
        const REAL FDPart3tmp218 = -FDPart3tmp16 * FDPart3tmp97 + FDPart3tmp23 - FDPart3tmp44 * FDPart3tmp50;
        const REAL FDPart3tmp222 = FDPart3tmp221 + FDPart3tmp40 * FDPart3tmp44;
        const REAL FDPart3tmp224 = -FDPart3tmp145 * FDPart3tmp16;
        const REAL FDPart3tmp229 = FDPart3tmp16 * FDPart3tmp225 + FDPart3tmp228 * FDPart3tmp24 + FDPart3tmp44 * FDPart3tmp65;
        const REAL FDPart3tmp295 = -FDPart3tmp16 * FDPart3tmp293;
        const REAL FDPart3tmp327 = FDPart3tmp145 * FDPart3tmp44 + FDPart3tmp4 * hDD_dDD1112;
        const REAL FDPart3tmp362 = -FDPart3tmp293 * FDPart3tmp44;
        const REAL FDPart3tmp14 = (1.0F / 2.0F) * FDPart3tmp13;
        const REAL FDPart3tmp26 = (1.0F / 2.0F) * FDPart3tmp25;
        const REAL FDPart3tmp34 = (1.0F / 2.0F) * FDPart3tmp33;
        const REAL FDPart3tmp49 = (1.0F / 2.0F) * FDPart3tmp48;
        const REAL FDPart3tmp59 = (1.0F / 2.0F) * FDPart3tmp58;
        const REAL FDPart3tmp80 = (1.0F / 2.0F) * FDPart3tmp79;
        const REAL FDPart3tmp85 = FDPart3tmp13 * FDPart3tmp83 + FDPart3tmp25 * FDPart3tmp82 + FDPart3tmp33 * FDPart3tmp84;
        const REAL FDPart3tmp86 = FDPart3tmp25 * FDPart3tmp83 + FDPart3tmp48 * FDPart3tmp84 + FDPart3tmp79 * FDPart3tmp82;
        const REAL FDPart3tmp87 = FDPart3tmp33 * FDPart3tmp83 + FDPart3tmp48 * FDPart3tmp82 + FDPart3tmp58 * FDPart3tmp84;
        const REAL FDPart3tmp90 = 3 * FDPart3tmp13;
        const REAL FDPart3tmp101 = 3 * FDPart3tmp33;
        const REAL FDPart3tmp109 = 3 * FDPart3tmp25;
        const REAL FDPart3tmp117 = 3 * FDPart3tmp48;
        const REAL FDPart3tmp119 = 3 * FDPart3tmp58;
        const REAL FDPart3tmp127 = 3 * FDPart3tmp79;
        const REAL FDPart3tmp99 = FDPart3tmp14 * hDD_dD001 + FDPart3tmp26 * FDPart3tmp98 + FDPart3tmp34 * FDPart3tmp93;
        const REAL FDPart3tmp107 = FDPart3tmp105 * FDPart3tmp26 + FDPart3tmp106 * FDPart3tmp34 + FDPart3tmp14 * hDD_dD002;
        const REAL FDPart3tmp110 = FDPart3tmp26 * hDD_dD001 + FDPart3tmp49 * FDPart3tmp93 + FDPart3tmp80 * FDPart3tmp98;
        const REAL FDPart3tmp112 = -FDPart3tmp16 + FDPart3tmp34 * hDD_dD001 + FDPart3tmp49 * FDPart3tmp98 + FDPart3tmp59 * FDPart3tmp93;
        const REAL FDPart3tmp120 = FDPart3tmp105 * FDPart3tmp80 + FDPart3tmp106 * FDPart3tmp49 - FDPart3tmp16 + FDPart3tmp26 * hDD_dD002;
        const REAL FDPart3tmp121 = FDPart3tmp105 * FDPart3tmp49 + FDPart3tmp106 * FDPart3tmp59 + FDPart3tmp34 * hDD_dD002;
        const REAL FDPart3tmp128 = FDPart3tmp32 * FDPart3tmp85 + FDPart3tmp47 * FDPart3tmp86 + FDPart3tmp6 * FDPart3tmp87;
        const REAL FDPart3tmp129 = 2 * FDPart3tmp87;
        const REAL FDPart3tmp130 = FDPart3tmp10 * FDPart3tmp86 + FDPart3tmp24 * FDPart3tmp85 + FDPart3tmp47 * FDPart3tmp87;
        const REAL FDPart3tmp131 = 2 * FDPart3tmp86;
        const REAL FDPart3tmp147 = FDPart3tmp139 * FDPart3tmp49 + FDPart3tmp140 * FDPart3tmp26 + FDPart3tmp146 * FDPart3tmp80;
        const REAL FDPart3tmp148 = FDPart3tmp139 * FDPart3tmp59 + FDPart3tmp140 * FDPart3tmp34 + FDPart3tmp146 * FDPart3tmp49;
        const REAL FDPart3tmp149 = FDPart3tmp139 * FDPart3tmp34 + FDPart3tmp14 * FDPart3tmp140 + FDPart3tmp146 * FDPart3tmp26 + f0_of_xx0;
        const REAL FDPart3tmp158 = FDPart3tmp142 * FDPart3tmp49 + FDPart3tmp156 * FDPart3tmp80 + FDPart3tmp157 * FDPart3tmp26 - FDPart3tmp44;
        const REAL FDPart3tmp159 = FDPart3tmp142 * FDPart3tmp59 + FDPart3tmp156 * FDPart3tmp49 + FDPart3tmp157 * FDPart3tmp34;
        const REAL FDPart3tmp160 = FDPart3tmp14 * FDPart3tmp157 + FDPart3tmp142 * FDPart3tmp34 + FDPart3tmp156 * FDPart3tmp26;
        const REAL FDPart3tmp178 = FDPart3tmp175 * FDPart3tmp80 + FDPart3tmp176 * FDPart3tmp49 + FDPart3tmp177 * FDPart3tmp26;
        const REAL FDPart3tmp179 = FDPart3tmp175 * FDPart3tmp49 + FDPart3tmp176 * FDPart3tmp59 + FDPart3tmp177 * FDPart3tmp34 + FDPart3tmp73;
        const REAL FDPart3tmp180 = FDPart3tmp14 * FDPart3tmp177 + FDPart3tmp175 * FDPart3tmp26 + FDPart3tmp176 * FDPart3tmp34 + FDPart3tmp74;
        const REAL FDPart3tmp89 = FDPart3tmp0 * FDPart3tmp85 + FDPart3tmp24 * FDPart3tmp86 + FDPart3tmp32 * FDPart3tmp87;
        const REAL FDPart3tmp150 = FDPart3tmp0 * FDPart3tmp149 + FDPart3tmp147 * FDPart3tmp24 + FDPart3tmp148 * FDPart3tmp32;
        const REAL FDPart3tmp151 = FDPart3tmp10 * FDPart3tmp110 + FDPart3tmp112 * FDPart3tmp47 + FDPart3tmp24 * FDPart3tmp99;
        const REAL FDPart3tmp161 = FDPart3tmp0 * FDPart3tmp160 + FDPart3tmp158 * FDPart3tmp24 + FDPart3tmp159 * FDPart3tmp32;
        const REAL FDPart3tmp170 = FDPart3tmp107 * FDPart3tmp32 + FDPart3tmp120 * FDPart3tmp47 + FDPart3tmp121 * FDPart3tmp6;
        const REAL FDPart3tmp181 = FDPart3tmp0 * FDPart3tmp180 + FDPart3tmp178 * FDPart3tmp24 + FDPart3tmp179 * FDPart3tmp32;
        const REAL FDPart3tmp197 = FDPart3tmp107 * FDPart3tmp25 + FDPart3tmp14 * FDPart3tmp85 + FDPart3tmp149 * FDPart3tmp59 +
                                   FDPart3tmp160 * FDPart3tmp48 + FDPart3tmp180 * FDPart3tmp80 + FDPart3tmp33 * FDPart3tmp99;
        const REAL FDPart3tmp251 = FDPart3tmp10 * FDPart3tmp147 + FDPart3tmp148 * FDPart3tmp47 + FDPart3tmp149 * FDPart3tmp24;
        const REAL FDPart3tmp270 = FDPart3tmp178 * FDPart3tmp47 + FDPart3tmp179 * FDPart3tmp6 + FDPart3tmp180 * FDPart3tmp32;
        const REAL FDPart3tmp340 = 2 * FDPart3tmp149;
        const REAL FDPart3tmp342 = 2 * FDPart3tmp147;
        const REAL FDPart3tmp383 = 2 * FDPart3tmp179;
        const REAL FDPart3tmp114 = FDPart3tmp0 * FDPart3tmp99 + FDPart3tmp110 * FDPart3tmp24 + FDPart3tmp112 * FDPart3tmp32;
        const REAL FDPart3tmp123 = FDPart3tmp0 * FDPart3tmp107 + FDPart3tmp120 * FDPart3tmp24 + FDPart3tmp121 * FDPart3tmp32;
        const REAL FDPart3tmp137 = FDPart3tmp110 * FDPart3tmp47 + FDPart3tmp112 * FDPart3tmp6 + FDPart3tmp32 * FDPart3tmp99;
        const REAL FDPart3tmp162 = FDPart3tmp110 * FDPart3tmp151;
        const REAL FDPart3tmp173 = FDPart3tmp10 * FDPart3tmp120 + FDPart3tmp107 * FDPart3tmp24 + FDPart3tmp121 * FDPart3tmp47;
        const REAL FDPart3tmp186 = FDPart3tmp121 * FDPart3tmp170;
        const REAL FDPart3tmp194 = FDPart3tmp110 * FDPart3tmp33 + FDPart3tmp120 * FDPart3tmp25 + FDPart3tmp14 * FDPart3tmp86 +
                                   FDPart3tmp147 * FDPart3tmp59 + FDPart3tmp158 * FDPart3tmp48 + FDPart3tmp178 * FDPart3tmp80;
        const REAL FDPart3tmp196 = FDPart3tmp112 * FDPart3tmp33 + FDPart3tmp121 * FDPart3tmp25 + FDPart3tmp14 * FDPart3tmp87 +
                                   FDPart3tmp148 * FDPart3tmp59 + FDPart3tmp159 * FDPart3tmp48 + FDPart3tmp179 * FDPart3tmp80;
        const REAL FDPart3tmp237 = FDPart3tmp147 * FDPart3tmp47 + FDPart3tmp148 * FDPart3tmp6 + FDPart3tmp149 * FDPart3tmp32;
        const REAL FDPart3tmp242 = FDPart3tmp158 * FDPart3tmp47 + FDPart3tmp159 * FDPart3tmp6 + FDPart3tmp160 * FDPart3tmp32;
        const REAL FDPart3tmp266 = FDPart3tmp161 * FDPart3tmp99;
        const REAL FDPart3tmp268 = FDPart3tmp10 * FDPart3tmp158 + FDPart3tmp159 * FDPart3tmp47 + FDPart3tmp160 * FDPart3tmp24;
        const REAL FDPart3tmp273 = FDPart3tmp107 * FDPart3tmp161;
        const REAL FDPart3tmp304 = FDPart3tmp10 * FDPart3tmp178 + FDPart3tmp179 * FDPart3tmp47 + FDPart3tmp180 * FDPart3tmp24;
        const REAL FDPart3tmp349 = FDPart3tmp160 * FDPart3tmp161;
        const REAL FDPart3tmp116 = FDPart3tmp107 * FDPart3tmp114;
        const REAL FDPart3tmp118 = FDPart3tmp114 * FDPart3tmp99;
        const REAL FDPart3tmp125 = FDPart3tmp123 * FDPart3tmp99;
        const REAL FDPart3tmp126 = FDPart3tmp107 * FDPart3tmp123;
        const REAL FDPart3tmp164 = FDPart3tmp121 * FDPart3tmp137;
        const REAL FDPart3tmp166 = FDPart3tmp112 * FDPart3tmp137;
        const REAL FDPart3tmp188 = FDPart3tmp120 * FDPart3tmp173;
        const REAL FDPart3tmp244 = FDPart3tmp112 * FDPart3tmp242;
        const REAL FDPart3tmp246 = FDPart3tmp121 * FDPart3tmp242;
        const REAL FDPart3tmp255 = FDPart3tmp114 * FDPart3tmp160;
        const REAL FDPart3tmp271 = FDPart3tmp123 * FDPart3tmp160;
        const REAL FDPart3tmp274 = FDPart3tmp110 * FDPart3tmp268;
        const REAL FDPart3tmp276 = FDPart3tmp120 * FDPart3tmp268;
        const REAL FDPart3tmp305 = 2 * FDPart3tmp304;
        const REAL FDPart3tmp309 = FDPart3tmp112 * FDPart3tmp151 + FDPart3tmp112 * FDPart3tmp170;
        const REAL FDPart3tmp311 = FDPart3tmp151 * FDPart3tmp99 + FDPart3tmp266;
        const REAL FDPart3tmp315 = FDPart3tmp173 * FDPart3tmp99 + FDPart3tmp181 * FDPart3tmp99;
        const REAL FDPart3tmp316 = FDPart3tmp112 * FDPart3tmp268 + FDPart3tmp112 * FDPart3tmp270;
        const REAL FDPart3tmp339 = FDPart3tmp159 * FDPart3tmp242;
        const REAL FDPart3tmp350 = FDPart3tmp158 * FDPart3tmp268;
        const REAL FDPart3tmp301 = 2 * FDPart3tmp110 * FDPart3tmp173;
        const REAL FDPart3tmp307 = FDPart3tmp125 + FDPart3tmp130 * FDPart3tmp99;
        const REAL FDPart3tmp312 = FDPart3tmp112 * FDPart3tmp251 + FDPart3tmp244;
        auxevol_gfs[IDX4(RBARDD00GF, i0, i1, i2)] =
            FDPart3tmp0 * lambdaU_dD00 + FDPart3tmp101 * FDPart3tmp114 * FDPart3tmp85 + FDPart3tmp101 * FDPart3tmp89 * FDPart3tmp99 +
            FDPart3tmp107 * FDPart3tmp109 * FDPart3tmp89 + FDPart3tmp109 * FDPart3tmp123 * FDPart3tmp85 + FDPart3tmp116 * FDPart3tmp117 +
            FDPart3tmp117 * FDPart3tmp125 + FDPart3tmp118 * FDPart3tmp119 + FDPart3tmp126 * FDPart3tmp127 +
            FDPart3tmp13 * (FDPart3tmp114 * FDPart3tmp129 + FDPart3tmp128 * FDPart3tmp87) +
            FDPart3tmp13 * (FDPart3tmp123 * FDPart3tmp131 + FDPart3tmp130 * FDPart3tmp86) - FDPart3tmp14 * hDD_dDD0000 +
            FDPart3tmp194 * (2 * FDPart3tmp0 * FDPart3tmp107 + FDPart3tmp120 * FDPart3tmp190 + FDPart3tmp121 * FDPart3tmp191) +
            FDPart3tmp196 * (2 * FDPart3tmp0 * FDPart3tmp99 + FDPart3tmp110 * FDPart3tmp190 + FDPart3tmp112 * FDPart3tmp191) +
            FDPart3tmp197 * (2 * FDPart3tmp0 * FDPart3tmp85 + FDPart3tmp190 * FDPart3tmp86 + FDPart3tmp191 * FDPart3tmp87) +
            FDPart3tmp25 * (2 * FDPart3tmp114 * FDPart3tmp121 + FDPart3tmp121 * FDPart3tmp128) +
            FDPart3tmp25 * (2 * FDPart3tmp120 * FDPart3tmp123 + FDPart3tmp120 * FDPart3tmp130) +
            FDPart3tmp25 * (FDPart3tmp129 * FDPart3tmp161 + FDPart3tmp170 * FDPart3tmp87) +
            FDPart3tmp25 * (FDPart3tmp131 * FDPart3tmp181 + FDPart3tmp173 * FDPart3tmp86) -
            FDPart3tmp26 * (-2 * FDPart3tmp15 - FDPart3tmp21 + hDD_dDD0002) -
            FDPart3tmp26 * (2 * FDPart3tmp16 * FDPart3tmp17 - 2 * FDPart3tmp16 * FDPart3tmp50 - FDPart3tmp21 + hDD_dDD0002) +
            FDPart3tmp33 * (2 * FDPart3tmp110 * FDPart3tmp123 + FDPart3tmp110 * FDPart3tmp130) +
            FDPart3tmp33 * (2 * FDPart3tmp112 * FDPart3tmp114 + FDPart3tmp112 * FDPart3tmp128) +
            FDPart3tmp33 * (FDPart3tmp129 * FDPart3tmp150 + FDPart3tmp137 * FDPart3tmp87) +
            FDPart3tmp33 * (FDPart3tmp131 * FDPart3tmp161 + FDPart3tmp151 * FDPart3tmp86) -
            FDPart3tmp34 * (-FDPart3tmp31 - 2 * hDD_dD010 + hDD_dDD0001) -
            FDPart3tmp34 * (FDPart3tmp16 * FDPart3tmp28 - 2 * FDPart3tmp16 * FDPart3tmp61 - FDPart3tmp31 + hDD_dDD0001) +
            FDPart3tmp48 * (FDPart3tmp110 * FDPart3tmp173 + 2 * FDPart3tmp110 * FDPart3tmp181) +
            FDPart3tmp48 * (2 * FDPart3tmp112 * FDPart3tmp161 + FDPart3tmp112 * FDPart3tmp170) +
            FDPart3tmp48 * (FDPart3tmp120 * FDPart3tmp151 + 2 * FDPart3tmp120 * FDPart3tmp161) +
            FDPart3tmp48 * (2 * FDPart3tmp121 * FDPart3tmp150 + FDPart3tmp164) - FDPart3tmp49 * (-FDPart3tmp42 - FDPart3tmp45 - 2 * hDD_dD012) -
            FDPart3tmp49 * (-2 * FDPart3tmp16 * FDPart3tmp65 - FDPart3tmp45 - FDPart3tmp71) +
            FDPart3tmp58 * (2 * FDPart3tmp110 * FDPart3tmp161 + FDPart3tmp162) + FDPart3tmp58 * (2 * FDPart3tmp112 * FDPart3tmp150 + FDPart3tmp166) -
            FDPart3tmp59 * (FDPart3tmp57 - 2 * hDD_dD011 + hDD_dDD0011) + FDPart3tmp79 * (2 * FDPart3tmp120 * FDPart3tmp181 + FDPart3tmp188) +
            FDPart3tmp79 * (2 * FDPart3tmp121 * FDPart3tmp161 + FDPart3tmp186) -
            FDPart3tmp80 * (-FDPart3tmp29 * FDPart3tmp73 - 2 * FDPart3tmp72 + FDPart3tmp78 + hDD_dDD0022) +
            FDPart3tmp85 * FDPart3tmp89 * FDPart3tmp90 + hDD01 * lambdaU_dD10 + hDD02 * lambdaU_dD20;
        auxevol_gfs[IDX4(RBARDD01GF, i0, i1, i2)] =
            (1.0F / 2.0F) * FDPart3tmp0 * FDPart3tmp202 + FDPart3tmp13 * (FDPart3tmp112 * FDPart3tmp114 + 2 * FDPart3tmp137 * FDPart3tmp87) +
            FDPart3tmp13 * (FDPart3tmp110 * FDPart3tmp123 + FDPart3tmp151 * FDPart3tmp86 + FDPart3tmp170 * FDPart3tmp86) +
            FDPart3tmp13 * (FDPart3tmp114 * FDPart3tmp85 + FDPart3tmp128 * FDPart3tmp85 + FDPart3tmp89 * FDPart3tmp99) -
            FDPart3tmp14 * (-FDPart3tmp16 * FDPart3tmp61 + FDPart3tmp16 * hDD01 + f0_of_xx0 * hDD_dDD0100 + hDD_dD010) +
            (1.0F / 2.0F) * FDPart3tmp16 * FDPart3tmp6 * lambdaU_dD10 + FDPart3tmp194 * (FDPart3tmp161 + FDPart3tmp170) +
            FDPart3tmp196 * (FDPart3tmp137 + FDPart3tmp150) + FDPart3tmp197 * (FDPart3tmp114 + FDPart3tmp128) + FDPart3tmp199 * FDPart3tmp206 +
            FDPart3tmp199 * lambdaU_dD00 + FDPart3tmp25 * (FDPart3tmp112 * FDPart3tmp161 + 2 * FDPart3tmp242 * FDPart3tmp87) +
            FDPart3tmp25 * (FDPart3tmp114 * FDPart3tmp159 + 2 * FDPart3tmp164) +
            FDPart3tmp25 * (FDPart3tmp125 + FDPart3tmp161 * FDPart3tmp85 + FDPart3tmp170 * FDPart3tmp85) +
            FDPart3tmp25 * (FDPart3tmp107 * FDPart3tmp128 + FDPart3tmp116 + FDPart3tmp160 * FDPart3tmp89) +
            FDPart3tmp25 * (FDPart3tmp110 * FDPart3tmp181 + FDPart3tmp268 * FDPart3tmp86 + FDPart3tmp270 * FDPart3tmp86) +
            FDPart3tmp25 * (FDPart3tmp120 * FDPart3tmp151 + FDPart3tmp120 * FDPart3tmp170 + FDPart3tmp123 * FDPart3tmp158) -
            FDPart3tmp26 * (FDPart3tmp218 - FDPart3tmp71 + f0_of_xx0 * hDD_dDD0102 + hDD_dD012) -
            FDPart3tmp26 * (-FDPart3tmp16 * FDPart3tmp69 - FDPart3tmp207 * hDD_dD120 - FDPart3tmp210 + f0_of_xx0 * hDD_dDD0102) +
            FDPart3tmp33 * (FDPart3tmp112 * FDPart3tmp150 + FDPart3tmp129 * FDPart3tmp237) +
            FDPart3tmp33 * (FDPart3tmp114 * FDPart3tmp148 + 2 * FDPart3tmp166) +
            FDPart3tmp33 * (FDPart3tmp118 + FDPart3tmp128 * FDPart3tmp99 + FDPart3tmp149 * FDPart3tmp89) +
            FDPart3tmp33 * (FDPart3tmp118 + FDPart3tmp137 * FDPart3tmp85 + FDPart3tmp150 * FDPart3tmp85) +
            FDPart3tmp33 * (FDPart3tmp110 * FDPart3tmp161 + FDPart3tmp242 * FDPart3tmp86 + FDPart3tmp251 * FDPart3tmp86) +
            FDPart3tmp33 * (FDPart3tmp110 * FDPart3tmp170 + FDPart3tmp123 * FDPart3tmp147 + FDPart3tmp162) -
            FDPart3tmp34 * (-FDPart3tmp16 * FDPart3tmp55 + f0_of_xx0 * hDD_dD000 - f0_of_xx0 * hDD_dD110 + f0_of_xx0 * hDD_dDD0101) -
            FDPart3tmp34 * (-FDPart3tmp16 * FDPart3tmp92 + FDPart3tmp57 + f0_of_xx0 * hDD_dDD0101 + hDD00 + hDD11 + hDD_dD011) +
            FDPart3tmp48 * (2 * FDPart3tmp121 * FDPart3tmp237 + FDPart3tmp150 * FDPart3tmp159) +
            FDPart3tmp48 * (FDPart3tmp148 * FDPart3tmp161 + 2 * FDPart3tmp244) +
            FDPart3tmp48 * (FDPart3tmp107 * FDPart3tmp137 + FDPart3tmp107 * FDPart3tmp150 + FDPart3tmp255) +
            FDPart3tmp48 * (FDPart3tmp110 * FDPart3tmp270 + FDPart3tmp147 * FDPart3tmp181 + FDPart3tmp274) +
            FDPart3tmp48 * (FDPart3tmp120 * FDPart3tmp242 + FDPart3tmp120 * FDPart3tmp251 + FDPart3tmp158 * FDPart3tmp161) +
            FDPart3tmp48 * (FDPart3tmp123 * FDPart3tmp149 + FDPart3tmp170 * FDPart3tmp99 + FDPart3tmp266) -
            FDPart3tmp49 * (-FDPart3tmp219 - FDPart3tmp222 + f0_of_xx0 * hDD_dD002 - f0_of_xx0 * hDD_dD112) -
            FDPart3tmp49 * (-FDPart3tmp19 * f0_of_xx0 - FDPart3tmp219 - FDPart3tmp224 - FDPart3tmp229) +
            FDPart3tmp58 * (2 * FDPart3tmp112 * FDPart3tmp237 + FDPart3tmp148 * FDPart3tmp150) +
            FDPart3tmp58 * (FDPart3tmp110 * FDPart3tmp242 + FDPart3tmp110 * FDPart3tmp251 + FDPart3tmp147 * FDPart3tmp161) +
            FDPart3tmp58 * (FDPart3tmp114 * FDPart3tmp149 + FDPart3tmp137 * FDPart3tmp99 + FDPart3tmp150 * FDPart3tmp99) -
            FDPart3tmp59 * (FDPart3tmp216 - FDPart3tmp29 * f0_of_xx0 + FDPart3tmp4 * hDD_dD010 + f0_of_xx0 * hDD_dD001 - f0_of_xx0 * hDD_dD111 +
                            f0_of_xx0 * hDD_dDD0111) +
            FDPart3tmp79 * (FDPart3tmp159 * FDPart3tmp161 + 2 * FDPart3tmp246) +
            FDPart3tmp79 * (FDPart3tmp107 * FDPart3tmp170 + FDPart3tmp271 + FDPart3tmp273) +
            FDPart3tmp79 * (FDPart3tmp120 * FDPart3tmp270 + FDPart3tmp158 * FDPart3tmp181 + FDPart3tmp276) -
            FDPart3tmp80 * (FDPart3tmp2 * FDPart3tmp4 * hDD_dD010 - FDPart3tmp232 - FDPart3tmp233 - FDPart3tmp235 + f0_of_xx0 * hDD_dDD0122) +
            (1.0F / 2.0F) * f0_of_xx0 * hDD12 * lambdaU_dD20 + (1.0F / 2.0F) * hDD02 * lambdaU_dD21;
        auxevol_gfs[IDX4(RBARDD02GF, i0, i1, i2)] =
            (1.0F / 2.0F) * FDPart3tmp0 * FDPart3tmp280 + FDPart3tmp13 * (FDPart3tmp120 * FDPart3tmp123 + 2 * FDPart3tmp173 * FDPart3tmp86) +
            FDPart3tmp13 * (FDPart3tmp107 * FDPart3tmp89 + FDPart3tmp123 * FDPart3tmp85 + FDPart3tmp130 * FDPart3tmp85) +
            FDPart3tmp13 * (FDPart3tmp114 * FDPart3tmp121 + FDPart3tmp151 * FDPart3tmp87 + FDPart3tmp170 * FDPart3tmp87) -
            FDPart3tmp14 * (FDPart3tmp15 + FDPart3tmp16 * FDPart3tmp17 - FDPart3tmp16 * FDPart3tmp50 + FDPart3tmp207 * hDD_dDD0200) +
            FDPart3tmp194 * (FDPart3tmp173 + FDPart3tmp181) + FDPart3tmp196 * (FDPart3tmp151 + FDPart3tmp161) +
            FDPart3tmp197 * (FDPart3tmp123 + FDPart3tmp130) + FDPart3tmp199 * FDPart3tmp281 +
            FDPart3tmp25 * (FDPart3tmp120 * FDPart3tmp181 + FDPart3tmp131 * FDPart3tmp304) +
            FDPart3tmp25 * (FDPart3tmp123 * FDPart3tmp178 + 2 * FDPart3tmp188) +
            FDPart3tmp25 * (FDPart3tmp126 + FDPart3tmp173 * FDPart3tmp85 + FDPart3tmp181 * FDPart3tmp85) +
            FDPart3tmp25 * (FDPart3tmp107 * FDPart3tmp130 + FDPart3tmp126 + FDPart3tmp180 * FDPart3tmp89) +
            FDPart3tmp25 * (FDPart3tmp114 * FDPart3tmp179 + FDPart3tmp121 * FDPart3tmp151 + FDPart3tmp186) +
            FDPart3tmp25 * (FDPart3tmp121 * FDPart3tmp161 + FDPart3tmp268 * FDPart3tmp87 + FDPart3tmp270 * FDPart3tmp87) -
            FDPart3tmp26 * (-FDPart3tmp16 * FDPart3tmp75 + FDPart3tmp207 * hDD_dDD0202 + FDPart3tmp73 * f0_of_xx0 * hDD_dD010 +
                            FDPart3tmp74 * hDD_dD000 - FDPart3tmp74 * hDD_dD220) -
            FDPart3tmp26 * (FDPart3tmp103 - FDPart3tmp104 * FDPart3tmp16 + FDPart3tmp2 * hDD00 + FDPart3tmp207 * hDD_dDD0202 +
                            FDPart3tmp61 * FDPart3tmp73 + FDPart3tmp72 + FDPart3tmp78) +
            FDPart3tmp279 * FDPart3tmp284 + FDPart3tmp279 * lambdaU_dD00 + (1.0F / 2.0F) * FDPart3tmp283 * lambdaU_dD20 +
            FDPart3tmp33 * (FDPart3tmp114 * FDPart3tmp159 + FDPart3tmp309) +
            FDPart3tmp33 * (FDPart3tmp120 * FDPart3tmp161 + 2 * FDPart3tmp268 * FDPart3tmp86) +
            FDPart3tmp33 * (FDPart3tmp123 * FDPart3tmp158 + FDPart3tmp301) + FDPart3tmp33 * (FDPart3tmp160 * FDPart3tmp89 + FDPart3tmp307) +
            FDPart3tmp33 * (FDPart3tmp116 + FDPart3tmp151 * FDPart3tmp85 + FDPart3tmp161 * FDPart3tmp85) +
            FDPart3tmp33 * (FDPart3tmp121 * FDPart3tmp150 + FDPart3tmp242 * FDPart3tmp87 + FDPart3tmp251 * FDPart3tmp87) -
            FDPart3tmp34 * (-FDPart3tmp16 * FDPart3tmp40 - FDPart3tmp16 * FDPart3tmp65 - FDPart3tmp207 * hDD_dD120 +
                            f0_of_xx0 * f1_of_xx1 * hDD_dDD0201 + f1_of_xx1 * hDD_dD021 + f1_of_xx1__D1 * hDD02) -
            FDPart3tmp34 * (FDPart3tmp207 * hDD_dDD0201 + FDPart3tmp210 + FDPart3tmp218 - FDPart3tmp42 + FDPart3tmp63 + f1_of_xx1 * hDD_dD021) +
            (1.0F / 2.0F) * FDPart3tmp36 * lambdaU_dD10 + FDPart3tmp48 * (FDPart3tmp271 + FDPart3tmp315) +
            FDPart3tmp48 * (FDPart3tmp110 * FDPart3tmp305 + FDPart3tmp158 * FDPart3tmp181) +
            FDPart3tmp48 * (FDPart3tmp159 * FDPart3tmp161 + FDPart3tmp316) + FDPart3tmp48 * (FDPart3tmp161 * FDPart3tmp178 + 2 * FDPart3tmp276) +
            FDPart3tmp48 * (FDPart3tmp107 * FDPart3tmp151 + FDPart3tmp114 * FDPart3tmp180 + FDPart3tmp273) +
            FDPart3tmp48 * (FDPart3tmp121 * FDPart3tmp251 + FDPart3tmp150 * FDPart3tmp179 + FDPart3tmp246) -
            FDPart3tmp49 * (-FDPart3tmp2 * FDPart3tmp29 * f0_of_xx0 - FDPart3tmp235 - FDPart3tmp286 + f0_of_xx0 * f1_of_xx1 * hDD_dDD0212) -
            FDPart3tmp49 * (-FDPart3tmp155 * FDPart3tmp16 + FDPart3tmp207 * hDD_dDD0212 - FDPart3tmp232 + FDPart3tmp233 +
                            FDPart3tmp297 * FDPart3tmp32 - 2 * FDPart3tmp44 * FDPart3tmp75 + FDPart3tmp73 * FDPart3tmp81 * hDD00 +
                            FDPart3tmp73 * f0_of_xx0 * hDD_dD011 + FDPart3tmp74 * hDD_dD001) +
            FDPart3tmp58 * (FDPart3tmp255 + FDPart3tmp311) + FDPart3tmp58 * (FDPart3tmp150 * FDPart3tmp159 + FDPart3tmp312) +
            FDPart3tmp58 * (FDPart3tmp158 * FDPart3tmp161 + 2 * FDPart3tmp274) -
            FDPart3tmp59 * (-FDPart3tmp222 - FDPart3tmp229 + FDPart3tmp4 * f1_of_xx1 * hDD_dD020 + f0_of_xx0 * f1_of_xx1 * hDD_dDD0211 +
                            2 * f0_of_xx0 * f1_of_xx1__D1 * hDD_dD021 + f0_of_xx0 * f1_of_xx1__DD11 * hDD02) +
            FDPart3tmp79 * (FDPart3tmp120 * FDPart3tmp305 + FDPart3tmp178 * FDPart3tmp181) +
            FDPart3tmp79 * (FDPart3tmp107 * FDPart3tmp173 + FDPart3tmp107 * FDPart3tmp181 + FDPart3tmp123 * FDPart3tmp180) +
            FDPart3tmp79 * (FDPart3tmp121 * FDPart3tmp268 + FDPart3tmp121 * FDPart3tmp270 + FDPart3tmp161 * FDPart3tmp179) -
            FDPart3tmp80 * (-FDPart3tmp19 * FDPart3tmp74 + FDPart3tmp207 * hDD_dDD0222 + FDPart3tmp289 * hDD_dD020 + FDPart3tmp295 +
                            FDPart3tmp40 * FDPart3tmp73 + FDPart3tmp67 * FDPart3tmp73 + FDPart3tmp69 * FDPart3tmp73 + FDPart3tmp74 * hDD_dD002 -
                            FDPart3tmp74 * hDD_dD222);
        auxevol_gfs[IDX4(RBARDD11GF, i0, i1, i2)] =
            FDPart3tmp101 * FDPart3tmp137 * FDPart3tmp148 + FDPart3tmp109 * FDPart3tmp137 * FDPart3tmp159 + FDPart3tmp109 * FDPart3tmp244 +
            3 * FDPart3tmp112 * FDPart3tmp237 * FDPart3tmp33 + FDPart3tmp117 * FDPart3tmp148 * FDPart3tmp242 +
            FDPart3tmp117 * FDPart3tmp159 * FDPart3tmp237 + FDPart3tmp119 * FDPart3tmp148 * FDPart3tmp237 + FDPart3tmp127 * FDPart3tmp339 +
            FDPart3tmp13 * (FDPart3tmp118 + 2 * FDPart3tmp128 * FDPart3tmp99) + FDPart3tmp13 * (2 * FDPart3tmp110 * FDPart3tmp170 + FDPart3tmp162) -
            FDPart3tmp14 * (-2 * FDPart3tmp16 * FDPart3tmp92 + FDPart3tmp4 * hDD_dDD1100 + FDPart3tmp81 * hDD_dD110 + 4 * hDD11) +
            FDPart3tmp166 * FDPart3tmp90 +
            FDPart3tmp194 * (FDPart3tmp158 * FDPart3tmp352 + 2 * FDPart3tmp159 * FDPart3tmp6 + FDPart3tmp160 * FDPart3tmp191) +
            FDPart3tmp196 * (2 * FDPart3tmp148 * FDPart3tmp6 + FDPart3tmp149 * FDPart3tmp191 + FDPart3tmp342 * FDPart3tmp47) +
            FDPart3tmp197 * (FDPart3tmp110 * FDPart3tmp352 + 2 * FDPart3tmp112 * FDPart3tmp6 + FDPart3tmp191 * FDPart3tmp99) +
            FDPart3tmp202 * FDPart3tmp32 + FDPart3tmp206 * FDPart3tmp6 + FDPart3tmp25 * (2 * FDPart3tmp110 * FDPart3tmp270 + FDPart3tmp274) +
            FDPart3tmp25 * (2 * FDPart3tmp128 * FDPart3tmp160 + FDPart3tmp255) +
            FDPart3tmp25 * (FDPart3tmp151 * FDPart3tmp158 + 2 * FDPart3tmp158 * FDPart3tmp170) +
            FDPart3tmp25 * (2 * FDPart3tmp170 * FDPart3tmp99 + FDPart3tmp266) -
            FDPart3tmp26 * (-FDPart3tmp224 - 2 * FDPart3tmp322 * hDD_dD120 + FDPart3tmp4 * hDD_dDD1102) -
            FDPart3tmp26 * (-3 * FDPart3tmp224 + FDPart3tmp4 * hDD_dDD1102 - 2 * FDPart3tmp44 * FDPart3tmp97 + FDPart3tmp81 * hDD_dD112) +
            FDPart3tmp33 * (2 * FDPart3tmp110 * FDPart3tmp242 + FDPart3tmp110 * FDPart3tmp251) +
            FDPart3tmp33 * (FDPart3tmp114 * FDPart3tmp149 + FDPart3tmp128 * FDPart3tmp340) +
            FDPart3tmp33 * (2 * FDPart3tmp137 * FDPart3tmp99 + FDPart3tmp150 * FDPart3tmp99) +
            FDPart3tmp33 * (FDPart3tmp147 * FDPart3tmp151 + FDPart3tmp170 * FDPart3tmp342) -
            FDPart3tmp34 * (FDPart3tmp216 + 2 * FDPart3tmp4 * hDD_dD010 + FDPart3tmp4 * hDD_dDD1101) -
            FDPart3tmp34 * (-3 * FDPart3tmp16 * FDPart3tmp214 + FDPart3tmp191 + FDPart3tmp4 * hDD_dDD1101 + FDPart3tmp61 * FDPart3tmp81 +
                            FDPart3tmp81 * hDD_dD111) +
            FDPart3tmp48 * (2 * FDPart3tmp137 * FDPart3tmp160 + FDPart3tmp150 * FDPart3tmp160) +
            FDPart3tmp48 * (FDPart3tmp147 * FDPart3tmp268 + FDPart3tmp270 * FDPart3tmp342) +
            FDPart3tmp48 * (FDPart3tmp149 * FDPart3tmp161 + FDPart3tmp170 * FDPart3tmp340) +
            FDPart3tmp48 * (2 * FDPart3tmp158 * FDPart3tmp242 + FDPart3tmp158 * FDPart3tmp251) -
            FDPart3tmp49 * (-2 * FDPart3tmp220 * FDPart3tmp44 + FDPart3tmp327 + 2 * FDPart3tmp4 * hDD_dD012) -
            FDPart3tmp49 *
                (-2 * FDPart3tmp225 * FDPart3tmp44 - FDPart3tmp228 * FDPart3tmp47 + FDPart3tmp327 + FDPart3tmp335 + FDPart3tmp69 * FDPart3tmp81) +
            FDPart3tmp58 * (FDPart3tmp137 * FDPart3tmp340 + FDPart3tmp149 * FDPart3tmp150) +
            FDPart3tmp58 * (FDPart3tmp147 * FDPart3tmp251 + FDPart3tmp242 * FDPart3tmp342) -
            FDPart3tmp59 * (FDPart3tmp22 * hDD_dD110 + 2 * FDPart3tmp4 * hDD_dD011 + FDPart3tmp4 * hDD_dDD1111 + FDPart3tmp55 * FDPart3tmp81) +
            FDPart3tmp79 * (2 * FDPart3tmp158 * FDPart3tmp270 + FDPart3tmp350) + FDPart3tmp79 * (2 * FDPart3tmp160 * FDPart3tmp170 + FDPart3tmp349) -
            FDPart3tmp80 * (FDPart3tmp2 * FDPart3tmp22 * hDD_dD110 + FDPart3tmp214 * FDPart3tmp73 - 2 * FDPart3tmp322 * hDD_dD122 + FDPart3tmp332 +
                            FDPart3tmp4 * hDD_dDD1122) +
            f0_of_xx0 * hDD12 * lambdaU_dD21;
        auxevol_gfs[IDX4(RBARDD12GF, i0, i1, i2)] =
            FDPart3tmp13 * (FDPart3tmp164 + FDPart3tmp309) + FDPart3tmp13 * (FDPart3tmp107 * FDPart3tmp128 + FDPart3tmp307) +
            FDPart3tmp13 * (FDPart3tmp120 * FDPart3tmp170 + FDPart3tmp301) -
            FDPart3tmp14 * (-2 * FDPart3tmp16 * FDPart3tmp97 + 4 * FDPart3tmp23 + FDPart3tmp354 * hDD_dD120 + FDPart3tmp95 * hDD_dDD1200) +
            FDPart3tmp194 * (FDPart3tmp268 + FDPart3tmp270) + FDPart3tmp196 * (FDPart3tmp242 + FDPart3tmp251) +
            FDPart3tmp197 * (FDPart3tmp151 + FDPart3tmp170) + FDPart3tmp199 * FDPart3tmp280 + FDPart3tmp202 * FDPart3tmp279 +
            (1.0F / 2.0F) * FDPart3tmp206 * FDPart3tmp47 + FDPart3tmp25 * (FDPart3tmp246 + FDPart3tmp316) +
            FDPart3tmp25 * (FDPart3tmp107 * FDPart3tmp170 + FDPart3tmp315) +
            FDPart3tmp25 * (FDPart3tmp110 * FDPart3tmp305 + FDPart3tmp120 * FDPart3tmp270) +
            FDPart3tmp25 * (2 * FDPart3tmp158 * FDPart3tmp173 + FDPart3tmp170 * FDPart3tmp178) +
            FDPart3tmp25 * (FDPart3tmp128 * FDPart3tmp180 + FDPart3tmp130 * FDPart3tmp160 + FDPart3tmp271) +
            FDPart3tmp25 * (FDPart3tmp137 * FDPart3tmp179 + FDPart3tmp151 * FDPart3tmp159 + FDPart3tmp159 * FDPart3tmp170) -
            FDPart3tmp26 * (-FDPart3tmp230 * hDD_dD220 - FDPart3tmp232 + FDPart3tmp4 * FDPart3tmp73 * hDD_dD110 + FDPart3tmp7 * hDD_dD010 +
                            FDPart3tmp95 * hDD_dDD1202) -
            FDPart3tmp26 * (-FDPart3tmp104 * FDPart3tmp44 - 3 * FDPart3tmp232 + FDPart3tmp354 * hDD_dD122 + FDPart3tmp61 * FDPart3tmp74 +
                            FDPart3tmp73 * FDPart3tmp92 + FDPart3tmp74 * hDD01 + FDPart3tmp95 * hDD_dDD1202) +
            (1.0F / 2.0F) * FDPart3tmp281 * FDPart3tmp6 + (1.0F / 2.0F) * FDPart3tmp283 * lambdaU_dD21 +
            (1.0F / 2.0F) * FDPart3tmp284 * FDPart3tmp47 + FDPart3tmp33 * (FDPart3tmp107 * FDPart3tmp137 + FDPart3tmp311) +
            FDPart3tmp33 * (FDPart3tmp120 * FDPart3tmp242 + 2 * FDPart3tmp274) + FDPart3tmp33 * (FDPart3tmp121 * FDPart3tmp237 + FDPart3tmp312) +
            FDPart3tmp33 * (FDPart3tmp158 * FDPart3tmp170 + FDPart3tmp173 * FDPart3tmp342) +
            FDPart3tmp33 * (FDPart3tmp123 * FDPart3tmp149 + FDPart3tmp128 * FDPart3tmp160 + FDPart3tmp130 * FDPart3tmp149) +
            FDPart3tmp33 * (FDPart3tmp137 * FDPart3tmp159 + FDPart3tmp148 * FDPart3tmp151 + FDPart3tmp148 * FDPart3tmp170) -
            FDPart3tmp34 * (FDPart3tmp15 * FDPart3tmp4 - 2 * FDPart3tmp16 * FDPart3tmp225 - FDPart3tmp221 + FDPart3tmp356) -
            FDPart3tmp34 * (-3 * FDPart3tmp221 + FDPart3tmp24 + FDPart3tmp322 * hDD_dD120 + FDPart3tmp356 - FDPart3tmp44 * FDPart3tmp97 +
                            FDPart3tmp50 * f0_of_xx0) +
            FDPart3tmp48 * (FDPart3tmp147 * FDPart3tmp305 + FDPart3tmp158 * FDPart3tmp270) +
            FDPart3tmp48 * (FDPart3tmp178 * FDPart3tmp242 + 2 * FDPart3tmp350) +
            FDPart3tmp48 * (FDPart3tmp137 * FDPart3tmp180 + FDPart3tmp151 * FDPart3tmp160 + FDPart3tmp349) +
            FDPart3tmp48 * (FDPart3tmp148 * FDPart3tmp268 + FDPart3tmp148 * FDPart3tmp270 + FDPart3tmp339) +
            FDPart3tmp48 * (FDPart3tmp149 * FDPart3tmp173 + FDPart3tmp149 * FDPart3tmp181 + FDPart3tmp160 * FDPart3tmp170) +
            FDPart3tmp48 * (FDPart3tmp159 * FDPart3tmp251 + FDPart3tmp179 * FDPart3tmp237 + FDPart3tmp339) -
            FDPart3tmp49 * (FDPart3tmp214 * FDPart3tmp73 - FDPart3tmp230 * hDD_dD221 - FDPart3tmp231 * FDPart3tmp44 + FDPart3tmp4 * FDPart3tmp72 +
                            FDPart3tmp55 * FDPart3tmp74 + FDPart3tmp95 * hDD_dDD1212) -
            FDPart3tmp49 * (FDPart3tmp139 * FDPart3tmp73 - FDPart3tmp155 * FDPart3tmp44 + FDPart3tmp28 * FDPart3tmp4 * FDPart3tmp73 +
                            FDPart3tmp297 * FDPart3tmp5 + FDPart3tmp322 * hDD_dD122 + FDPart3tmp332 + FDPart3tmp363 + FDPart3tmp7 * hDD_dD011 +
                            FDPart3tmp75 * f0_of_xx0 + FDPart3tmp95 * hDD_dDD1212) +
            FDPart3tmp58 * (2 * FDPart3tmp147 * FDPart3tmp268 + FDPart3tmp158 * FDPart3tmp242) +
            FDPart3tmp58 * (FDPart3tmp137 * FDPart3tmp160 + FDPart3tmp149 * FDPart3tmp151 + FDPart3tmp149 * FDPart3tmp161) +
            FDPart3tmp58 * (FDPart3tmp148 * FDPart3tmp242 + FDPart3tmp148 * FDPart3tmp251 + FDPart3tmp159 * FDPart3tmp237) -
            FDPart3tmp59 * (FDPart3tmp22 * f1_of_xx1 * hDD_dD120 - FDPart3tmp220 * FDPart3tmp44 - FDPart3tmp225 * FDPart3tmp44 +
                            2 * FDPart3tmp322 * hDD_dD121 + FDPart3tmp335 + FDPart3tmp4 * f1_of_xx1__DD11 * hDD12 + FDPart3tmp40 * f0_of_xx0 +
                            FDPart3tmp65 * f0_of_xx0 + FDPart3tmp95 * hDD_dDD1211) +
            FDPart3tmp79 * (FDPart3tmp158 * FDPart3tmp305 + FDPart3tmp178 * FDPart3tmp270) +
            FDPart3tmp79 * (FDPart3tmp159 * FDPart3tmp268 + FDPart3tmp159 * FDPart3tmp270 + FDPart3tmp179 * FDPart3tmp242) +
            FDPart3tmp79 * (FDPart3tmp160 * FDPart3tmp173 + FDPart3tmp160 * FDPart3tmp181 + FDPart3tmp170 * FDPart3tmp180) -
            FDPart3tmp80 * (FDPart3tmp142 * FDPart3tmp73 - FDPart3tmp145 * FDPart3tmp73 + FDPart3tmp22 * FDPart3tmp288 * hDD_dD120 +
                            FDPart3tmp220 * FDPart3tmp73 - FDPart3tmp230 * hDD_dD222 + FDPart3tmp362 + FDPart3tmp69 * FDPart3tmp74 +
                            FDPart3tmp7 * hDD_dD012 + FDPart3tmp95 * hDD_dDD1222);
        auxevol_gfs[IDX4(RBARDD22GF, i0, i1, i2)] =
            FDPart3tmp10 * FDPart3tmp284 + FDPart3tmp101 * FDPart3tmp158 * FDPart3tmp173 + FDPart3tmp101 * FDPart3tmp276 +
            FDPart3tmp109 * FDPart3tmp173 * FDPart3tmp178 + FDPart3tmp117 * FDPart3tmp178 * FDPart3tmp268 + FDPart3tmp119 * FDPart3tmp350 +
            3 * FDPart3tmp120 * FDPart3tmp25 * FDPart3tmp304 + FDPart3tmp127 * FDPart3tmp178 * FDPart3tmp304 +
            FDPart3tmp13 * (2 * FDPart3tmp107 * FDPart3tmp130 + FDPart3tmp126) + FDPart3tmp13 * (2 * FDPart3tmp121 * FDPart3tmp151 + FDPart3tmp186) -
            FDPart3tmp14 * (FDPart3tmp102 * hDD_dD220 + 4 * FDPart3tmp103 - 2 * FDPart3tmp104 * FDPart3tmp16 + FDPart3tmp7 * hDD_dDD2200) +
            3 * FDPart3tmp158 * FDPart3tmp304 * FDPart3tmp48 + FDPart3tmp188 * FDPart3tmp90 +
            FDPart3tmp194 * (2 * FDPart3tmp10 * FDPart3tmp178 + FDPart3tmp180 * FDPart3tmp190 + FDPart3tmp383 * FDPart3tmp47) +
            FDPart3tmp196 * (2 * FDPart3tmp10 * FDPart3tmp158 + FDPart3tmp159 * FDPart3tmp352 + FDPart3tmp160 * FDPart3tmp190) +
            FDPart3tmp197 * (2 * FDPart3tmp10 * FDPart3tmp120 + FDPart3tmp107 * FDPart3tmp190 + FDPart3tmp121 * FDPart3tmp352) +
            FDPart3tmp24 * FDPart3tmp280 + FDPart3tmp25 * (2 * FDPart3tmp107 * FDPart3tmp173 + FDPart3tmp107 * FDPart3tmp181) +
            FDPart3tmp25 * (2 * FDPart3tmp121 * FDPart3tmp268 + FDPart3tmp121 * FDPart3tmp270) +
            FDPart3tmp25 * (FDPart3tmp123 * FDPart3tmp180 + 2 * FDPart3tmp130 * FDPart3tmp180) +
            FDPart3tmp25 * (FDPart3tmp151 * FDPart3tmp383 + FDPart3tmp170 * FDPart3tmp179) -
            FDPart3tmp26 * (2 * FDPart3tmp289 * hDD_dD020 + FDPart3tmp292 * hDD_dD120 + FDPart3tmp295 + FDPart3tmp7 * hDD_dDD2202) -
            FDPart3tmp26 * (FDPart3tmp102 * FDPart3tmp50 + FDPart3tmp102 * hDD_dD222 - 3 * FDPart3tmp16 * FDPart3tmp293 +
                            FDPart3tmp288 * FDPart3tmp81 * hDD02 + FDPart3tmp379 * FDPart3tmp97 + FDPart3tmp7 * hDD_dDD2202) +
            FDPart3tmp281 * FDPart3tmp47 + FDPart3tmp33 * (2 * FDPart3tmp107 * FDPart3tmp151 + FDPart3tmp273) +
            FDPart3tmp33 * (2 * FDPart3tmp121 * FDPart3tmp251 + FDPart3tmp246) + FDPart3tmp33 * (2 * FDPart3tmp130 * FDPart3tmp160 + FDPart3tmp271) +
            FDPart3tmp33 * (2 * FDPart3tmp151 * FDPart3tmp159 + FDPart3tmp159 * FDPart3tmp170) -
            FDPart3tmp34 * (-2 * FDPart3tmp155 * FDPart3tmp16 + FDPart3tmp286 + FDPart3tmp376) -
            FDPart3tmp34 * (-2 * FDPart3tmp104 * FDPart3tmp44 + FDPart3tmp154 * hDD_dD220 - FDPart3tmp286 + FDPart3tmp376) +
            FDPart3tmp48 * (2 * FDPart3tmp151 * FDPart3tmp180 + FDPart3tmp161 * FDPart3tmp180) +
            FDPart3tmp48 * (2 * FDPart3tmp159 * FDPart3tmp268 + FDPart3tmp159 * FDPart3tmp270) +
            FDPart3tmp48 * (2 * FDPart3tmp160 * FDPart3tmp173 + FDPart3tmp160 * FDPart3tmp181) +
            FDPart3tmp48 * (FDPart3tmp179 * FDPart3tmp242 + FDPart3tmp251 * FDPart3tmp383) -
            FDPart3tmp49 * (FDPart3tmp102 * FDPart3tmp40 + 2 * FDPart3tmp220 * FDPart3tmp73 + FDPart3tmp362 + FDPart3tmp7 * hDD_dDD2212) -
            FDPart3tmp49 *
                (FDPart3tmp102 * FDPart3tmp65 + FDPart3tmp154 * hDD_dD222 + FDPart3tmp225 * FDPart3tmp379 - 3 * FDPart3tmp293 * FDPart3tmp44 +
                 2 * FDPart3tmp297 * FDPart3tmp47 + 4 * FDPart3tmp63 * FDPart3tmp7 + FDPart3tmp7 * hDD_dDD2212) +
            FDPart3tmp58 * (2 * FDPart3tmp151 * FDPart3tmp160 + FDPart3tmp349) + FDPart3tmp58 * (2 * FDPart3tmp159 * FDPart3tmp251 + FDPart3tmp339) -
            FDPart3tmp59 *
                (FDPart3tmp154 * hDD_dD221 - 2 * FDPart3tmp155 * FDPart3tmp44 + FDPart3tmp2 * FDPart3tmp22 * hDD_dD220 - FDPart3tmp228 * FDPart3tmp9 +
                 FDPart3tmp363 + FDPart3tmp7 * hDD_dDD2211 + hDD22 * (FDPart3tmp153 * FDPart3tmp227 + FDPart3tmp153 * f1_of_xx1 * f1_of_xx1__DD11)) +
            FDPart3tmp79 * (2 * FDPart3tmp173 * FDPart3tmp180 + FDPart3tmp180 * FDPart3tmp181) +
            FDPart3tmp79 * (FDPart3tmp179 * FDPart3tmp270 + FDPart3tmp268 * FDPart3tmp383) -
            FDPart3tmp80 * (FDPart3tmp102 * FDPart3tmp75 + FDPart3tmp22 * ((f1_of_xx1) * (f1_of_xx1) * (f1_of_xx1) * (f1_of_xx1)) * hDD_dD220 +
                            FDPart3tmp231 * FDPart3tmp379 + FDPart3tmp289 * f1_of_xx1__D1 * hDD_dD221 + 2 * FDPart3tmp289 * hDD_dD022 +
                            FDPart3tmp292 * hDD_dD122 + FDPart3tmp7 * hDD_dDD2222);

      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    }   // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  }     // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}

void Ricci_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, REAL *restrict auxevol_gfs) {
#include "../set_CodeParameters.h"
  int threads_in_x_dir = 32; //MIN(GPU_THREADX_MAX, params->Nxx0 / 32);
  int threads_in_y_dir = 2U * NGHOSTS; //MIN(GPU_THREADX_MAX / threads_in_x_dir, params->Nxx1);
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
  Ricci_eval__rfm__Spherical_gpu<<<grid_blocks, block_threads>>>(rfmstruct->f0_of_xx0, rfmstruct->f1_of_xx1, 
    rfmstruct->f1_of_xx1__D1, rfmstruct->f1_of_xx1__DD11, in_gfs, auxevol_gfs);
}
