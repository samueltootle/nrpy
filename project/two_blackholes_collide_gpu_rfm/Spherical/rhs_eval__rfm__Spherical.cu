#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
#include <functional>

__device__ REAL return_zero(const REAL&,const REAL&, const REAL&) { return 0.; }
__device__ REAL upwind_eval_func(const REAL& upwind, const REAL& a, const REAL& b) {
    return upwind * (-a + b) + a;
}

struct Upwind_eval_base {
    
    __device__  Upwind_eval_base () : f_(nullptr), upwind_(0.) {}
    __device__  Upwind_eval_base (const REAL & upwind) : upwind_(upwind) {
        if(upwind <= 0) {
            f_ = &return_zero;
        } else {
            f_ = &upwind_eval_func;
        }
    }
    __device__ REAL operator() (const REAL& a, const REAL& b) {
        return f_(upwind_, a, b);
    }

    private:
    REAL (*f_)(const REAL&, const REAL&, const REAL&);
    const REAL upwind_;
};

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
 * Finite difference function for operator ddnD0, with FD accuracy order 4.
 */
__device__ REAL fd_function_ddnD0_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0m3,
                                       const REAL FDPROTO_i0p1, const REAL invdxx0) {

  const REAL FD_result = invdxx0 * (FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i0m1 * FDPart1_Rational_3_2 + FDPROTO_i0m2 * FDPart1_Rational_1_2 -
                                    FDPROTO_i0m3 * FDPart1_Rational_1_12 + FDPROTO_i0p1 * FDPart1_Rational_1_4);

  return FD_result;
}
/*
 * Finite difference function for operator ddnD1, with FD accuracy order 4.
 */
__device__ REAL fd_function_ddnD1_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1m3,
                                       const REAL FDPROTO_i1p1, const REAL invdxx1) {

  const REAL FD_result = invdxx1 * (FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i1m1 * FDPart1_Rational_3_2 + FDPROTO_i1m2 * FDPart1_Rational_1_2 -
                                    FDPROTO_i1m3 * FDPart1_Rational_1_12 + FDPROTO_i1p1 * FDPart1_Rational_1_4);

  return FD_result;
}
/*
 * Finite difference function for operator ddnD2, with FD accuracy order 4.
 */
__device__ REAL fd_function_ddnD2_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2m3,
                                       const REAL FDPROTO_i2p1, const REAL invdxx2) {

  const REAL FD_result = invdxx2 * (FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i2m1 * FDPart1_Rational_3_2 + FDPROTO_i2m2 * FDPart1_Rational_1_2 -
                                    FDPROTO_i2m3 * FDPart1_Rational_1_12 + FDPROTO_i2p1 * FDPart1_Rational_1_4);

  return FD_result;
}
/*
 * Finite difference function for operator dupD0, with FD accuracy order 4.
 */
__device__ REAL fd_function_dupD0_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i0m1, const REAL FDPROTO_i0p1, const REAL FDPROTO_i0p2,
                                       const REAL FDPROTO_i0p3, const REAL invdxx0) {

  const REAL FD_result = invdxx0 * (-FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i0m1 * FDPart1_Rational_1_4 + FDPROTO_i0p1 * FDPart1_Rational_3_2 -
                                    FDPROTO_i0p2 * FDPart1_Rational_1_2 + FDPROTO_i0p3 * FDPart1_Rational_1_12);

  return FD_result;
}
/*
 * Finite difference function for operator dupD1, with FD accuracy order 4.
 */
__device__ REAL fd_function_dupD1_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i1m1, const REAL FDPROTO_i1p1, const REAL FDPROTO_i1p2,
                                       const REAL FDPROTO_i1p3, const REAL invdxx1) {
  const REAL FD_result = invdxx1 * (-FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i1m1 * FDPart1_Rational_1_4 + FDPROTO_i1p1 * FDPart1_Rational_3_2 -
                                    FDPROTO_i1p2 * FDPart1_Rational_1_2 + FDPROTO_i1p3 * FDPart1_Rational_1_12);

  return FD_result;
}
/*
 * Finite difference function for operator dupD2, with FD accuracy order 4.
 */
__device__ REAL fd_function_dupD2_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i2m1, const REAL FDPROTO_i2p1, const REAL FDPROTO_i2p2,
                                       const REAL FDPROTO_i2p3, const REAL invdxx2) {
  const REAL FD_result = invdxx2 * (-FDPROTO * FDPart1_Rational_5_6 - FDPROTO_i2m1 * FDPart1_Rational_1_4 + FDPROTO_i2p1 * FDPart1_Rational_3_2 -
                                    FDPROTO_i2p2 * FDPart1_Rational_1_2 + FDPROTO_i2p3 * FDPart1_Rational_1_12);

  return FD_result;
}
/*
 * Set RHSs for the BSSN evolution equations.
 */
__global__
void rhs_eval__rfm__Spherical_gpu(const REAL eta, const REAL *restrict _f0_of_xx0, const REAL *restrict _f1_of_xx1, 
    const REAL *restrict _f1_of_xx1__D1, const REAL *restrict _f1_of_xx1__DD11, 
        const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
  const REAL & invdxx0 = d_params.invdxx0;
  const REAL & invdxx1 = d_params.invdxx1;
  const REAL & invdxx2 = d_params.invdxx2;

  const int & Nxx0 = d_params.Nxx0;
  const int & Nxx1 = d_params.Nxx1;
  const int & Nxx2 = d_params.Nxx2;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  const int tid0  = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1  = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2  = blockIdx.z * blockDim.z + threadIdx.z;
  
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
         * NRPy+-Generated GF Access/FD Code, Step 1 of 3:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL RbarDD00 = auxevol_gfs[IDX4(RBARDD00GF, i0, i1, i2)];
        const REAL RbarDD01 = auxevol_gfs[IDX4(RBARDD01GF, i0, i1, i2)];
        const REAL RbarDD02 = auxevol_gfs[IDX4(RBARDD02GF, i0, i1, i2)];
        const REAL RbarDD11 = auxevol_gfs[IDX4(RBARDD11GF, i0, i1, i2)];
        const REAL RbarDD12 = auxevol_gfs[IDX4(RBARDD12GF, i0, i1, i2)];
        const REAL RbarDD22 = auxevol_gfs[IDX4(RBARDD22GF, i0, i1, i2)];
        const REAL aDD00_i2m3 = in_gfs[IDX4(ADD00GF, i0, i1, i2 - 3)];
        const REAL aDD00_i2m2 = in_gfs[IDX4(ADD00GF, i0, i1, i2 - 2)];
        const REAL aDD00_i2m1 = in_gfs[IDX4(ADD00GF, i0, i1, i2 - 1)];
        const REAL aDD00_i1m3 = in_gfs[IDX4(ADD00GF, i0, i1 - 3, i2)];
        const REAL aDD00_i1m2 = in_gfs[IDX4(ADD00GF, i0, i1 - 2, i2)];
        const REAL aDD00_i1m1 = in_gfs[IDX4(ADD00GF, i0, i1 - 1, i2)];
        const REAL aDD00_i0m3 = in_gfs[IDX4(ADD00GF, i0 - 3, i1, i2)];
        const REAL aDD00_i0m2 = in_gfs[IDX4(ADD00GF, i0 - 2, i1, i2)];
        const REAL aDD00_i0m1 = in_gfs[IDX4(ADD00GF, i0 - 1, i1, i2)];
        const REAL aDD00 = in_gfs[IDX4(ADD00GF, i0, i1, i2)];
        const REAL aDD00_i0p1 = in_gfs[IDX4(ADD00GF, i0 + 1, i1, i2)];
        const REAL aDD00_i0p2 = in_gfs[IDX4(ADD00GF, i0 + 2, i1, i2)];
        const REAL aDD00_i0p3 = in_gfs[IDX4(ADD00GF, i0 + 3, i1, i2)];
        const REAL aDD00_i1p1 = in_gfs[IDX4(ADD00GF, i0, i1 + 1, i2)];
        const REAL aDD00_i1p2 = in_gfs[IDX4(ADD00GF, i0, i1 + 2, i2)];
        const REAL aDD00_i1p3 = in_gfs[IDX4(ADD00GF, i0, i1 + 3, i2)];
        const REAL aDD00_i2p1 = in_gfs[IDX4(ADD00GF, i0, i1, i2 + 1)];
        const REAL aDD00_i2p2 = in_gfs[IDX4(ADD00GF, i0, i1, i2 + 2)];
        const REAL aDD00_i2p3 = in_gfs[IDX4(ADD00GF, i0, i1, i2 + 3)];
        const REAL aDD01_i2m3 = in_gfs[IDX4(ADD01GF, i0, i1, i2 - 3)];
        const REAL aDD01_i2m2 = in_gfs[IDX4(ADD01GF, i0, i1, i2 - 2)];
        const REAL aDD01_i2m1 = in_gfs[IDX4(ADD01GF, i0, i1, i2 - 1)];
        const REAL aDD01_i1m3 = in_gfs[IDX4(ADD01GF, i0, i1 - 3, i2)];
        const REAL aDD01_i1m2 = in_gfs[IDX4(ADD01GF, i0, i1 - 2, i2)];
        const REAL aDD01_i1m1 = in_gfs[IDX4(ADD01GF, i0, i1 - 1, i2)];
        const REAL aDD01_i0m3 = in_gfs[IDX4(ADD01GF, i0 - 3, i1, i2)];
        const REAL aDD01_i0m2 = in_gfs[IDX4(ADD01GF, i0 - 2, i1, i2)];
        const REAL aDD01_i0m1 = in_gfs[IDX4(ADD01GF, i0 - 1, i1, i2)];
        const REAL aDD01 = in_gfs[IDX4(ADD01GF, i0, i1, i2)];
        const REAL aDD01_i0p1 = in_gfs[IDX4(ADD01GF, i0 + 1, i1, i2)];
        const REAL aDD01_i0p2 = in_gfs[IDX4(ADD01GF, i0 + 2, i1, i2)];
        const REAL aDD01_i0p3 = in_gfs[IDX4(ADD01GF, i0 + 3, i1, i2)];
        const REAL aDD01_i1p1 = in_gfs[IDX4(ADD01GF, i0, i1 + 1, i2)];
        const REAL aDD01_i1p2 = in_gfs[IDX4(ADD01GF, i0, i1 + 2, i2)];
        const REAL aDD01_i1p3 = in_gfs[IDX4(ADD01GF, i0, i1 + 3, i2)];
        const REAL aDD01_i2p1 = in_gfs[IDX4(ADD01GF, i0, i1, i2 + 1)];
        const REAL aDD01_i2p2 = in_gfs[IDX4(ADD01GF, i0, i1, i2 + 2)];
        const REAL aDD01_i2p3 = in_gfs[IDX4(ADD01GF, i0, i1, i2 + 3)];
        const REAL aDD02_i2m3 = in_gfs[IDX4(ADD02GF, i0, i1, i2 - 3)];
        const REAL aDD02_i2m2 = in_gfs[IDX4(ADD02GF, i0, i1, i2 - 2)];
        const REAL aDD02_i2m1 = in_gfs[IDX4(ADD02GF, i0, i1, i2 - 1)];
        const REAL aDD02_i1m3 = in_gfs[IDX4(ADD02GF, i0, i1 - 3, i2)];
        const REAL aDD02_i1m2 = in_gfs[IDX4(ADD02GF, i0, i1 - 2, i2)];
        const REAL aDD02_i1m1 = in_gfs[IDX4(ADD02GF, i0, i1 - 1, i2)];
        const REAL aDD02_i0m3 = in_gfs[IDX4(ADD02GF, i0 - 3, i1, i2)];
        const REAL aDD02_i0m2 = in_gfs[IDX4(ADD02GF, i0 - 2, i1, i2)];
        const REAL aDD02_i0m1 = in_gfs[IDX4(ADD02GF, i0 - 1, i1, i2)];
        const REAL aDD02 = in_gfs[IDX4(ADD02GF, i0, i1, i2)];
        const REAL aDD02_i0p1 = in_gfs[IDX4(ADD02GF, i0 + 1, i1, i2)];
        const REAL aDD02_i0p2 = in_gfs[IDX4(ADD02GF, i0 + 2, i1, i2)];
        const REAL aDD02_i0p3 = in_gfs[IDX4(ADD02GF, i0 + 3, i1, i2)];
        const REAL aDD02_i1p1 = in_gfs[IDX4(ADD02GF, i0, i1 + 1, i2)];
        const REAL aDD02_i1p2 = in_gfs[IDX4(ADD02GF, i0, i1 + 2, i2)];
        const REAL aDD02_i1p3 = in_gfs[IDX4(ADD02GF, i0, i1 + 3, i2)];
        const REAL aDD02_i2p1 = in_gfs[IDX4(ADD02GF, i0, i1, i2 + 1)];
        const REAL aDD02_i2p2 = in_gfs[IDX4(ADD02GF, i0, i1, i2 + 2)];
        const REAL aDD02_i2p3 = in_gfs[IDX4(ADD02GF, i0, i1, i2 + 3)];
        const REAL aDD11_i2m3 = in_gfs[IDX4(ADD11GF, i0, i1, i2 - 3)];
        const REAL aDD11_i2m2 = in_gfs[IDX4(ADD11GF, i0, i1, i2 - 2)];
        const REAL aDD11_i2m1 = in_gfs[IDX4(ADD11GF, i0, i1, i2 - 1)];
        const REAL aDD11_i1m3 = in_gfs[IDX4(ADD11GF, i0, i1 - 3, i2)];
        const REAL aDD11_i1m2 = in_gfs[IDX4(ADD11GF, i0, i1 - 2, i2)];
        const REAL aDD11_i1m1 = in_gfs[IDX4(ADD11GF, i0, i1 - 1, i2)];
        const REAL aDD11_i0m3 = in_gfs[IDX4(ADD11GF, i0 - 3, i1, i2)];
        const REAL aDD11_i0m2 = in_gfs[IDX4(ADD11GF, i0 - 2, i1, i2)];
        const REAL aDD11_i0m1 = in_gfs[IDX4(ADD11GF, i0 - 1, i1, i2)];
        const REAL aDD11 = in_gfs[IDX4(ADD11GF, i0, i1, i2)];
        const REAL aDD11_i0p1 = in_gfs[IDX4(ADD11GF, i0 + 1, i1, i2)];
        const REAL aDD11_i0p2 = in_gfs[IDX4(ADD11GF, i0 + 2, i1, i2)];
        const REAL aDD11_i0p3 = in_gfs[IDX4(ADD11GF, i0 + 3, i1, i2)];
        const REAL aDD11_i1p1 = in_gfs[IDX4(ADD11GF, i0, i1 + 1, i2)];
        const REAL aDD11_i1p2 = in_gfs[IDX4(ADD11GF, i0, i1 + 2, i2)];
        const REAL aDD11_i1p3 = in_gfs[IDX4(ADD11GF, i0, i1 + 3, i2)];
        const REAL aDD11_i2p1 = in_gfs[IDX4(ADD11GF, i0, i1, i2 + 1)];
        const REAL aDD11_i2p2 = in_gfs[IDX4(ADD11GF, i0, i1, i2 + 2)];
        const REAL aDD11_i2p3 = in_gfs[IDX4(ADD11GF, i0, i1, i2 + 3)];
        const REAL aDD12_i2m3 = in_gfs[IDX4(ADD12GF, i0, i1, i2 - 3)];
        const REAL aDD12_i2m2 = in_gfs[IDX4(ADD12GF, i0, i1, i2 - 2)];
        const REAL aDD12_i2m1 = in_gfs[IDX4(ADD12GF, i0, i1, i2 - 1)];
        const REAL aDD12_i1m3 = in_gfs[IDX4(ADD12GF, i0, i1 - 3, i2)];
        const REAL aDD12_i1m2 = in_gfs[IDX4(ADD12GF, i0, i1 - 2, i2)];
        const REAL aDD12_i1m1 = in_gfs[IDX4(ADD12GF, i0, i1 - 1, i2)];
        const REAL aDD12_i0m3 = in_gfs[IDX4(ADD12GF, i0 - 3, i1, i2)];
        const REAL aDD12_i0m2 = in_gfs[IDX4(ADD12GF, i0 - 2, i1, i2)];
        const REAL aDD12_i0m1 = in_gfs[IDX4(ADD12GF, i0 - 1, i1, i2)];
        const REAL aDD12 = in_gfs[IDX4(ADD12GF, i0, i1, i2)];
        const REAL aDD12_i0p1 = in_gfs[IDX4(ADD12GF, i0 + 1, i1, i2)];
        const REAL aDD12_i0p2 = in_gfs[IDX4(ADD12GF, i0 + 2, i1, i2)];
        const REAL aDD12_i0p3 = in_gfs[IDX4(ADD12GF, i0 + 3, i1, i2)];
        const REAL aDD12_i1p1 = in_gfs[IDX4(ADD12GF, i0, i1 + 1, i2)];
        const REAL aDD12_i1p2 = in_gfs[IDX4(ADD12GF, i0, i1 + 2, i2)];
        const REAL aDD12_i1p3 = in_gfs[IDX4(ADD12GF, i0, i1 + 3, i2)];
        const REAL aDD12_i2p1 = in_gfs[IDX4(ADD12GF, i0, i1, i2 + 1)];
        const REAL aDD12_i2p2 = in_gfs[IDX4(ADD12GF, i0, i1, i2 + 2)];
        const REAL aDD12_i2p3 = in_gfs[IDX4(ADD12GF, i0, i1, i2 + 3)];
        const REAL aDD22_i2m3 = in_gfs[IDX4(ADD22GF, i0, i1, i2 - 3)];
        const REAL aDD22_i2m2 = in_gfs[IDX4(ADD22GF, i0, i1, i2 - 2)];
        const REAL aDD22_i2m1 = in_gfs[IDX4(ADD22GF, i0, i1, i2 - 1)];
        const REAL aDD22_i1m3 = in_gfs[IDX4(ADD22GF, i0, i1 - 3, i2)];
        const REAL aDD22_i1m2 = in_gfs[IDX4(ADD22GF, i0, i1 - 2, i2)];
        const REAL aDD22_i1m1 = in_gfs[IDX4(ADD22GF, i0, i1 - 1, i2)];
        const REAL aDD22_i0m3 = in_gfs[IDX4(ADD22GF, i0 - 3, i1, i2)];
        const REAL aDD22_i0m2 = in_gfs[IDX4(ADD22GF, i0 - 2, i1, i2)];
        const REAL aDD22_i0m1 = in_gfs[IDX4(ADD22GF, i0 - 1, i1, i2)];
        const REAL aDD22 = in_gfs[IDX4(ADD22GF, i0, i1, i2)];
        const REAL aDD22_i0p1 = in_gfs[IDX4(ADD22GF, i0 + 1, i1, i2)];
        const REAL aDD22_i0p2 = in_gfs[IDX4(ADD22GF, i0 + 2, i1, i2)];
        const REAL aDD22_i0p3 = in_gfs[IDX4(ADD22GF, i0 + 3, i1, i2)];
        const REAL aDD22_i1p1 = in_gfs[IDX4(ADD22GF, i0, i1 + 1, i2)];
        const REAL aDD22_i1p2 = in_gfs[IDX4(ADD22GF, i0, i1 + 2, i2)];
        const REAL aDD22_i1p3 = in_gfs[IDX4(ADD22GF, i0, i1 + 3, i2)];
        const REAL aDD22_i2p1 = in_gfs[IDX4(ADD22GF, i0, i1, i2 + 1)];
        const REAL aDD22_i2p2 = in_gfs[IDX4(ADD22GF, i0, i1, i2 + 2)];
        const REAL aDD22_i2p3 = in_gfs[IDX4(ADD22GF, i0, i1, i2 + 3)];
        const REAL alpha_i2m3 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 - 3)];
        const REAL alpha_i1m2_i2m2 = in_gfs[IDX4(ALPHAGF, i0, i1 - 2, i2 - 2)];
        const REAL alpha_i1m1_i2m2 = in_gfs[IDX4(ALPHAGF, i0, i1 - 1, i2 - 2)];
        const REAL alpha_i0m2_i2m2 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1, i2 - 2)];
        const REAL alpha_i0m1_i2m2 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1, i2 - 2)];
        const REAL alpha_i2m2 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 - 2)];
        const REAL alpha_i0p1_i2m2 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1, i2 - 2)];
        const REAL alpha_i0p2_i2m2 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1, i2 - 2)];
        const REAL alpha_i1p1_i2m2 = in_gfs[IDX4(ALPHAGF, i0, i1 + 1, i2 - 2)];
        const REAL alpha_i1p2_i2m2 = in_gfs[IDX4(ALPHAGF, i0, i1 + 2, i2 - 2)];
        const REAL alpha_i1m2_i2m1 = in_gfs[IDX4(ALPHAGF, i0, i1 - 2, i2 - 1)];
        const REAL alpha_i1m1_i2m1 = in_gfs[IDX4(ALPHAGF, i0, i1 - 1, i2 - 1)];
        const REAL alpha_i0m2_i2m1 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1, i2 - 1)];
        const REAL alpha_i0m1_i2m1 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1, i2 - 1)];
        const REAL alpha_i2m1 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 - 1)];
        const REAL alpha_i0p1_i2m1 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1, i2 - 1)];
        const REAL alpha_i0p2_i2m1 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1, i2 - 1)];
        const REAL alpha_i1p1_i2m1 = in_gfs[IDX4(ALPHAGF, i0, i1 + 1, i2 - 1)];
        const REAL alpha_i1p2_i2m1 = in_gfs[IDX4(ALPHAGF, i0, i1 + 2, i2 - 1)];
        const REAL alpha_i1m3 = in_gfs[IDX4(ALPHAGF, i0, i1 - 3, i2)];
        const REAL alpha_i0m2_i1m2 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1 - 2, i2)];
        const REAL alpha_i0m1_i1m2 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1 - 2, i2)];
        const REAL alpha_i1m2 = in_gfs[IDX4(ALPHAGF, i0, i1 - 2, i2)];
        const REAL alpha_i0p1_i1m2 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1 - 2, i2)];
        const REAL alpha_i0p2_i1m2 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1 - 2, i2)];
        const REAL alpha_i0m2_i1m1 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1 - 1, i2)];
        const REAL alpha_i0m1_i1m1 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1 - 1, i2)];
        const REAL alpha_i1m1 = in_gfs[IDX4(ALPHAGF, i0, i1 - 1, i2)];
        const REAL alpha_i0p1_i1m1 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1 - 1, i2)];
        const REAL alpha_i0p2_i1m1 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1 - 1, i2)];
        const REAL alpha_i0m3 = in_gfs[IDX4(ALPHAGF, i0 - 3, i1, i2)];
        const REAL alpha_i0m2 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1, i2)];
        const REAL alpha_i0m1 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1, i2)];
        const REAL alpha = in_gfs[IDX4(ALPHAGF, i0, i1, i2)];
        const REAL alpha_i0p1 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1, i2)];
        const REAL alpha_i0p2 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1, i2)];
        const REAL alpha_i0p3 = in_gfs[IDX4(ALPHAGF, i0 + 3, i1, i2)];
        const REAL alpha_i0m2_i1p1 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1 + 1, i2)];
        const REAL alpha_i0m1_i1p1 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1 + 1, i2)];
        const REAL alpha_i1p1 = in_gfs[IDX4(ALPHAGF, i0, i1 + 1, i2)];
        const REAL alpha_i0p1_i1p1 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1 + 1, i2)];
        const REAL alpha_i0p2_i1p1 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1 + 1, i2)];
        const REAL alpha_i0m2_i1p2 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1 + 2, i2)];
        const REAL alpha_i0m1_i1p2 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1 + 2, i2)];
        const REAL alpha_i1p2 = in_gfs[IDX4(ALPHAGF, i0, i1 + 2, i2)];
        const REAL alpha_i0p1_i1p2 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1 + 2, i2)];
        const REAL alpha_i0p2_i1p2 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1 + 2, i2)];
        const REAL alpha_i1p3 = in_gfs[IDX4(ALPHAGF, i0, i1 + 3, i2)];
        const REAL alpha_i1m2_i2p1 = in_gfs[IDX4(ALPHAGF, i0, i1 - 2, i2 + 1)];
        const REAL alpha_i1m1_i2p1 = in_gfs[IDX4(ALPHAGF, i0, i1 - 1, i2 + 1)];
        const REAL alpha_i0m2_i2p1 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1, i2 + 1)];
        const REAL alpha_i0m1_i2p1 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1, i2 + 1)];
        const REAL alpha_i2p1 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 + 1)];
        const REAL alpha_i0p1_i2p1 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1, i2 + 1)];
        const REAL alpha_i0p2_i2p1 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1, i2 + 1)];
        const REAL alpha_i1p1_i2p1 = in_gfs[IDX4(ALPHAGF, i0, i1 + 1, i2 + 1)];
        const REAL alpha_i1p2_i2p1 = in_gfs[IDX4(ALPHAGF, i0, i1 + 2, i2 + 1)];
        const REAL alpha_i1m2_i2p2 = in_gfs[IDX4(ALPHAGF, i0, i1 - 2, i2 + 2)];
        const REAL alpha_i1m1_i2p2 = in_gfs[IDX4(ALPHAGF, i0, i1 - 1, i2 + 2)];
        const REAL alpha_i0m2_i2p2 = in_gfs[IDX4(ALPHAGF, i0 - 2, i1, i2 + 2)];
        const REAL alpha_i0m1_i2p2 = in_gfs[IDX4(ALPHAGF, i0 - 1, i1, i2 + 2)];
        const REAL alpha_i2p2 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 + 2)];
        const REAL alpha_i0p1_i2p2 = in_gfs[IDX4(ALPHAGF, i0 + 1, i1, i2 + 2)];
        const REAL alpha_i0p2_i2p2 = in_gfs[IDX4(ALPHAGF, i0 + 2, i1, i2 + 2)];
        const REAL alpha_i1p1_i2p2 = in_gfs[IDX4(ALPHAGF, i0, i1 + 1, i2 + 2)];
        const REAL alpha_i1p2_i2p2 = in_gfs[IDX4(ALPHAGF, i0, i1 + 2, i2 + 2)];
        const REAL alpha_i2p3 = in_gfs[IDX4(ALPHAGF, i0, i1, i2 + 3)];
        const REAL betU0_i2m3 = in_gfs[IDX4(BETU0GF, i0, i1, i2 - 3)];
        const REAL betU0_i2m2 = in_gfs[IDX4(BETU0GF, i0, i1, i2 - 2)];
        const REAL betU0_i2m1 = in_gfs[IDX4(BETU0GF, i0, i1, i2 - 1)];
        const REAL betU0_i1m3 = in_gfs[IDX4(BETU0GF, i0, i1 - 3, i2)];
        const REAL betU0_i1m2 = in_gfs[IDX4(BETU0GF, i0, i1 - 2, i2)];
        const REAL betU0_i1m1 = in_gfs[IDX4(BETU0GF, i0, i1 - 1, i2)];
        const REAL betU0_i0m3 = in_gfs[IDX4(BETU0GF, i0 - 3, i1, i2)];
        const REAL betU0_i0m2 = in_gfs[IDX4(BETU0GF, i0 - 2, i1, i2)];
        const REAL betU0_i0m1 = in_gfs[IDX4(BETU0GF, i0 - 1, i1, i2)];
        const REAL betU0 = in_gfs[IDX4(BETU0GF, i0, i1, i2)];
        const REAL betU0_i0p1 = in_gfs[IDX4(BETU0GF, i0 + 1, i1, i2)];
        const REAL betU0_i0p2 = in_gfs[IDX4(BETU0GF, i0 + 2, i1, i2)];
        const REAL betU0_i0p3 = in_gfs[IDX4(BETU0GF, i0 + 3, i1, i2)];
        const REAL betU0_i1p1 = in_gfs[IDX4(BETU0GF, i0, i1 + 1, i2)];
        const REAL betU0_i1p2 = in_gfs[IDX4(BETU0GF, i0, i1 + 2, i2)];
        const REAL betU0_i1p3 = in_gfs[IDX4(BETU0GF, i0, i1 + 3, i2)];
        const REAL betU0_i2p1 = in_gfs[IDX4(BETU0GF, i0, i1, i2 + 1)];
        const REAL betU0_i2p2 = in_gfs[IDX4(BETU0GF, i0, i1, i2 + 2)];
        const REAL betU0_i2p3 = in_gfs[IDX4(BETU0GF, i0, i1, i2 + 3)];
        const REAL betU1_i2m3 = in_gfs[IDX4(BETU1GF, i0, i1, i2 - 3)];
        const REAL betU1_i2m2 = in_gfs[IDX4(BETU1GF, i0, i1, i2 - 2)];
        const REAL betU1_i2m1 = in_gfs[IDX4(BETU1GF, i0, i1, i2 - 1)];
        const REAL betU1_i1m3 = in_gfs[IDX4(BETU1GF, i0, i1 - 3, i2)];
        const REAL betU1_i1m2 = in_gfs[IDX4(BETU1GF, i0, i1 - 2, i2)];
        const REAL betU1_i1m1 = in_gfs[IDX4(BETU1GF, i0, i1 - 1, i2)];
        const REAL betU1_i0m3 = in_gfs[IDX4(BETU1GF, i0 - 3, i1, i2)];
        const REAL betU1_i0m2 = in_gfs[IDX4(BETU1GF, i0 - 2, i1, i2)];
        const REAL betU1_i0m1 = in_gfs[IDX4(BETU1GF, i0 - 1, i1, i2)];
        const REAL betU1 = in_gfs[IDX4(BETU1GF, i0, i1, i2)];
        const REAL betU1_i0p1 = in_gfs[IDX4(BETU1GF, i0 + 1, i1, i2)];
        const REAL betU1_i0p2 = in_gfs[IDX4(BETU1GF, i0 + 2, i1, i2)];
        const REAL betU1_i0p3 = in_gfs[IDX4(BETU1GF, i0 + 3, i1, i2)];
        const REAL betU1_i1p1 = in_gfs[IDX4(BETU1GF, i0, i1 + 1, i2)];
        const REAL betU1_i1p2 = in_gfs[IDX4(BETU1GF, i0, i1 + 2, i2)];
        const REAL betU1_i1p3 = in_gfs[IDX4(BETU1GF, i0, i1 + 3, i2)];
        const REAL betU1_i2p1 = in_gfs[IDX4(BETU1GF, i0, i1, i2 + 1)];
        const REAL betU1_i2p2 = in_gfs[IDX4(BETU1GF, i0, i1, i2 + 2)];
        const REAL betU1_i2p3 = in_gfs[IDX4(BETU1GF, i0, i1, i2 + 3)];
        const REAL betU2_i2m3 = in_gfs[IDX4(BETU2GF, i0, i1, i2 - 3)];
        const REAL betU2_i2m2 = in_gfs[IDX4(BETU2GF, i0, i1, i2 - 2)];
        const REAL betU2_i2m1 = in_gfs[IDX4(BETU2GF, i0, i1, i2 - 1)];
        const REAL betU2_i1m3 = in_gfs[IDX4(BETU2GF, i0, i1 - 3, i2)];
        const REAL betU2_i1m2 = in_gfs[IDX4(BETU2GF, i0, i1 - 2, i2)];
        const REAL betU2_i1m1 = in_gfs[IDX4(BETU2GF, i0, i1 - 1, i2)];
        const REAL betU2_i0m3 = in_gfs[IDX4(BETU2GF, i0 - 3, i1, i2)];
        const REAL betU2_i0m2 = in_gfs[IDX4(BETU2GF, i0 - 2, i1, i2)];
        const REAL betU2_i0m1 = in_gfs[IDX4(BETU2GF, i0 - 1, i1, i2)];
        const REAL betU2 = in_gfs[IDX4(BETU2GF, i0, i1, i2)];
        const REAL betU2_i0p1 = in_gfs[IDX4(BETU2GF, i0 + 1, i1, i2)];
        const REAL betU2_i0p2 = in_gfs[IDX4(BETU2GF, i0 + 2, i1, i2)];
        const REAL betU2_i0p3 = in_gfs[IDX4(BETU2GF, i0 + 3, i1, i2)];
        const REAL betU2_i1p1 = in_gfs[IDX4(BETU2GF, i0, i1 + 1, i2)];
        const REAL betU2_i1p2 = in_gfs[IDX4(BETU2GF, i0, i1 + 2, i2)];
        const REAL betU2_i1p3 = in_gfs[IDX4(BETU2GF, i0, i1 + 3, i2)];
        const REAL betU2_i2p1 = in_gfs[IDX4(BETU2GF, i0, i1, i2 + 1)];
        const REAL betU2_i2p2 = in_gfs[IDX4(BETU2GF, i0, i1, i2 + 2)];
        const REAL betU2_i2p3 = in_gfs[IDX4(BETU2GF, i0, i1, i2 + 3)];
        const REAL cf_i2m3 = in_gfs[IDX4(CFGF, i0, i1, i2 - 3)];
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
        const REAL cf_i1m3 = in_gfs[IDX4(CFGF, i0, i1 - 3, i2)];
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
        const REAL cf_i0m3 = in_gfs[IDX4(CFGF, i0 - 3, i1, i2)];
        const REAL cf_i0m2 = in_gfs[IDX4(CFGF, i0 - 2, i1, i2)];
        const REAL cf_i0m1 = in_gfs[IDX4(CFGF, i0 - 1, i1, i2)];
        const REAL cf = in_gfs[IDX4(CFGF, i0, i1, i2)];
        const REAL cf_i0p1 = in_gfs[IDX4(CFGF, i0 + 1, i1, i2)];
        const REAL cf_i0p2 = in_gfs[IDX4(CFGF, i0 + 2, i1, i2)];
        const REAL cf_i0p3 = in_gfs[IDX4(CFGF, i0 + 3, i1, i2)];
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
        const REAL cf_i1p3 = in_gfs[IDX4(CFGF, i0, i1 + 3, i2)];
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
        const REAL cf_i2p3 = in_gfs[IDX4(CFGF, i0, i1, i2 + 3)];
        const REAL hDD00_i2m3 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 3)];
        const REAL hDD00_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 2)];
        const REAL hDD00_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 1)];
        const REAL hDD00_i1m3 = in_gfs[IDX4(HDD00GF, i0, i1 - 3, i2)];
        const REAL hDD00_i1m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2)];
        const REAL hDD00_i1m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2)];
        const REAL hDD00_i0m3 = in_gfs[IDX4(HDD00GF, i0 - 3, i1, i2)];
        const REAL hDD00_i0m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2)];
        const REAL hDD00_i0m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2)];
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD00_i0p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2)];
        const REAL hDD00_i0p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2)];
        const REAL hDD00_i0p3 = in_gfs[IDX4(HDD00GF, i0 + 3, i1, i2)];
        const REAL hDD00_i1p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2)];
        const REAL hDD00_i1p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2)];
        const REAL hDD00_i1p3 = in_gfs[IDX4(HDD00GF, i0, i1 + 3, i2)];
        const REAL hDD00_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 1)];
        const REAL hDD00_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 2)];
        const REAL hDD00_i2p3 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 3)];
        const REAL hDD01_i2m3 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 3)];
        const REAL hDD01_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 2)];
        const REAL hDD01_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 1)];
        const REAL hDD01_i1m3 = in_gfs[IDX4(HDD01GF, i0, i1 - 3, i2)];
        const REAL hDD01_i1m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2)];
        const REAL hDD01_i1m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2)];
        const REAL hDD01_i0m3 = in_gfs[IDX4(HDD01GF, i0 - 3, i1, i2)];
        const REAL hDD01_i0m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2)];
        const REAL hDD01_i0m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD01_i0p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2)];
        const REAL hDD01_i0p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2)];
        const REAL hDD01_i0p3 = in_gfs[IDX4(HDD01GF, i0 + 3, i1, i2)];
        const REAL hDD01_i1p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2)];
        const REAL hDD01_i1p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2)];
        const REAL hDD01_i1p3 = in_gfs[IDX4(HDD01GF, i0, i1 + 3, i2)];
        const REAL hDD01_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 1)];
        const REAL hDD01_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 2)];
        const REAL hDD01_i2p3 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 3)];
        const REAL hDD02_i2m3 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 3)];
        const REAL hDD02_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 2)];
        const REAL hDD02_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 1)];
        const REAL hDD02_i1m3 = in_gfs[IDX4(HDD02GF, i0, i1 - 3, i2)];
        const REAL hDD02_i1m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2)];
        const REAL hDD02_i1m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2)];
        const REAL hDD02_i0m3 = in_gfs[IDX4(HDD02GF, i0 - 3, i1, i2)];
        const REAL hDD02_i0m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2)];
        const REAL hDD02_i0m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD02_i0p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2)];
        const REAL hDD02_i0p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2)];
        const REAL hDD02_i0p3 = in_gfs[IDX4(HDD02GF, i0 + 3, i1, i2)];
        const REAL hDD02_i1p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2)];
        const REAL hDD02_i1p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2)];
        const REAL hDD02_i1p3 = in_gfs[IDX4(HDD02GF, i0, i1 + 3, i2)];
        const REAL hDD02_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 1)];
        const REAL hDD02_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 2)];
        const REAL hDD02_i2p3 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 3)];
        const REAL hDD11_i2m3 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 3)];
        const REAL hDD11_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 2)];
        const REAL hDD11_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 1)];
        const REAL hDD11_i1m3 = in_gfs[IDX4(HDD11GF, i0, i1 - 3, i2)];
        const REAL hDD11_i1m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2)];
        const REAL hDD11_i1m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2)];
        const REAL hDD11_i0m3 = in_gfs[IDX4(HDD11GF, i0 - 3, i1, i2)];
        const REAL hDD11_i0m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2)];
        const REAL hDD11_i0m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD11_i0p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2)];
        const REAL hDD11_i0p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2)];
        const REAL hDD11_i0p3 = in_gfs[IDX4(HDD11GF, i0 + 3, i1, i2)];
        const REAL hDD11_i1p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2)];
        const REAL hDD11_i1p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2)];
        const REAL hDD11_i1p3 = in_gfs[IDX4(HDD11GF, i0, i1 + 3, i2)];
        const REAL hDD11_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 1)];
        const REAL hDD11_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 2)];
        const REAL hDD11_i2p3 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 3)];
        const REAL hDD12_i2m3 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 3)];
        const REAL hDD12_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 2)];
        const REAL hDD12_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 1)];
        const REAL hDD12_i1m3 = in_gfs[IDX4(HDD12GF, i0, i1 - 3, i2)];
        const REAL hDD12_i1m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2)];
        const REAL hDD12_i1m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2)];
        const REAL hDD12_i0m3 = in_gfs[IDX4(HDD12GF, i0 - 3, i1, i2)];
        const REAL hDD12_i0m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2)];
        const REAL hDD12_i0m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD12_i0p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2)];
        const REAL hDD12_i0p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2)];
        const REAL hDD12_i0p3 = in_gfs[IDX4(HDD12GF, i0 + 3, i1, i2)];
        const REAL hDD12_i1p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2)];
        const REAL hDD12_i1p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2)];
        const REAL hDD12_i1p3 = in_gfs[IDX4(HDD12GF, i0, i1 + 3, i2)];
        const REAL hDD12_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 1)];
        const REAL hDD12_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 2)];
        const REAL hDD12_i2p3 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 3)];
        const REAL hDD22_i2m3 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 3)];
        const REAL hDD22_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 2)];
        const REAL hDD22_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 1)];
        const REAL hDD22_i1m3 = in_gfs[IDX4(HDD22GF, i0, i1 - 3, i2)];
        const REAL hDD22_i1m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2)];
        const REAL hDD22_i1m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2)];
        const REAL hDD22_i0m3 = in_gfs[IDX4(HDD22GF, i0 - 3, i1, i2)];
        const REAL hDD22_i0m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2)];
        const REAL hDD22_i0m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];
        const REAL hDD22_i0p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2)];
        const REAL hDD22_i0p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2)];
        const REAL hDD22_i0p3 = in_gfs[IDX4(HDD22GF, i0 + 3, i1, i2)];
        const REAL hDD22_i1p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2)];
        const REAL hDD22_i1p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2)];
        const REAL hDD22_i1p3 = in_gfs[IDX4(HDD22GF, i0, i1 + 3, i2)];
        const REAL hDD22_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 1)];
        const REAL hDD22_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 2)];
        const REAL hDD22_i2p3 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 3)];
        const REAL lambdaU0_i2m3 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 - 3)];
        const REAL lambdaU0_i2m2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 - 2)];
        const REAL lambdaU0_i2m1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 - 1)];
        const REAL lambdaU0_i1m3 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 - 3, i2)];
        const REAL lambdaU0_i1m2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 - 2, i2)];
        const REAL lambdaU0_i1m1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 - 1, i2)];
        const REAL lambdaU0_i0m3 = in_gfs[IDX4(LAMBDAU0GF, i0 - 3, i1, i2)];
        const REAL lambdaU0_i0m2 = in_gfs[IDX4(LAMBDAU0GF, i0 - 2, i1, i2)];
        const REAL lambdaU0_i0m1 = in_gfs[IDX4(LAMBDAU0GF, i0 - 1, i1, i2)];
        const REAL lambdaU0 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2)];
        const REAL lambdaU0_i0p1 = in_gfs[IDX4(LAMBDAU0GF, i0 + 1, i1, i2)];
        const REAL lambdaU0_i0p2 = in_gfs[IDX4(LAMBDAU0GF, i0 + 2, i1, i2)];
        const REAL lambdaU0_i0p3 = in_gfs[IDX4(LAMBDAU0GF, i0 + 3, i1, i2)];
        const REAL lambdaU0_i1p1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 + 1, i2)];
        const REAL lambdaU0_i1p2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 + 2, i2)];
        const REAL lambdaU0_i1p3 = in_gfs[IDX4(LAMBDAU0GF, i0, i1 + 3, i2)];
        const REAL lambdaU0_i2p1 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 + 1)];
        const REAL lambdaU0_i2p2 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 + 2)];
        const REAL lambdaU0_i2p3 = in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2 + 3)];
        const REAL lambdaU1_i2m3 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 - 3)];
        const REAL lambdaU1_i2m2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 - 2)];
        const REAL lambdaU1_i2m1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 - 1)];
        const REAL lambdaU1_i1m3 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 - 3, i2)];
        const REAL lambdaU1_i1m2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 - 2, i2)];
        const REAL lambdaU1_i1m1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 - 1, i2)];
        const REAL lambdaU1_i0m3 = in_gfs[IDX4(LAMBDAU1GF, i0 - 3, i1, i2)];
        const REAL lambdaU1_i0m2 = in_gfs[IDX4(LAMBDAU1GF, i0 - 2, i1, i2)];
        const REAL lambdaU1_i0m1 = in_gfs[IDX4(LAMBDAU1GF, i0 - 1, i1, i2)];
        const REAL lambdaU1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2)];
        const REAL lambdaU1_i0p1 = in_gfs[IDX4(LAMBDAU1GF, i0 + 1, i1, i2)];
        const REAL lambdaU1_i0p2 = in_gfs[IDX4(LAMBDAU1GF, i0 + 2, i1, i2)];
        const REAL lambdaU1_i0p3 = in_gfs[IDX4(LAMBDAU1GF, i0 + 3, i1, i2)];
        const REAL lambdaU1_i1p1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 + 1, i2)];
        const REAL lambdaU1_i1p2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 + 2, i2)];
        const REAL lambdaU1_i1p3 = in_gfs[IDX4(LAMBDAU1GF, i0, i1 + 3, i2)];
        const REAL lambdaU1_i2p1 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 + 1)];
        const REAL lambdaU1_i2p2 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 + 2)];
        const REAL lambdaU1_i2p3 = in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2 + 3)];
        const REAL lambdaU2_i2m3 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 - 3)];
        const REAL lambdaU2_i2m2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 - 2)];
        const REAL lambdaU2_i2m1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 - 1)];
        const REAL lambdaU2_i1m3 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 - 3, i2)];
        const REAL lambdaU2_i1m2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 - 2, i2)];
        const REAL lambdaU2_i1m1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 - 1, i2)];
        const REAL lambdaU2_i0m3 = in_gfs[IDX4(LAMBDAU2GF, i0 - 3, i1, i2)];
        const REAL lambdaU2_i0m2 = in_gfs[IDX4(LAMBDAU2GF, i0 - 2, i1, i2)];
        const REAL lambdaU2_i0m1 = in_gfs[IDX4(LAMBDAU2GF, i0 - 1, i1, i2)];
        const REAL lambdaU2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2)];
        const REAL lambdaU2_i0p1 = in_gfs[IDX4(LAMBDAU2GF, i0 + 1, i1, i2)];
        const REAL lambdaU2_i0p2 = in_gfs[IDX4(LAMBDAU2GF, i0 + 2, i1, i2)];
        const REAL lambdaU2_i0p3 = in_gfs[IDX4(LAMBDAU2GF, i0 + 3, i1, i2)];
        const REAL lambdaU2_i1p1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 + 1, i2)];
        const REAL lambdaU2_i1p2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 + 2, i2)];
        const REAL lambdaU2_i1p3 = in_gfs[IDX4(LAMBDAU2GF, i0, i1 + 3, i2)];
        const REAL lambdaU2_i2p1 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 + 1)];
        const REAL lambdaU2_i2p2 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 + 2)];
        const REAL lambdaU2_i2p3 = in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2 + 3)];
        const REAL trK_i2m3 = in_gfs[IDX4(TRKGF, i0, i1, i2 - 3)];
        const REAL trK_i2m2 = in_gfs[IDX4(TRKGF, i0, i1, i2 - 2)];
        const REAL trK_i2m1 = in_gfs[IDX4(TRKGF, i0, i1, i2 - 1)];
        const REAL trK_i1m3 = in_gfs[IDX4(TRKGF, i0, i1 - 3, i2)];
        const REAL trK_i1m2 = in_gfs[IDX4(TRKGF, i0, i1 - 2, i2)];
        const REAL trK_i1m1 = in_gfs[IDX4(TRKGF, i0, i1 - 1, i2)];
        const REAL trK_i0m3 = in_gfs[IDX4(TRKGF, i0 - 3, i1, i2)];
        const REAL trK_i0m2 = in_gfs[IDX4(TRKGF, i0 - 2, i1, i2)];
        const REAL trK_i0m1 = in_gfs[IDX4(TRKGF, i0 - 1, i1, i2)];
        const REAL trK = in_gfs[IDX4(TRKGF, i0, i1, i2)];
        const REAL trK_i0p1 = in_gfs[IDX4(TRKGF, i0 + 1, i1, i2)];
        const REAL trK_i0p2 = in_gfs[IDX4(TRKGF, i0 + 2, i1, i2)];
        const REAL trK_i0p3 = in_gfs[IDX4(TRKGF, i0 + 3, i1, i2)];
        const REAL trK_i1p1 = in_gfs[IDX4(TRKGF, i0, i1 + 1, i2)];
        const REAL trK_i1p2 = in_gfs[IDX4(TRKGF, i0, i1 + 2, i2)];
        const REAL trK_i1p3 = in_gfs[IDX4(TRKGF, i0, i1 + 3, i2)];
        const REAL trK_i2p1 = in_gfs[IDX4(TRKGF, i0, i1, i2 + 1)];
        const REAL trK_i2p2 = in_gfs[IDX4(TRKGF, i0, i1, i2 + 2)];
        const REAL trK_i2p3 = in_gfs[IDX4(TRKGF, i0, i1, i2 + 3)];
        const REAL vetU0_i2m3 = in_gfs[IDX4(VETU0GF, i0, i1, i2 - 3)];
        const REAL vetU0_i1m2_i2m2 = in_gfs[IDX4(VETU0GF, i0, i1 - 2, i2 - 2)];
        const REAL vetU0_i1m1_i2m2 = in_gfs[IDX4(VETU0GF, i0, i1 - 1, i2 - 2)];
        const REAL vetU0_i0m2_i2m2 = in_gfs[IDX4(VETU0GF, i0 - 2, i1, i2 - 2)];
        const REAL vetU0_i0m1_i2m2 = in_gfs[IDX4(VETU0GF, i0 - 1, i1, i2 - 2)];
        const REAL vetU0_i2m2 = in_gfs[IDX4(VETU0GF, i0, i1, i2 - 2)];
        const REAL vetU0_i0p1_i2m2 = in_gfs[IDX4(VETU0GF, i0 + 1, i1, i2 - 2)];
        const REAL vetU0_i0p2_i2m2 = in_gfs[IDX4(VETU0GF, i0 + 2, i1, i2 - 2)];
        const REAL vetU0_i1p1_i2m2 = in_gfs[IDX4(VETU0GF, i0, i1 + 1, i2 - 2)];
        const REAL vetU0_i1p2_i2m2 = in_gfs[IDX4(VETU0GF, i0, i1 + 2, i2 - 2)];
        const REAL vetU0_i1m2_i2m1 = in_gfs[IDX4(VETU0GF, i0, i1 - 2, i2 - 1)];
        const REAL vetU0_i1m1_i2m1 = in_gfs[IDX4(VETU0GF, i0, i1 - 1, i2 - 1)];
        const REAL vetU0_i0m2_i2m1 = in_gfs[IDX4(VETU0GF, i0 - 2, i1, i2 - 1)];
        const REAL vetU0_i0m1_i2m1 = in_gfs[IDX4(VETU0GF, i0 - 1, i1, i2 - 1)];
        const REAL vetU0_i2m1 = in_gfs[IDX4(VETU0GF, i0, i1, i2 - 1)];
        const REAL vetU0_i0p1_i2m1 = in_gfs[IDX4(VETU0GF, i0 + 1, i1, i2 - 1)];
        const REAL vetU0_i0p2_i2m1 = in_gfs[IDX4(VETU0GF, i0 + 2, i1, i2 - 1)];
        const REAL vetU0_i1p1_i2m1 = in_gfs[IDX4(VETU0GF, i0, i1 + 1, i2 - 1)];
        const REAL vetU0_i1p2_i2m1 = in_gfs[IDX4(VETU0GF, i0, i1 + 2, i2 - 1)];
        const REAL vetU0_i1m3 = in_gfs[IDX4(VETU0GF, i0, i1 - 3, i2)];
        const REAL vetU0_i0m2_i1m2 = in_gfs[IDX4(VETU0GF, i0 - 2, i1 - 2, i2)];
        const REAL vetU0_i0m1_i1m2 = in_gfs[IDX4(VETU0GF, i0 - 1, i1 - 2, i2)];
        const REAL vetU0_i1m2 = in_gfs[IDX4(VETU0GF, i0, i1 - 2, i2)];
        const REAL vetU0_i0p1_i1m2 = in_gfs[IDX4(VETU0GF, i0 + 1, i1 - 2, i2)];
        const REAL vetU0_i0p2_i1m2 = in_gfs[IDX4(VETU0GF, i0 + 2, i1 - 2, i2)];
        const REAL vetU0_i0m2_i1m1 = in_gfs[IDX4(VETU0GF, i0 - 2, i1 - 1, i2)];
        const REAL vetU0_i0m1_i1m1 = in_gfs[IDX4(VETU0GF, i0 - 1, i1 - 1, i2)];
        const REAL vetU0_i1m1 = in_gfs[IDX4(VETU0GF, i0, i1 - 1, i2)];
        const REAL vetU0_i0p1_i1m1 = in_gfs[IDX4(VETU0GF, i0 + 1, i1 - 1, i2)];
        const REAL vetU0_i0p2_i1m1 = in_gfs[IDX4(VETU0GF, i0 + 2, i1 - 1, i2)];
        const REAL vetU0_i0m3 = in_gfs[IDX4(VETU0GF, i0 - 3, i1, i2)];
        const REAL vetU0_i0m2 = in_gfs[IDX4(VETU0GF, i0 - 2, i1, i2)];
        const REAL vetU0_i0m1 = in_gfs[IDX4(VETU0GF, i0 - 1, i1, i2)];
        const REAL vetU0 = in_gfs[IDX4(VETU0GF, i0, i1, i2)];
        const REAL vetU0_i0p1 = in_gfs[IDX4(VETU0GF, i0 + 1, i1, i2)];
        const REAL vetU0_i0p2 = in_gfs[IDX4(VETU0GF, i0 + 2, i1, i2)];
        const REAL vetU0_i0p3 = in_gfs[IDX4(VETU0GF, i0 + 3, i1, i2)];
        const REAL vetU0_i0m2_i1p1 = in_gfs[IDX4(VETU0GF, i0 - 2, i1 + 1, i2)];
        const REAL vetU0_i0m1_i1p1 = in_gfs[IDX4(VETU0GF, i0 - 1, i1 + 1, i2)];
        const REAL vetU0_i1p1 = in_gfs[IDX4(VETU0GF, i0, i1 + 1, i2)];
        const REAL vetU0_i0p1_i1p1 = in_gfs[IDX4(VETU0GF, i0 + 1, i1 + 1, i2)];
        const REAL vetU0_i0p2_i1p1 = in_gfs[IDX4(VETU0GF, i0 + 2, i1 + 1, i2)];
        const REAL vetU0_i0m2_i1p2 = in_gfs[IDX4(VETU0GF, i0 - 2, i1 + 2, i2)];
        const REAL vetU0_i0m1_i1p2 = in_gfs[IDX4(VETU0GF, i0 - 1, i1 + 2, i2)];
        const REAL vetU0_i1p2 = in_gfs[IDX4(VETU0GF, i0, i1 + 2, i2)];
        const REAL vetU0_i0p1_i1p2 = in_gfs[IDX4(VETU0GF, i0 + 1, i1 + 2, i2)];
        const REAL vetU0_i0p2_i1p2 = in_gfs[IDX4(VETU0GF, i0 + 2, i1 + 2, i2)];
        const REAL vetU0_i1p3 = in_gfs[IDX4(VETU0GF, i0, i1 + 3, i2)];
        const REAL vetU0_i1m2_i2p1 = in_gfs[IDX4(VETU0GF, i0, i1 - 2, i2 + 1)];
        const REAL vetU0_i1m1_i2p1 = in_gfs[IDX4(VETU0GF, i0, i1 - 1, i2 + 1)];
        const REAL vetU0_i0m2_i2p1 = in_gfs[IDX4(VETU0GF, i0 - 2, i1, i2 + 1)];
        const REAL vetU0_i0m1_i2p1 = in_gfs[IDX4(VETU0GF, i0 - 1, i1, i2 + 1)];
        const REAL vetU0_i2p1 = in_gfs[IDX4(VETU0GF, i0, i1, i2 + 1)];
        const REAL vetU0_i0p1_i2p1 = in_gfs[IDX4(VETU0GF, i0 + 1, i1, i2 + 1)];
        const REAL vetU0_i0p2_i2p1 = in_gfs[IDX4(VETU0GF, i0 + 2, i1, i2 + 1)];
        const REAL vetU0_i1p1_i2p1 = in_gfs[IDX4(VETU0GF, i0, i1 + 1, i2 + 1)];
        const REAL vetU0_i1p2_i2p1 = in_gfs[IDX4(VETU0GF, i0, i1 + 2, i2 + 1)];
        const REAL vetU0_i1m2_i2p2 = in_gfs[IDX4(VETU0GF, i0, i1 - 2, i2 + 2)];
        const REAL vetU0_i1m1_i2p2 = in_gfs[IDX4(VETU0GF, i0, i1 - 1, i2 + 2)];
        const REAL vetU0_i0m2_i2p2 = in_gfs[IDX4(VETU0GF, i0 - 2, i1, i2 + 2)];
        const REAL vetU0_i0m1_i2p2 = in_gfs[IDX4(VETU0GF, i0 - 1, i1, i2 + 2)];
        const REAL vetU0_i2p2 = in_gfs[IDX4(VETU0GF, i0, i1, i2 + 2)];
        const REAL vetU0_i0p1_i2p2 = in_gfs[IDX4(VETU0GF, i0 + 1, i1, i2 + 2)];
        const REAL vetU0_i0p2_i2p2 = in_gfs[IDX4(VETU0GF, i0 + 2, i1, i2 + 2)];
        const REAL vetU0_i1p1_i2p2 = in_gfs[IDX4(VETU0GF, i0, i1 + 1, i2 + 2)];
        const REAL vetU0_i1p2_i2p2 = in_gfs[IDX4(VETU0GF, i0, i1 + 2, i2 + 2)];
        const REAL vetU0_i2p3 = in_gfs[IDX4(VETU0GF, i0, i1, i2 + 3)];
        const REAL vetU1_i2m3 = in_gfs[IDX4(VETU1GF, i0, i1, i2 - 3)];
        const REAL vetU1_i1m2_i2m2 = in_gfs[IDX4(VETU1GF, i0, i1 - 2, i2 - 2)];
        const REAL vetU1_i1m1_i2m2 = in_gfs[IDX4(VETU1GF, i0, i1 - 1, i2 - 2)];
        const REAL vetU1_i0m2_i2m2 = in_gfs[IDX4(VETU1GF, i0 - 2, i1, i2 - 2)];
        const REAL vetU1_i0m1_i2m2 = in_gfs[IDX4(VETU1GF, i0 - 1, i1, i2 - 2)];
        const REAL vetU1_i2m2 = in_gfs[IDX4(VETU1GF, i0, i1, i2 - 2)];
        const REAL vetU1_i0p1_i2m2 = in_gfs[IDX4(VETU1GF, i0 + 1, i1, i2 - 2)];
        const REAL vetU1_i0p2_i2m2 = in_gfs[IDX4(VETU1GF, i0 + 2, i1, i2 - 2)];
        const REAL vetU1_i1p1_i2m2 = in_gfs[IDX4(VETU1GF, i0, i1 + 1, i2 - 2)];
        const REAL vetU1_i1p2_i2m2 = in_gfs[IDX4(VETU1GF, i0, i1 + 2, i2 - 2)];
        const REAL vetU1_i1m2_i2m1 = in_gfs[IDX4(VETU1GF, i0, i1 - 2, i2 - 1)];
        const REAL vetU1_i1m1_i2m1 = in_gfs[IDX4(VETU1GF, i0, i1 - 1, i2 - 1)];
        const REAL vetU1_i0m2_i2m1 = in_gfs[IDX4(VETU1GF, i0 - 2, i1, i2 - 1)];
        const REAL vetU1_i0m1_i2m1 = in_gfs[IDX4(VETU1GF, i0 - 1, i1, i2 - 1)];
        const REAL vetU1_i2m1 = in_gfs[IDX4(VETU1GF, i0, i1, i2 - 1)];
        const REAL vetU1_i0p1_i2m1 = in_gfs[IDX4(VETU1GF, i0 + 1, i1, i2 - 1)];
        const REAL vetU1_i0p2_i2m1 = in_gfs[IDX4(VETU1GF, i0 + 2, i1, i2 - 1)];
        const REAL vetU1_i1p1_i2m1 = in_gfs[IDX4(VETU1GF, i0, i1 + 1, i2 - 1)];
        const REAL vetU1_i1p2_i2m1 = in_gfs[IDX4(VETU1GF, i0, i1 + 2, i2 - 1)];
        const REAL vetU1_i1m3 = in_gfs[IDX4(VETU1GF, i0, i1 - 3, i2)];
        const REAL vetU1_i0m2_i1m2 = in_gfs[IDX4(VETU1GF, i0 - 2, i1 - 2, i2)];
        const REAL vetU1_i0m1_i1m2 = in_gfs[IDX4(VETU1GF, i0 - 1, i1 - 2, i2)];
        const REAL vetU1_i1m2 = in_gfs[IDX4(VETU1GF, i0, i1 - 2, i2)];
        const REAL vetU1_i0p1_i1m2 = in_gfs[IDX4(VETU1GF, i0 + 1, i1 - 2, i2)];
        const REAL vetU1_i0p2_i1m2 = in_gfs[IDX4(VETU1GF, i0 + 2, i1 - 2, i2)];
        const REAL vetU1_i0m2_i1m1 = in_gfs[IDX4(VETU1GF, i0 - 2, i1 - 1, i2)];
        const REAL vetU1_i0m1_i1m1 = in_gfs[IDX4(VETU1GF, i0 - 1, i1 - 1, i2)];
        const REAL vetU1_i1m1 = in_gfs[IDX4(VETU1GF, i0, i1 - 1, i2)];
        const REAL vetU1_i0p1_i1m1 = in_gfs[IDX4(VETU1GF, i0 + 1, i1 - 1, i2)];
        const REAL vetU1_i0p2_i1m1 = in_gfs[IDX4(VETU1GF, i0 + 2, i1 - 1, i2)];
        const REAL vetU1_i0m3 = in_gfs[IDX4(VETU1GF, i0 - 3, i1, i2)];
        const REAL vetU1_i0m2 = in_gfs[IDX4(VETU1GF, i0 - 2, i1, i2)];
        const REAL vetU1_i0m1 = in_gfs[IDX4(VETU1GF, i0 - 1, i1, i2)];
        const REAL vetU1 = in_gfs[IDX4(VETU1GF, i0, i1, i2)];
        const REAL vetU1_i0p1 = in_gfs[IDX4(VETU1GF, i0 + 1, i1, i2)];
        const REAL vetU1_i0p2 = in_gfs[IDX4(VETU1GF, i0 + 2, i1, i2)];
        const REAL vetU1_i0p3 = in_gfs[IDX4(VETU1GF, i0 + 3, i1, i2)];
        const REAL vetU1_i0m2_i1p1 = in_gfs[IDX4(VETU1GF, i0 - 2, i1 + 1, i2)];
        const REAL vetU1_i0m1_i1p1 = in_gfs[IDX4(VETU1GF, i0 - 1, i1 + 1, i2)];
        const REAL vetU1_i1p1 = in_gfs[IDX4(VETU1GF, i0, i1 + 1, i2)];
        const REAL vetU1_i0p1_i1p1 = in_gfs[IDX4(VETU1GF, i0 + 1, i1 + 1, i2)];
        const REAL vetU1_i0p2_i1p1 = in_gfs[IDX4(VETU1GF, i0 + 2, i1 + 1, i2)];
        const REAL vetU1_i0m2_i1p2 = in_gfs[IDX4(VETU1GF, i0 - 2, i1 + 2, i2)];
        const REAL vetU1_i0m1_i1p2 = in_gfs[IDX4(VETU1GF, i0 - 1, i1 + 2, i2)];
        const REAL vetU1_i1p2 = in_gfs[IDX4(VETU1GF, i0, i1 + 2, i2)];
        const REAL vetU1_i0p1_i1p2 = in_gfs[IDX4(VETU1GF, i0 + 1, i1 + 2, i2)];
        const REAL vetU1_i0p2_i1p2 = in_gfs[IDX4(VETU1GF, i0 + 2, i1 + 2, i2)];
        const REAL vetU1_i1p3 = in_gfs[IDX4(VETU1GF, i0, i1 + 3, i2)];
        const REAL vetU1_i1m2_i2p1 = in_gfs[IDX4(VETU1GF, i0, i1 - 2, i2 + 1)];
        const REAL vetU1_i1m1_i2p1 = in_gfs[IDX4(VETU1GF, i0, i1 - 1, i2 + 1)];
        const REAL vetU1_i0m2_i2p1 = in_gfs[IDX4(VETU1GF, i0 - 2, i1, i2 + 1)];
        const REAL vetU1_i0m1_i2p1 = in_gfs[IDX4(VETU1GF, i0 - 1, i1, i2 + 1)];
        const REAL vetU1_i2p1 = in_gfs[IDX4(VETU1GF, i0, i1, i2 + 1)];
        const REAL vetU1_i0p1_i2p1 = in_gfs[IDX4(VETU1GF, i0 + 1, i1, i2 + 1)];
        const REAL vetU1_i0p2_i2p1 = in_gfs[IDX4(VETU1GF, i0 + 2, i1, i2 + 1)];
        const REAL vetU1_i1p1_i2p1 = in_gfs[IDX4(VETU1GF, i0, i1 + 1, i2 + 1)];
        const REAL vetU1_i1p2_i2p1 = in_gfs[IDX4(VETU1GF, i0, i1 + 2, i2 + 1)];
        const REAL vetU1_i1m2_i2p2 = in_gfs[IDX4(VETU1GF, i0, i1 - 2, i2 + 2)];
        const REAL vetU1_i1m1_i2p2 = in_gfs[IDX4(VETU1GF, i0, i1 - 1, i2 + 2)];
        const REAL vetU1_i0m2_i2p2 = in_gfs[IDX4(VETU1GF, i0 - 2, i1, i2 + 2)];
        const REAL vetU1_i0m1_i2p2 = in_gfs[IDX4(VETU1GF, i0 - 1, i1, i2 + 2)];
        const REAL vetU1_i2p2 = in_gfs[IDX4(VETU1GF, i0, i1, i2 + 2)];
        const REAL vetU1_i0p1_i2p2 = in_gfs[IDX4(VETU1GF, i0 + 1, i1, i2 + 2)];
        const REAL vetU1_i0p2_i2p2 = in_gfs[IDX4(VETU1GF, i0 + 2, i1, i2 + 2)];
        const REAL vetU1_i1p1_i2p2 = in_gfs[IDX4(VETU1GF, i0, i1 + 1, i2 + 2)];
        const REAL vetU1_i1p2_i2p2 = in_gfs[IDX4(VETU1GF, i0, i1 + 2, i2 + 2)];
        const REAL vetU1_i2p3 = in_gfs[IDX4(VETU1GF, i0, i1, i2 + 3)];
        const REAL vetU2_i2m3 = in_gfs[IDX4(VETU2GF, i0, i1, i2 - 3)];
        const REAL vetU2_i1m2_i2m2 = in_gfs[IDX4(VETU2GF, i0, i1 - 2, i2 - 2)];
        const REAL vetU2_i1m1_i2m2 = in_gfs[IDX4(VETU2GF, i0, i1 - 1, i2 - 2)];
        const REAL vetU2_i0m2_i2m2 = in_gfs[IDX4(VETU2GF, i0 - 2, i1, i2 - 2)];
        const REAL vetU2_i0m1_i2m2 = in_gfs[IDX4(VETU2GF, i0 - 1, i1, i2 - 2)];
        const REAL vetU2_i2m2 = in_gfs[IDX4(VETU2GF, i0, i1, i2 - 2)];
        const REAL vetU2_i0p1_i2m2 = in_gfs[IDX4(VETU2GF, i0 + 1, i1, i2 - 2)];
        const REAL vetU2_i0p2_i2m2 = in_gfs[IDX4(VETU2GF, i0 + 2, i1, i2 - 2)];
        const REAL vetU2_i1p1_i2m2 = in_gfs[IDX4(VETU2GF, i0, i1 + 1, i2 - 2)];
        const REAL vetU2_i1p2_i2m2 = in_gfs[IDX4(VETU2GF, i0, i1 + 2, i2 - 2)];
        const REAL vetU2_i1m2_i2m1 = in_gfs[IDX4(VETU2GF, i0, i1 - 2, i2 - 1)];
        const REAL vetU2_i1m1_i2m1 = in_gfs[IDX4(VETU2GF, i0, i1 - 1, i2 - 1)];
        const REAL vetU2_i0m2_i2m1 = in_gfs[IDX4(VETU2GF, i0 - 2, i1, i2 - 1)];
        const REAL vetU2_i0m1_i2m1 = in_gfs[IDX4(VETU2GF, i0 - 1, i1, i2 - 1)];
        const REAL vetU2_i2m1 = in_gfs[IDX4(VETU2GF, i0, i1, i2 - 1)];
        const REAL vetU2_i0p1_i2m1 = in_gfs[IDX4(VETU2GF, i0 + 1, i1, i2 - 1)];
        const REAL vetU2_i0p2_i2m1 = in_gfs[IDX4(VETU2GF, i0 + 2, i1, i2 - 1)];
        const REAL vetU2_i1p1_i2m1 = in_gfs[IDX4(VETU2GF, i0, i1 + 1, i2 - 1)];
        const REAL vetU2_i1p2_i2m1 = in_gfs[IDX4(VETU2GF, i0, i1 + 2, i2 - 1)];
        const REAL vetU2_i1m3 = in_gfs[IDX4(VETU2GF, i0, i1 - 3, i2)];
        const REAL vetU2_i0m2_i1m2 = in_gfs[IDX4(VETU2GF, i0 - 2, i1 - 2, i2)];
        const REAL vetU2_i0m1_i1m2 = in_gfs[IDX4(VETU2GF, i0 - 1, i1 - 2, i2)];
        const REAL vetU2_i1m2 = in_gfs[IDX4(VETU2GF, i0, i1 - 2, i2)];
        const REAL vetU2_i0p1_i1m2 = in_gfs[IDX4(VETU2GF, i0 + 1, i1 - 2, i2)];
        const REAL vetU2_i0p2_i1m2 = in_gfs[IDX4(VETU2GF, i0 + 2, i1 - 2, i2)];
        const REAL vetU2_i0m2_i1m1 = in_gfs[IDX4(VETU2GF, i0 - 2, i1 - 1, i2)];
        const REAL vetU2_i0m1_i1m1 = in_gfs[IDX4(VETU2GF, i0 - 1, i1 - 1, i2)];
        const REAL vetU2_i1m1 = in_gfs[IDX4(VETU2GF, i0, i1 - 1, i2)];
        const REAL vetU2_i0p1_i1m1 = in_gfs[IDX4(VETU2GF, i0 + 1, i1 - 1, i2)];
        const REAL vetU2_i0p2_i1m1 = in_gfs[IDX4(VETU2GF, i0 + 2, i1 - 1, i2)];
        const REAL vetU2_i0m3 = in_gfs[IDX4(VETU2GF, i0 - 3, i1, i2)];
        const REAL vetU2_i0m2 = in_gfs[IDX4(VETU2GF, i0 - 2, i1, i2)];
        const REAL vetU2_i0m1 = in_gfs[IDX4(VETU2GF, i0 - 1, i1, i2)];
        const REAL vetU2 = in_gfs[IDX4(VETU2GF, i0, i1, i2)];
        const REAL vetU2_i0p1 = in_gfs[IDX4(VETU2GF, i0 + 1, i1, i2)];
        const REAL vetU2_i0p2 = in_gfs[IDX4(VETU2GF, i0 + 2, i1, i2)];
        const REAL vetU2_i0p3 = in_gfs[IDX4(VETU2GF, i0 + 3, i1, i2)];
        const REAL vetU2_i0m2_i1p1 = in_gfs[IDX4(VETU2GF, i0 - 2, i1 + 1, i2)];
        const REAL vetU2_i0m1_i1p1 = in_gfs[IDX4(VETU2GF, i0 - 1, i1 + 1, i2)];
        const REAL vetU2_i1p1 = in_gfs[IDX4(VETU2GF, i0, i1 + 1, i2)];
        const REAL vetU2_i0p1_i1p1 = in_gfs[IDX4(VETU2GF, i0 + 1, i1 + 1, i2)];
        const REAL vetU2_i0p2_i1p1 = in_gfs[IDX4(VETU2GF, i0 + 2, i1 + 1, i2)];
        const REAL vetU2_i0m2_i1p2 = in_gfs[IDX4(VETU2GF, i0 - 2, i1 + 2, i2)];
        const REAL vetU2_i0m1_i1p2 = in_gfs[IDX4(VETU2GF, i0 - 1, i1 + 2, i2)];
        const REAL vetU2_i1p2 = in_gfs[IDX4(VETU2GF, i0, i1 + 2, i2)];
        const REAL vetU2_i0p1_i1p2 = in_gfs[IDX4(VETU2GF, i0 + 1, i1 + 2, i2)];
        const REAL vetU2_i0p2_i1p2 = in_gfs[IDX4(VETU2GF, i0 + 2, i1 + 2, i2)];
        const REAL vetU2_i1p3 = in_gfs[IDX4(VETU2GF, i0, i1 + 3, i2)];
        const REAL vetU2_i1m2_i2p1 = in_gfs[IDX4(VETU2GF, i0, i1 - 2, i2 + 1)];
        const REAL vetU2_i1m1_i2p1 = in_gfs[IDX4(VETU2GF, i0, i1 - 1, i2 + 1)];
        const REAL vetU2_i0m2_i2p1 = in_gfs[IDX4(VETU2GF, i0 - 2, i1, i2 + 1)];
        const REAL vetU2_i0m1_i2p1 = in_gfs[IDX4(VETU2GF, i0 - 1, i1, i2 + 1)];
        const REAL vetU2_i2p1 = in_gfs[IDX4(VETU2GF, i0, i1, i2 + 1)];
        const REAL vetU2_i0p1_i2p1 = in_gfs[IDX4(VETU2GF, i0 + 1, i1, i2 + 1)];
        const REAL vetU2_i0p2_i2p1 = in_gfs[IDX4(VETU2GF, i0 + 2, i1, i2 + 1)];
        const REAL vetU2_i1p1_i2p1 = in_gfs[IDX4(VETU2GF, i0, i1 + 1, i2 + 1)];
        const REAL vetU2_i1p2_i2p1 = in_gfs[IDX4(VETU2GF, i0, i1 + 2, i2 + 1)];
        const REAL vetU2_i1m2_i2p2 = in_gfs[IDX4(VETU2GF, i0, i1 - 2, i2 + 2)];
        const REAL vetU2_i1m1_i2p2 = in_gfs[IDX4(VETU2GF, i0, i1 - 1, i2 + 2)];
        const REAL vetU2_i0m2_i2p2 = in_gfs[IDX4(VETU2GF, i0 - 2, i1, i2 + 2)];
        const REAL vetU2_i0m1_i2p2 = in_gfs[IDX4(VETU2GF, i0 - 1, i1, i2 + 2)];
        const REAL vetU2_i2p2 = in_gfs[IDX4(VETU2GF, i0, i1, i2 + 2)];
        const REAL vetU2_i0p1_i2p2 = in_gfs[IDX4(VETU2GF, i0 + 1, i1, i2 + 2)];
        const REAL vetU2_i0p2_i2p2 = in_gfs[IDX4(VETU2GF, i0 + 2, i1, i2 + 2)];
        const REAL vetU2_i1p1_i2p2 = in_gfs[IDX4(VETU2GF, i0, i1 + 1, i2 + 2)];
        const REAL vetU2_i1p2_i2p2 = in_gfs[IDX4(VETU2GF, i0, i1 + 2, i2 + 2)];
        const REAL vetU2_i2p3 = in_gfs[IDX4(VETU2GF, i0, i1, i2 + 3)];

        const REAL UpwindAlgInputaDD_ddnD000 = fd_function_ddnD0_fdorder4(aDD00, aDD00_i0m1, aDD00_i0m2, aDD00_i0m3, aDD00_i0p1, invdxx0);
        const REAL UpwindAlgInputaDD_ddnD010 = fd_function_ddnD0_fdorder4(aDD01, aDD01_i0m1, aDD01_i0m2, aDD01_i0m3, aDD01_i0p1, invdxx0);
        const REAL UpwindAlgInputaDD_ddnD020 = fd_function_ddnD0_fdorder4(aDD02, aDD02_i0m1, aDD02_i0m2, aDD02_i0m3, aDD02_i0p1, invdxx0);
        const REAL UpwindAlgInputaDD_ddnD110 = fd_function_ddnD0_fdorder4(aDD11, aDD11_i0m1, aDD11_i0m2, aDD11_i0m3, aDD11_i0p1, invdxx0);
        const REAL UpwindAlgInputaDD_ddnD120 = fd_function_ddnD0_fdorder4(aDD12, aDD12_i0m1, aDD12_i0m2, aDD12_i0m3, aDD12_i0p1, invdxx0);
        const REAL UpwindAlgInputaDD_ddnD220 = fd_function_ddnD0_fdorder4(aDD22, aDD22_i0m1, aDD22_i0m2, aDD22_i0m3, aDD22_i0p1, invdxx0);
        const REAL UpwindAlgInputalpha_ddnD0 = fd_function_ddnD0_fdorder4(alpha, alpha_i0m1, alpha_i0m2, alpha_i0m3, alpha_i0p1, invdxx0);
        const REAL UpwindAlgInputbetU_ddnD00 = fd_function_ddnD0_fdorder4(betU0, betU0_i0m1, betU0_i0m2, betU0_i0m3, betU0_i0p1, invdxx0);
        const REAL UpwindAlgInputbetU_ddnD10 = fd_function_ddnD0_fdorder4(betU1, betU1_i0m1, betU1_i0m2, betU1_i0m3, betU1_i0p1, invdxx0);
        const REAL UpwindAlgInputbetU_ddnD20 = fd_function_ddnD0_fdorder4(betU2, betU2_i0m1, betU2_i0m2, betU2_i0m3, betU2_i0p1, invdxx0);
        const REAL UpwindAlgInputcf_ddnD0 = fd_function_ddnD0_fdorder4(cf, cf_i0m1, cf_i0m2, cf_i0m3, cf_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD000 = fd_function_ddnD0_fdorder4(hDD00, hDD00_i0m1, hDD00_i0m2, hDD00_i0m3, hDD00_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD010 = fd_function_ddnD0_fdorder4(hDD01, hDD01_i0m1, hDD01_i0m2, hDD01_i0m3, hDD01_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD020 = fd_function_ddnD0_fdorder4(hDD02, hDD02_i0m1, hDD02_i0m2, hDD02_i0m3, hDD02_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD110 = fd_function_ddnD0_fdorder4(hDD11, hDD11_i0m1, hDD11_i0m2, hDD11_i0m3, hDD11_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD120 = fd_function_ddnD0_fdorder4(hDD12, hDD12_i0m1, hDD12_i0m2, hDD12_i0m3, hDD12_i0p1, invdxx0);
        const REAL UpwindAlgInputhDD_ddnD220 = fd_function_ddnD0_fdorder4(hDD22, hDD22_i0m1, hDD22_i0m2, hDD22_i0m3, hDD22_i0p1, invdxx0);
        const REAL UpwindAlgInputlambdaU_ddnD00 =
            fd_function_ddnD0_fdorder4(lambdaU0, lambdaU0_i0m1, lambdaU0_i0m2, lambdaU0_i0m3, lambdaU0_i0p1, invdxx0);
        const REAL UpwindAlgInputlambdaU_ddnD10 =
            fd_function_ddnD0_fdorder4(lambdaU1, lambdaU1_i0m1, lambdaU1_i0m2, lambdaU1_i0m3, lambdaU1_i0p1, invdxx0);
        const REAL UpwindAlgInputlambdaU_ddnD20 =
            fd_function_ddnD0_fdorder4(lambdaU2, lambdaU2_i0m1, lambdaU2_i0m2, lambdaU2_i0m3, lambdaU2_i0p1, invdxx0);
        const REAL UpwindAlgInputvetU_ddnD00 = fd_function_ddnD0_fdorder4(vetU0, vetU0_i0m1, vetU0_i0m2, vetU0_i0m3, vetU0_i0p1, invdxx0);
        const REAL UpwindAlgInputtrK_ddnD0 = fd_function_ddnD0_fdorder4(trK, trK_i0m1, trK_i0m2, trK_i0m3, trK_i0p1, invdxx0);
        const REAL UpwindAlgInputvetU_ddnD10 = fd_function_ddnD0_fdorder4(vetU1, vetU1_i0m1, vetU1_i0m2, vetU1_i0m3, vetU1_i0p1, invdxx0);
        const REAL UpwindAlgInputvetU_ddnD20 = fd_function_ddnD0_fdorder4(vetU2, vetU2_i0m1, vetU2_i0m2, vetU2_i0m3, vetU2_i0p1, invdxx0);
        
        const REAL UpwindAlgInputaDD_ddnD001 = fd_function_ddnD1_fdorder4(aDD00, aDD00_i1m1, aDD00_i1m2, aDD00_i1m3, aDD00_i1p1, invdxx1);
        const REAL UpwindAlgInputaDD_ddnD011 = fd_function_ddnD1_fdorder4(aDD01, aDD01_i1m1, aDD01_i1m2, aDD01_i1m3, aDD01_i1p1, invdxx1);
        const REAL UpwindAlgInputaDD_ddnD021 = fd_function_ddnD1_fdorder4(aDD02, aDD02_i1m1, aDD02_i1m2, aDD02_i1m3, aDD02_i1p1, invdxx1);
        const REAL UpwindAlgInputaDD_ddnD111 = fd_function_ddnD1_fdorder4(aDD11, aDD11_i1m1, aDD11_i1m2, aDD11_i1m3, aDD11_i1p1, invdxx1);
        const REAL UpwindAlgInputaDD_ddnD121 = fd_function_ddnD1_fdorder4(aDD12, aDD12_i1m1, aDD12_i1m2, aDD12_i1m3, aDD12_i1p1, invdxx1);
        const REAL UpwindAlgInputaDD_ddnD221 = fd_function_ddnD1_fdorder4(aDD22, aDD22_i1m1, aDD22_i1m2, aDD22_i1m3, aDD22_i1p1, invdxx1);
        const REAL UpwindAlgInputalpha_ddnD1 = fd_function_ddnD1_fdorder4(alpha, alpha_i1m1, alpha_i1m2, alpha_i1m3, alpha_i1p1, invdxx1);
        const REAL UpwindAlgInputbetU_ddnD01 = fd_function_ddnD1_fdorder4(betU0, betU0_i1m1, betU0_i1m2, betU0_i1m3, betU0_i1p1, invdxx1);
        const REAL UpwindAlgInputbetU_ddnD11 = fd_function_ddnD1_fdorder4(betU1, betU1_i1m1, betU1_i1m2, betU1_i1m3, betU1_i1p1, invdxx1);
        const REAL UpwindAlgInputbetU_ddnD21 = fd_function_ddnD1_fdorder4(betU2, betU2_i1m1, betU2_i1m2, betU2_i1m3, betU2_i1p1, invdxx1);
        const REAL UpwindAlgInputcf_ddnD1 = fd_function_ddnD1_fdorder4(cf, cf_i1m1, cf_i1m2, cf_i1m3, cf_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD001 = fd_function_ddnD1_fdorder4(hDD00, hDD00_i1m1, hDD00_i1m2, hDD00_i1m3, hDD00_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD011 = fd_function_ddnD1_fdorder4(hDD01, hDD01_i1m1, hDD01_i1m2, hDD01_i1m3, hDD01_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD021 = fd_function_ddnD1_fdorder4(hDD02, hDD02_i1m1, hDD02_i1m2, hDD02_i1m3, hDD02_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD111 = fd_function_ddnD1_fdorder4(hDD11, hDD11_i1m1, hDD11_i1m2, hDD11_i1m3, hDD11_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD121 = fd_function_ddnD1_fdorder4(hDD12, hDD12_i1m1, hDD12_i1m2, hDD12_i1m3, hDD12_i1p1, invdxx1);
        const REAL UpwindAlgInputhDD_ddnD221 = fd_function_ddnD1_fdorder4(hDD22, hDD22_i1m1, hDD22_i1m2, hDD22_i1m3, hDD22_i1p1, invdxx1);
        const REAL UpwindAlgInputlambdaU_ddnD01 =
            fd_function_ddnD1_fdorder4(lambdaU0, lambdaU0_i1m1, lambdaU0_i1m2, lambdaU0_i1m3, lambdaU0_i1p1, invdxx1);
        const REAL UpwindAlgInputlambdaU_ddnD11 =
            fd_function_ddnD1_fdorder4(lambdaU1, lambdaU1_i1m1, lambdaU1_i1m2, lambdaU1_i1m3, lambdaU1_i1p1, invdxx1);
        const REAL UpwindAlgInputlambdaU_ddnD21 =
            fd_function_ddnD1_fdorder4(lambdaU2, lambdaU2_i1m1, lambdaU2_i1m2, lambdaU2_i1m3, lambdaU2_i1p1, invdxx1);
        const REAL UpwindAlgInputtrK_ddnD1 = fd_function_ddnD1_fdorder4(trK, trK_i1m1, trK_i1m2, trK_i1m3, trK_i1p1, invdxx1);
        const REAL UpwindAlgInputvetU_ddnD01 = fd_function_ddnD1_fdorder4(vetU0, vetU0_i1m1, vetU0_i1m2, vetU0_i1m3, vetU0_i1p1, invdxx1);
        const REAL UpwindAlgInputvetU_ddnD11 = fd_function_ddnD1_fdorder4(vetU1, vetU1_i1m1, vetU1_i1m2, vetU1_i1m3, vetU1_i1p1, invdxx1);
        const REAL UpwindAlgInputvetU_ddnD21 = fd_function_ddnD1_fdorder4(vetU2, vetU2_i1m1, vetU2_i1m2, vetU2_i1m3, vetU2_i1p1, invdxx1);
        
        
        const REAL UpwindAlgInputaDD_ddnD002 = fd_function_ddnD2_fdorder4(aDD00, aDD00_i2m1, aDD00_i2m2, aDD00_i2m3, aDD00_i2p1, invdxx2);
        const REAL UpwindAlgInputaDD_ddnD012 = fd_function_ddnD2_fdorder4(aDD01, aDD01_i2m1, aDD01_i2m2, aDD01_i2m3, aDD01_i2p1, invdxx2);
        const REAL UpwindAlgInputaDD_ddnD022 = fd_function_ddnD2_fdorder4(aDD02, aDD02_i2m1, aDD02_i2m2, aDD02_i2m3, aDD02_i2p1, invdxx2);
        const REAL UpwindAlgInputaDD_ddnD112 = fd_function_ddnD2_fdorder4(aDD11, aDD11_i2m1, aDD11_i2m2, aDD11_i2m3, aDD11_i2p1, invdxx2);
        const REAL UpwindAlgInputaDD_ddnD122 = fd_function_ddnD2_fdorder4(aDD12, aDD12_i2m1, aDD12_i2m2, aDD12_i2m3, aDD12_i2p1, invdxx2);
        const REAL UpwindAlgInputaDD_ddnD222 = fd_function_ddnD2_fdorder4(aDD22, aDD22_i2m1, aDD22_i2m2, aDD22_i2m3, aDD22_i2p1, invdxx2);
        const REAL UpwindAlgInputalpha_ddnD2 = fd_function_ddnD2_fdorder4(alpha, alpha_i2m1, alpha_i2m2, alpha_i2m3, alpha_i2p1, invdxx2);
        const REAL UpwindAlgInputbetU_ddnD02 = fd_function_ddnD2_fdorder4(betU0, betU0_i2m1, betU0_i2m2, betU0_i2m3, betU0_i2p1, invdxx2);
        const REAL UpwindAlgInputbetU_ddnD12 = fd_function_ddnD2_fdorder4(betU1, betU1_i2m1, betU1_i2m2, betU1_i2m3, betU1_i2p1, invdxx2);
        const REAL UpwindAlgInputbetU_ddnD22 = fd_function_ddnD2_fdorder4(betU2, betU2_i2m1, betU2_i2m2, betU2_i2m3, betU2_i2p1, invdxx2);
        const REAL UpwindAlgInputcf_ddnD2 = fd_function_ddnD2_fdorder4(cf, cf_i2m1, cf_i2m2, cf_i2m3, cf_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD002 = fd_function_ddnD2_fdorder4(hDD00, hDD00_i2m1, hDD00_i2m2, hDD00_i2m3, hDD00_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD012 = fd_function_ddnD2_fdorder4(hDD01, hDD01_i2m1, hDD01_i2m2, hDD01_i2m3, hDD01_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD022 = fd_function_ddnD2_fdorder4(hDD02, hDD02_i2m1, hDD02_i2m2, hDD02_i2m3, hDD02_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD112 = fd_function_ddnD2_fdorder4(hDD11, hDD11_i2m1, hDD11_i2m2, hDD11_i2m3, hDD11_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD122 = fd_function_ddnD2_fdorder4(hDD12, hDD12_i2m1, hDD12_i2m2, hDD12_i2m3, hDD12_i2p1, invdxx2);
        const REAL UpwindAlgInputhDD_ddnD222 = fd_function_ddnD2_fdorder4(hDD22, hDD22_i2m1, hDD22_i2m2, hDD22_i2m3, hDD22_i2p1, invdxx2);
        const REAL UpwindAlgInputlambdaU_ddnD02 =
            fd_function_ddnD2_fdorder4(lambdaU0, lambdaU0_i2m1, lambdaU0_i2m2, lambdaU0_i2m3, lambdaU0_i2p1, invdxx2);

        const REAL UpwindAlgInputlambdaU_ddnD12 =
            fd_function_ddnD2_fdorder4(lambdaU1, lambdaU1_i2m1, lambdaU1_i2m2, lambdaU1_i2m3, lambdaU1_i2p1, invdxx2);

        const REAL UpwindAlgInputlambdaU_ddnD22 =
            fd_function_ddnD2_fdorder4(lambdaU2, lambdaU2_i2m1, lambdaU2_i2m2, lambdaU2_i2m3, lambdaU2_i2p1, invdxx2);
        const REAL UpwindAlgInputtrK_ddnD2 = fd_function_ddnD2_fdorder4(trK, trK_i2m1, trK_i2m2, trK_i2m3, trK_i2p1, invdxx2);
        const REAL UpwindAlgInputvetU_ddnD02 = fd_function_ddnD2_fdorder4(vetU0, vetU0_i2m1, vetU0_i2m2, vetU0_i2m3, vetU0_i2p1, invdxx2);
        const REAL UpwindAlgInputvetU_ddnD12 = fd_function_ddnD2_fdorder4(vetU1, vetU1_i2m1, vetU1_i2m2, vetU1_i2m3, vetU1_i2p1, invdxx2);
        const REAL UpwindAlgInputvetU_ddnD22 = fd_function_ddnD2_fdorder4(vetU2, vetU2_i2m1, vetU2_i2m2, vetU2_i2m3, vetU2_i2p1, invdxx2);

        const REAL UpwindAlgInputaDD_dupD000 = fd_function_dupD0_fdorder4(aDD00, aDD00_i0m1, aDD00_i0p1, aDD00_i0p2, aDD00_i0p3, invdxx0);
        const REAL UpwindAlgInputaDD_dupD010 = fd_function_dupD0_fdorder4(aDD01, aDD01_i0m1, aDD01_i0p1, aDD01_i0p2, aDD01_i0p3, invdxx0);
        const REAL UpwindAlgInputaDD_dupD020 = fd_function_dupD0_fdorder4(aDD02, aDD02_i0m1, aDD02_i0p1, aDD02_i0p2, aDD02_i0p3, invdxx0);
        const REAL UpwindAlgInputaDD_dupD110 = fd_function_dupD0_fdorder4(aDD11, aDD11_i0m1, aDD11_i0p1, aDD11_i0p2, aDD11_i0p3, invdxx0);
        const REAL UpwindAlgInputaDD_dupD120 = fd_function_dupD0_fdorder4(aDD12, aDD12_i0m1, aDD12_i0p1, aDD12_i0p2, aDD12_i0p3, invdxx0);
        const REAL UpwindAlgInputaDD_dupD220 = fd_function_dupD0_fdorder4(aDD22, aDD22_i0m1, aDD22_i0p1, aDD22_i0p2, aDD22_i0p3, invdxx0);
        const REAL UpwindAlgInputalpha_dupD0 = fd_function_dupD0_fdorder4(alpha, alpha_i0m1, alpha_i0p1, alpha_i0p2, alpha_i0p3, invdxx0);
        const REAL UpwindAlgInputbetU_dupD00 = fd_function_dupD0_fdorder4(betU0, betU0_i0m1, betU0_i0p1, betU0_i0p2, betU0_i0p3, invdxx0);
        const REAL UpwindAlgInputbetU_dupD10 = fd_function_dupD0_fdorder4(betU1, betU1_i0m1, betU1_i0p1, betU1_i0p2, betU1_i0p3, invdxx0);
        const REAL UpwindAlgInputbetU_dupD20 = fd_function_dupD0_fdorder4(betU2, betU2_i0m1, betU2_i0p1, betU2_i0p2, betU2_i0p3, invdxx0);
        const REAL UpwindAlgInputcf_dupD0 = fd_function_dupD0_fdorder4(cf, cf_i0m1, cf_i0p1, cf_i0p2, cf_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD000 = fd_function_dupD0_fdorder4(hDD00, hDD00_i0m1, hDD00_i0p1, hDD00_i0p2, hDD00_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD010 = fd_function_dupD0_fdorder4(hDD01, hDD01_i0m1, hDD01_i0p1, hDD01_i0p2, hDD01_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD020 = fd_function_dupD0_fdorder4(hDD02, hDD02_i0m1, hDD02_i0p1, hDD02_i0p2, hDD02_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD110 = fd_function_dupD0_fdorder4(hDD11, hDD11_i0m1, hDD11_i0p1, hDD11_i0p2, hDD11_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD120 = fd_function_dupD0_fdorder4(hDD12, hDD12_i0m1, hDD12_i0p1, hDD12_i0p2, hDD12_i0p3, invdxx0);
        const REAL UpwindAlgInputhDD_dupD220 = fd_function_dupD0_fdorder4(hDD22, hDD22_i0m1, hDD22_i0p1, hDD22_i0p2, hDD22_i0p3, invdxx0);
        const REAL UpwindAlgInputlambdaU_dupD00 =
            fd_function_dupD0_fdorder4(lambdaU0, lambdaU0_i0m1, lambdaU0_i0p1, lambdaU0_i0p2, lambdaU0_i0p3, invdxx0);
        const REAL UpwindAlgInputlambdaU_dupD10 =
            fd_function_dupD0_fdorder4(lambdaU1, lambdaU1_i0m1, lambdaU1_i0p1, lambdaU1_i0p2, lambdaU1_i0p3, invdxx0);
        const REAL UpwindAlgInputlambdaU_dupD20 =
            fd_function_dupD0_fdorder4(lambdaU2, lambdaU2_i0m1, lambdaU2_i0p1, lambdaU2_i0p2, lambdaU2_i0p3, invdxx0);
        const REAL UpwindAlgInputtrK_dupD0 = fd_function_dupD0_fdorder4(trK, trK_i0m1, trK_i0p1, trK_i0p2, trK_i0p3, invdxx0);
        const REAL UpwindAlgInputvetU_dupD00 = fd_function_dupD0_fdorder4(vetU0, vetU0_i0m1, vetU0_i0p1, vetU0_i0p2, vetU0_i0p3, invdxx0);
        const REAL UpwindAlgInputvetU_dupD10 = fd_function_dupD0_fdorder4(vetU1, vetU1_i0m1, vetU1_i0p1, vetU1_i0p2, vetU1_i0p3, invdxx0);
        const REAL UpwindAlgInputvetU_dupD20 = fd_function_dupD0_fdorder4(vetU2, vetU2_i0m1, vetU2_i0p1, vetU2_i0p2, vetU2_i0p3, invdxx0);

        const REAL UpwindAlgInputaDD_dupD001 = fd_function_dupD1_fdorder4(aDD00, aDD00_i1m1, aDD00_i1p1, aDD00_i1p2, aDD00_i1p3, invdxx1);
        const REAL UpwindAlgInputaDD_dupD011 = fd_function_dupD1_fdorder4(aDD01, aDD01_i1m1, aDD01_i1p1, aDD01_i1p2, aDD01_i1p3, invdxx1);
        const REAL UpwindAlgInputaDD_dupD021 = fd_function_dupD1_fdorder4(aDD02, aDD02_i1m1, aDD02_i1p1, aDD02_i1p2, aDD02_i1p3, invdxx1);
        const REAL UpwindAlgInputaDD_dupD111 = fd_function_dupD1_fdorder4(aDD11, aDD11_i1m1, aDD11_i1p1, aDD11_i1p2, aDD11_i1p3, invdxx1);
        const REAL UpwindAlgInputaDD_dupD121 = fd_function_dupD1_fdorder4(aDD12, aDD12_i1m1, aDD12_i1p1, aDD12_i1p2, aDD12_i1p3, invdxx1);
        const REAL UpwindAlgInputaDD_dupD221 = fd_function_dupD1_fdorder4(aDD22, aDD22_i1m1, aDD22_i1p1, aDD22_i1p2, aDD22_i1p3, invdxx1);
        const REAL UpwindAlgInputalpha_dupD1 = fd_function_dupD1_fdorder4(alpha, alpha_i1m1, alpha_i1p1, alpha_i1p2, alpha_i1p3, invdxx1);
        const REAL UpwindAlgInputbetU_dupD01 = fd_function_dupD1_fdorder4(betU0, betU0_i1m1, betU0_i1p1, betU0_i1p2, betU0_i1p3, invdxx1);
        const REAL UpwindAlgInputbetU_dupD11 = fd_function_dupD1_fdorder4(betU1, betU1_i1m1, betU1_i1p1, betU1_i1p2, betU1_i1p3, invdxx1);
        const REAL UpwindAlgInputbetU_dupD21 = fd_function_dupD1_fdorder4(betU2, betU2_i1m1, betU2_i1p1, betU2_i1p2, betU2_i1p3, invdxx1);
        const REAL UpwindAlgInputcf_dupD1 = fd_function_dupD1_fdorder4(cf, cf_i1m1, cf_i1p1, cf_i1p2, cf_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD001 = fd_function_dupD1_fdorder4(hDD00, hDD00_i1m1, hDD00_i1p1, hDD00_i1p2, hDD00_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD011 = fd_function_dupD1_fdorder4(hDD01, hDD01_i1m1, hDD01_i1p1, hDD01_i1p2, hDD01_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD021 = fd_function_dupD1_fdorder4(hDD02, hDD02_i1m1, hDD02_i1p1, hDD02_i1p2, hDD02_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD111 = fd_function_dupD1_fdorder4(hDD11, hDD11_i1m1, hDD11_i1p1, hDD11_i1p2, hDD11_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD121 = fd_function_dupD1_fdorder4(hDD12, hDD12_i1m1, hDD12_i1p1, hDD12_i1p2, hDD12_i1p3, invdxx1);
        const REAL UpwindAlgInputhDD_dupD221 = fd_function_dupD1_fdorder4(hDD22, hDD22_i1m1, hDD22_i1p1, hDD22_i1p2, hDD22_i1p3, invdxx1);
        const REAL UpwindAlgInputlambdaU_dupD01 =
            fd_function_dupD1_fdorder4(lambdaU0, lambdaU0_i1m1, lambdaU0_i1p1, lambdaU0_i1p2, lambdaU0_i1p3, invdxx1);
        const REAL UpwindAlgInputlambdaU_dupD11 =
            fd_function_dupD1_fdorder4(lambdaU1, lambdaU1_i1m1, lambdaU1_i1p1, lambdaU1_i1p2, lambdaU1_i1p3, invdxx1);
        const REAL UpwindAlgInputlambdaU_dupD21 =
            fd_function_dupD1_fdorder4(lambdaU2, lambdaU2_i1m1, lambdaU2_i1p1, lambdaU2_i1p2, lambdaU2_i1p3, invdxx1);
        const REAL UpwindAlgInputtrK_dupD1 = fd_function_dupD1_fdorder4(trK, trK_i1m1, trK_i1p1, trK_i1p2, trK_i1p3, invdxx1);
        const REAL UpwindAlgInputvetU_dupD01 = fd_function_dupD1_fdorder4(vetU0, vetU0_i1m1, vetU0_i1p1, vetU0_i1p2, vetU0_i1p3, invdxx1);
        const REAL UpwindAlgInputvetU_dupD11 = fd_function_dupD1_fdorder4(vetU1, vetU1_i1m1, vetU1_i1p1, vetU1_i1p2, vetU1_i1p3, invdxx1);
        const REAL UpwindAlgInputvetU_dupD21 = fd_function_dupD1_fdorder4(vetU2, vetU2_i1m1, vetU2_i1p1, vetU2_i1p2, vetU2_i1p3, invdxx1);


        const REAL UpwindAlgInputaDD_dupD002 = fd_function_dupD2_fdorder4(aDD00, aDD00_i2m1, aDD00_i2p1, aDD00_i2p2, aDD00_i2p3, invdxx2);
        const REAL UpwindAlgInputaDD_dupD012 = fd_function_dupD2_fdorder4(aDD01, aDD01_i2m1, aDD01_i2p1, aDD01_i2p2, aDD01_i2p3, invdxx2);
        const REAL UpwindAlgInputaDD_dupD022 = fd_function_dupD2_fdorder4(aDD02, aDD02_i2m1, aDD02_i2p1, aDD02_i2p2, aDD02_i2p3, invdxx2);
        const REAL UpwindAlgInputaDD_dupD112 = fd_function_dupD2_fdorder4(aDD11, aDD11_i2m1, aDD11_i2p1, aDD11_i2p2, aDD11_i2p3, invdxx2);
        const REAL UpwindAlgInputaDD_dupD122 = fd_function_dupD2_fdorder4(aDD12, aDD12_i2m1, aDD12_i2p1, aDD12_i2p2, aDD12_i2p3, invdxx2);
        const REAL UpwindAlgInputaDD_dupD222 = fd_function_dupD2_fdorder4(aDD22, aDD22_i2m1, aDD22_i2p1, aDD22_i2p2, aDD22_i2p3, invdxx2);
        const REAL UpwindAlgInputalpha_dupD2 = fd_function_dupD2_fdorder4(alpha, alpha_i2m1, alpha_i2p1, alpha_i2p2, alpha_i2p3, invdxx2);
        const REAL UpwindAlgInputbetU_dupD02 = fd_function_dupD2_fdorder4(betU0, betU0_i2m1, betU0_i2p1, betU0_i2p2, betU0_i2p3, invdxx2);
        const REAL UpwindAlgInputbetU_dupD12 = fd_function_dupD2_fdorder4(betU1, betU1_i2m1, betU1_i2p1, betU1_i2p2, betU1_i2p3, invdxx2);
        const REAL UpwindAlgInputbetU_dupD22 = fd_function_dupD2_fdorder4(betU2, betU2_i2m1, betU2_i2p1, betU2_i2p2, betU2_i2p3, invdxx2);
        const REAL UpwindAlgInputcf_dupD2 = fd_function_dupD2_fdorder4(cf, cf_i2m1, cf_i2p1, cf_i2p2, cf_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD002 = fd_function_dupD2_fdorder4(hDD00, hDD00_i2m1, hDD00_i2p1, hDD00_i2p2, hDD00_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD012 = fd_function_dupD2_fdorder4(hDD01, hDD01_i2m1, hDD01_i2p1, hDD01_i2p2, hDD01_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD022 = fd_function_dupD2_fdorder4(hDD02, hDD02_i2m1, hDD02_i2p1, hDD02_i2p2, hDD02_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD112 = fd_function_dupD2_fdorder4(hDD11, hDD11_i2m1, hDD11_i2p1, hDD11_i2p2, hDD11_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD122 = fd_function_dupD2_fdorder4(hDD12, hDD12_i2m1, hDD12_i2p1, hDD12_i2p2, hDD12_i2p3, invdxx2);
        const REAL UpwindAlgInputhDD_dupD222 = fd_function_dupD2_fdorder4(hDD22, hDD22_i2m1, hDD22_i2p1, hDD22_i2p2, hDD22_i2p3, invdxx2);
        const REAL UpwindAlgInputlambdaU_dupD02 =
            fd_function_dupD2_fdorder4(lambdaU0, lambdaU0_i2m1, lambdaU0_i2p1, lambdaU0_i2p2, lambdaU0_i2p3, invdxx2);
        const REAL UpwindAlgInputlambdaU_dupD12 =
            fd_function_dupD2_fdorder4(lambdaU1, lambdaU1_i2m1, lambdaU1_i2p1, lambdaU1_i2p2, lambdaU1_i2p3, invdxx2);
        const REAL UpwindAlgInputlambdaU_dupD22 =
            fd_function_dupD2_fdorder4(lambdaU2, lambdaU2_i2m1, lambdaU2_i2p1, lambdaU2_i2p2, lambdaU2_i2p3, invdxx2);
        const REAL UpwindAlgInputtrK_dupD2 = fd_function_dupD2_fdorder4(trK, trK_i2m1, trK_i2p1, trK_i2p2, trK_i2p3, invdxx2);
        const REAL UpwindAlgInputvetU_dupD02 = fd_function_dupD2_fdorder4(vetU0, vetU0_i2m1, vetU0_i2p1, vetU0_i2p2, vetU0_i2p3, invdxx2);
        const REAL UpwindAlgInputvetU_dupD12 = fd_function_dupD2_fdorder4(vetU1, vetU1_i2m1, vetU1_i2p1, vetU1_i2p2, vetU1_i2p3, invdxx2);
        const REAL UpwindAlgInputvetU_dupD22 = fd_function_dupD2_fdorder4(vetU2, vetU2_i2m1, vetU2_i2p1, vetU2_i2p2, vetU2_i2p3, invdxx2);
        
        
        const REAL alpha_dD0 = fd_function_dD0_fdorder4(alpha_i0m1, alpha_i0m2, alpha_i0p1, alpha_i0p2, invdxx0);
        const REAL cf_dD0 = fd_function_dD0_fdorder4(cf_i0m1, cf_i0m2, cf_i0p1, cf_i0p2, invdxx0);
        const REAL hDD_dD000 = fd_function_dD0_fdorder4(hDD00_i0m1, hDD00_i0m2, hDD00_i0p1, hDD00_i0p2, invdxx0);
        const REAL hDD_dD010 = fd_function_dD0_fdorder4(hDD01_i0m1, hDD01_i0m2, hDD01_i0p1, hDD01_i0p2, invdxx0);
        const REAL hDD_dD020 = fd_function_dD0_fdorder4(hDD02_i0m1, hDD02_i0m2, hDD02_i0p1, hDD02_i0p2, invdxx0);
        const REAL hDD_dD110 = fd_function_dD0_fdorder4(hDD11_i0m1, hDD11_i0m2, hDD11_i0p1, hDD11_i0p2, invdxx0);
        const REAL hDD_dD120 = fd_function_dD0_fdorder4(hDD12_i0m1, hDD12_i0m2, hDD12_i0p1, hDD12_i0p2, invdxx0);
        const REAL hDD_dD220 = fd_function_dD0_fdorder4(hDD22_i0m1, hDD22_i0m2, hDD22_i0p1, hDD22_i0p2, invdxx0);
        const REAL trK_dD0 = fd_function_dD0_fdorder4(trK_i0m1, trK_i0m2, trK_i0p1, trK_i0p2, invdxx0);
        const REAL vetU_dD00 = fd_function_dD0_fdorder4(vetU0_i0m1, vetU0_i0m2, vetU0_i0p1, vetU0_i0p2, invdxx0);
        const REAL vetU_dD10 = fd_function_dD0_fdorder4(vetU1_i0m1, vetU1_i0m2, vetU1_i0p1, vetU1_i0p2, invdxx0);
        const REAL vetU_dD20 = fd_function_dD0_fdorder4(vetU2_i0m1, vetU2_i0m2, vetU2_i0p1, vetU2_i0p2, invdxx0);

        const REAL alpha_dD1 = fd_function_dD1_fdorder4(alpha_i1m1, alpha_i1m2, alpha_i1p1, alpha_i1p2, invdxx1);
        const REAL cf_dD1 = fd_function_dD1_fdorder4(cf_i1m1, cf_i1m2, cf_i1p1, cf_i1p2, invdxx1);
        const REAL hDD_dD001 = fd_function_dD1_fdorder4(hDD00_i1m1, hDD00_i1m2, hDD00_i1p1, hDD00_i1p2, invdxx1);
        const REAL hDD_dD011 = fd_function_dD1_fdorder4(hDD01_i1m1, hDD01_i1m2, hDD01_i1p1, hDD01_i1p2, invdxx1);
        const REAL hDD_dD021 = fd_function_dD1_fdorder4(hDD02_i1m1, hDD02_i1m2, hDD02_i1p1, hDD02_i1p2, invdxx1);
        const REAL hDD_dD111 = fd_function_dD1_fdorder4(hDD11_i1m1, hDD11_i1m2, hDD11_i1p1, hDD11_i1p2, invdxx1);
        const REAL hDD_dD121 = fd_function_dD1_fdorder4(hDD12_i1m1, hDD12_i1m2, hDD12_i1p1, hDD12_i1p2, invdxx1);
        const REAL hDD_dD221 = fd_function_dD1_fdorder4(hDD22_i1m1, hDD22_i1m2, hDD22_i1p1, hDD22_i1p2, invdxx1);
        const REAL trK_dD1 = fd_function_dD1_fdorder4(trK_i1m1, trK_i1m2, trK_i1p1, trK_i1p2, invdxx1);
        const REAL vetU_dD01 = fd_function_dD1_fdorder4(vetU0_i1m1, vetU0_i1m2, vetU0_i1p1, vetU0_i1p2, invdxx1);
        const REAL vetU_dD11 = fd_function_dD1_fdorder4(vetU1_i1m1, vetU1_i1m2, vetU1_i1p1, vetU1_i1p2, invdxx1);
        const REAL vetU_dD21 = fd_function_dD1_fdorder4(vetU2_i1m1, vetU2_i1m2, vetU2_i1p1, vetU2_i1p2, invdxx1);

        const REAL alpha_dD2 = fd_function_dD2_fdorder4(alpha_i2m1, alpha_i2m2, alpha_i2p1, alpha_i2p2, invdxx2);
        const REAL cf_dD2 = fd_function_dD2_fdorder4(cf_i2m1, cf_i2m2, cf_i2p1, cf_i2p2, invdxx2);
        const REAL hDD_dD002 = fd_function_dD2_fdorder4(hDD00_i2m1, hDD00_i2m2, hDD00_i2p1, hDD00_i2p2, invdxx2);
        const REAL hDD_dD012 = fd_function_dD2_fdorder4(hDD01_i2m1, hDD01_i2m2, hDD01_i2p1, hDD01_i2p2, invdxx2);
        const REAL hDD_dD022 = fd_function_dD2_fdorder4(hDD02_i2m1, hDD02_i2m2, hDD02_i2p1, hDD02_i2p2, invdxx2);
        const REAL hDD_dD112 = fd_function_dD2_fdorder4(hDD11_i2m1, hDD11_i2m2, hDD11_i2p1, hDD11_i2p2, invdxx2);
        const REAL hDD_dD122 = fd_function_dD2_fdorder4(hDD12_i2m1, hDD12_i2m2, hDD12_i2p1, hDD12_i2p2, invdxx2);
        const REAL hDD_dD222 = fd_function_dD2_fdorder4(hDD22_i2m1, hDD22_i2m2, hDD22_i2p1, hDD22_i2p2, invdxx2);
        const REAL trK_dD2 = fd_function_dD2_fdorder4(trK_i2m1, trK_i2m2, trK_i2p1, trK_i2p2, invdxx2);
        const REAL vetU_dD02 = fd_function_dD2_fdorder4(vetU0_i2m1, vetU0_i2m2, vetU0_i2p1, vetU0_i2p2, invdxx2);
        const REAL vetU_dD12 = fd_function_dD2_fdorder4(vetU1_i2m1, vetU1_i2m2, vetU1_i2p1, vetU1_i2p2, invdxx2);
        const REAL vetU_dD22 = fd_function_dD2_fdorder4(vetU2_i2m1, vetU2_i2m2, vetU2_i2p1, vetU2_i2p2, invdxx2);

        const REAL alpha_dDD00 = fd_function_dDD00_fdorder4(alpha, alpha_i0m1, alpha_i0m2, alpha_i0p1, alpha_i0p2, invdxx0);
        const REAL cf_dDD00 = fd_function_dDD00_fdorder4(cf, cf_i0m1, cf_i0m2, cf_i0p1, cf_i0p2, invdxx0);
        const REAL vetU_dDD000 = fd_function_dDD00_fdorder4(vetU0, vetU0_i0m1, vetU0_i0m2, vetU0_i0p1, vetU0_i0p2, invdxx0);
        const REAL vetU_dDD100 = fd_function_dDD00_fdorder4(vetU1, vetU1_i0m1, vetU1_i0m2, vetU1_i0p1, vetU1_i0p2, invdxx0);
        const REAL vetU_dDD200 = fd_function_dDD00_fdorder4(vetU2, vetU2_i0m1, vetU2_i0m2, vetU2_i0p1, vetU2_i0p2, invdxx0);
        const REAL alpha_dDD01 =
            fd_function_dDD01_fdorder4(alpha_i0m1_i1m1, alpha_i0m1_i1m2, alpha_i0m1_i1p1, alpha_i0m1_i1p2, alpha_i0m2_i1m1, alpha_i0m2_i1m2,
                                       alpha_i0m2_i1p1, alpha_i0m2_i1p2, alpha_i0p1_i1m1, alpha_i0p1_i1m2, alpha_i0p1_i1p1, alpha_i0p1_i1p2,
                                       alpha_i0p2_i1m1, alpha_i0p2_i1m2, alpha_i0p2_i1p1, alpha_i0p2_i1p2, invdxx0, invdxx1);
        const REAL cf_dDD01 = fd_function_dDD01_fdorder4(cf_i0m1_i1m1, cf_i0m1_i1m2, cf_i0m1_i1p1, cf_i0m1_i1p2, cf_i0m2_i1m1, cf_i0m2_i1m2,
                                                         cf_i0m2_i1p1, cf_i0m2_i1p2, cf_i0p1_i1m1, cf_i0p1_i1m2, cf_i0p1_i1p1, cf_i0p1_i1p2,
                                                         cf_i0p2_i1m1, cf_i0p2_i1m2, cf_i0p2_i1p1, cf_i0p2_i1p2, invdxx0, invdxx1);
        const REAL vetU_dDD001 =
            fd_function_dDD01_fdorder4(vetU0_i0m1_i1m1, vetU0_i0m1_i1m2, vetU0_i0m1_i1p1, vetU0_i0m1_i1p2, vetU0_i0m2_i1m1, vetU0_i0m2_i1m2,
                                       vetU0_i0m2_i1p1, vetU0_i0m2_i1p2, vetU0_i0p1_i1m1, vetU0_i0p1_i1m2, vetU0_i0p1_i1p1, vetU0_i0p1_i1p2,
                                       vetU0_i0p2_i1m1, vetU0_i0p2_i1m2, vetU0_i0p2_i1p1, vetU0_i0p2_i1p2, invdxx0, invdxx1);        
        const REAL vetU_dDD101 =
            fd_function_dDD01_fdorder4(vetU1_i0m1_i1m1, vetU1_i0m1_i1m2, vetU1_i0m1_i1p1, vetU1_i0m1_i1p2, vetU1_i0m2_i1m1, vetU1_i0m2_i1m2,
                                       vetU1_i0m2_i1p1, vetU1_i0m2_i1p2, vetU1_i0p1_i1m1, vetU1_i0p1_i1m2, vetU1_i0p1_i1p1, vetU1_i0p1_i1p2,
                                       vetU1_i0p2_i1m1, vetU1_i0p2_i1m2, vetU1_i0p2_i1p1, vetU1_i0p2_i1p2, invdxx0, invdxx1);
        const REAL vetU_dDD201 =
            fd_function_dDD01_fdorder4(vetU2_i0m1_i1m1, vetU2_i0m1_i1m2, vetU2_i0m1_i1p1, vetU2_i0m1_i1p2, vetU2_i0m2_i1m1, vetU2_i0m2_i1m2,
                                       vetU2_i0m2_i1p1, vetU2_i0m2_i1p2, vetU2_i0p1_i1m1, vetU2_i0p1_i1m2, vetU2_i0p1_i1p1, vetU2_i0p1_i1p2,
                                       vetU2_i0p2_i1m1, vetU2_i0p2_i1m2, vetU2_i0p2_i1p1, vetU2_i0p2_i1p2, invdxx0, invdxx1);

        const REAL alpha_dDD02 =
            fd_function_dDD02_fdorder4(alpha_i0m1_i2m1, alpha_i0m1_i2m2, alpha_i0m1_i2p1, alpha_i0m1_i2p2, alpha_i0m2_i2m1, alpha_i0m2_i2m2,
                                       alpha_i0m2_i2p1, alpha_i0m2_i2p2, alpha_i0p1_i2m1, alpha_i0p1_i2m2, alpha_i0p1_i2p1, alpha_i0p1_i2p2,
                                       alpha_i0p2_i2m1, alpha_i0p2_i2m2, alpha_i0p2_i2p1, alpha_i0p2_i2p2, invdxx0, invdxx2);
        const REAL cf_dDD02 = fd_function_dDD02_fdorder4(cf_i0m1_i2m1, cf_i0m1_i2m2, cf_i0m1_i2p1, cf_i0m1_i2p2, cf_i0m2_i2m1, cf_i0m2_i2m2,
                                                         cf_i0m2_i2p1, cf_i0m2_i2p2, cf_i0p1_i2m1, cf_i0p1_i2m2, cf_i0p1_i2p1, cf_i0p1_i2p2,
                                                         cf_i0p2_i2m1, cf_i0p2_i2m2, cf_i0p2_i2p1, cf_i0p2_i2p2, invdxx0, invdxx2);        
        const REAL vetU_dDD002 =
            fd_function_dDD02_fdorder4(vetU0_i0m1_i2m1, vetU0_i0m1_i2m2, vetU0_i0m1_i2p1, vetU0_i0m1_i2p2, vetU0_i0m2_i2m1, vetU0_i0m2_i2m2,
                                       vetU0_i0m2_i2p1, vetU0_i0m2_i2p2, vetU0_i0p1_i2m1, vetU0_i0p1_i2m2, vetU0_i0p1_i2p1, vetU0_i0p1_i2p2,
                                       vetU0_i0p2_i2m1, vetU0_i0p2_i2m2, vetU0_i0p2_i2p1, vetU0_i0p2_i2p2, invdxx0, invdxx2);        
        const REAL vetU_dDD102 =
            fd_function_dDD02_fdorder4(vetU1_i0m1_i2m1, vetU1_i0m1_i2m2, vetU1_i0m1_i2p1, vetU1_i0m1_i2p2, vetU1_i0m2_i2m1, vetU1_i0m2_i2m2,
                                       vetU1_i0m2_i2p1, vetU1_i0m2_i2p2, vetU1_i0p1_i2m1, vetU1_i0p1_i2m2, vetU1_i0p1_i2p1, vetU1_i0p1_i2p2,
                                       vetU1_i0p2_i2m1, vetU1_i0p2_i2m2, vetU1_i0p2_i2p1, vetU1_i0p2_i2p2, invdxx0, invdxx2);        
        const REAL vetU_dDD202 =
            fd_function_dDD02_fdorder4(vetU2_i0m1_i2m1, vetU2_i0m1_i2m2, vetU2_i0m1_i2p1, vetU2_i0m1_i2p2, vetU2_i0m2_i2m1, vetU2_i0m2_i2m2,
                                       vetU2_i0m2_i2p1, vetU2_i0m2_i2p2, vetU2_i0p1_i2m1, vetU2_i0p1_i2m2, vetU2_i0p1_i2p1, vetU2_i0p1_i2p2,
                                       vetU2_i0p2_i2m1, vetU2_i0p2_i2m2, vetU2_i0p2_i2p1, vetU2_i0p2_i2p2, invdxx0, invdxx2);
        
        const REAL alpha_dDD11 = fd_function_dDD11_fdorder4(alpha, alpha_i1m1, alpha_i1m2, alpha_i1p1, alpha_i1p2, invdxx1);
        const REAL cf_dDD11 = fd_function_dDD11_fdorder4(cf, cf_i1m1, cf_i1m2, cf_i1p1, cf_i1p2, invdxx1);
        const REAL vetU_dDD011 = fd_function_dDD11_fdorder4(vetU0, vetU0_i1m1, vetU0_i1m2, vetU0_i1p1, vetU0_i1p2, invdxx1);
        const REAL vetU_dDD111 = fd_function_dDD11_fdorder4(vetU1, vetU1_i1m1, vetU1_i1m2, vetU1_i1p1, vetU1_i1p2, invdxx1);
        const REAL vetU_dDD211 = fd_function_dDD11_fdorder4(vetU2, vetU2_i1m1, vetU2_i1m2, vetU2_i1p1, vetU2_i1p2, invdxx1);
        
        const REAL alpha_dDD12 =
            fd_function_dDD12_fdorder4(alpha_i1m1_i2m1, alpha_i1m1_i2m2, alpha_i1m1_i2p1, alpha_i1m1_i2p2, alpha_i1m2_i2m1, alpha_i1m2_i2m2,
                                       alpha_i1m2_i2p1, alpha_i1m2_i2p2, alpha_i1p1_i2m1, alpha_i1p1_i2m2, alpha_i1p1_i2p1, alpha_i1p1_i2p2,
                                       alpha_i1p2_i2m1, alpha_i1p2_i2m2, alpha_i1p2_i2p1, alpha_i1p2_i2p2, invdxx1, invdxx2);
        

        const REAL cf_dDD12 = fd_function_dDD12_fdorder4(cf_i1m1_i2m1, cf_i1m1_i2m2, cf_i1m1_i2p1, cf_i1m1_i2p2, cf_i1m2_i2m1, cf_i1m2_i2m2,
                                                         cf_i1m2_i2p1, cf_i1m2_i2p2, cf_i1p1_i2m1, cf_i1p1_i2m2, cf_i1p1_i2p1, cf_i1p1_i2p2,
                                                         cf_i1p2_i2m1, cf_i1p2_i2m2, cf_i1p2_i2p1, cf_i1p2_i2p2, invdxx1, invdxx2);
        const REAL vetU_dDD012 =
            fd_function_dDD12_fdorder4(vetU0_i1m1_i2m1, vetU0_i1m1_i2m2, vetU0_i1m1_i2p1, vetU0_i1m1_i2p2, vetU0_i1m2_i2m1, vetU0_i1m2_i2m2,
                                       vetU0_i1m2_i2p1, vetU0_i1m2_i2p2, vetU0_i1p1_i2m1, vetU0_i1p1_i2m2, vetU0_i1p1_i2p1, vetU0_i1p1_i2p2,
                                       vetU0_i1p2_i2m1, vetU0_i1p2_i2m2, vetU0_i1p2_i2p1, vetU0_i1p2_i2p2, invdxx1, invdxx2);


        const REAL vetU_dDD112 =
            fd_function_dDD12_fdorder4(vetU1_i1m1_i2m1, vetU1_i1m1_i2m2, vetU1_i1m1_i2p1, vetU1_i1m1_i2p2, vetU1_i1m2_i2m1, vetU1_i1m2_i2m2,
                                       vetU1_i1m2_i2p1, vetU1_i1m2_i2p2, vetU1_i1p1_i2m1, vetU1_i1p1_i2m2, vetU1_i1p1_i2p1, vetU1_i1p1_i2p2,
                                       vetU1_i1p2_i2m1, vetU1_i1p2_i2m2, vetU1_i1p2_i2p1, vetU1_i1p2_i2p2, invdxx1, invdxx2);


        const REAL vetU_dDD212 =
            fd_function_dDD12_fdorder4(vetU2_i1m1_i2m1, vetU2_i1m1_i2m2, vetU2_i1m1_i2p1, vetU2_i1m1_i2p2, vetU2_i1m2_i2m1, vetU2_i1m2_i2m2,
                                       vetU2_i1m2_i2p1, vetU2_i1m2_i2p2, vetU2_i1p1_i2m1, vetU2_i1p1_i2m2, vetU2_i1p1_i2p1, vetU2_i1p1_i2p2,
                                       vetU2_i1p2_i2m1, vetU2_i1p2_i2m2, vetU2_i1p2_i2p1, vetU2_i1p2_i2p2, invdxx1, invdxx2);        
        
        const REAL alpha_dDD22 = fd_function_dDD22_fdorder4(alpha, alpha_i2m1, alpha_i2m2, alpha_i2p1, alpha_i2p2, invdxx2);
        const REAL cf_dDD22 = fd_function_dDD22_fdorder4(cf, cf_i2m1, cf_i2m2, cf_i2p1, cf_i2p2, invdxx2);
        const REAL vetU_dDD022 = fd_function_dDD22_fdorder4(vetU0, vetU0_i2m1, vetU0_i2m2, vetU0_i2p1, vetU0_i2p2, invdxx2);
        const REAL vetU_dDD122 = fd_function_dDD22_fdorder4(vetU1, vetU1_i2m1, vetU1_i2m2, vetU1_i2p1, vetU1_i2p2, invdxx2);
        const REAL vetU_dDD222 = fd_function_dDD22_fdorder4(vetU2, vetU2_i2m1, vetU2_i2m2, vetU2_i2p1, vetU2_i2p2, invdxx2);


        // WARP SYNC NEEDED
        // EACH WARP CAN COMPUTE THE FOLLOWING?

        const REAL FDPart1tmp0 = (1.0 / (f0_of_xx0));
        const REAL UpwindControlVectorU0 = vetU0;
        const REAL UpwindControlVectorU1 = FDPart1tmp0 * vetU1;
        const REAL UpwindControlVectorU2 = FDPart1tmp0 * vetU2 / f1_of_xx1;

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 3:
         * Implement upwinding algorithm.
         * #define UPWIND_ALG(UpwindVecU) UpwindVecU > 0.0 ? 1.0 : 0.0
         */
        const REAL Upwind0 = UPWIND_ALG(UpwindControlVectorU0);
        const REAL Upwind1 = UPWIND_ALG(UpwindControlVectorU1);
        const REAL Upwind2 = UPWIND_ALG(UpwindControlVectorU2);
        
        // auto Upwind_zero_eval = [](const REAL& upwind, const REAL& a, const REAL& b) {return 0.;};
        // auto Upwind_eval_func = [](const REAL& upwind, const REAL& a, const REAL& b) {
        //         return upwind * (-a + b) + a;
        //     };
        
        // std::function<REAL(const REAL&,const REAL&, const REAL&)> Upwind0_eval = (Upwind0 > 0.) ? Upwind_eval_func : Upwind_zero_eval;
        // std::function<REAL(const REAL&,const REAL&, const REAL&)> Upwind1_eval = (Upwind1 > 0.) ? Upwind_eval_func : Upwind_zero_eval;
        // std::function<REAL(const REAL&,const REAL&, const REAL&)> Upwind2_eval = (Upwind2 > 0.) ? Upwind_eval_func : Upwind_zero_eval;
        
        Upwind_eval_base Upwind0_eval(Upwind0);
        Upwind_eval_base Upwind1_eval(Upwind1);
        Upwind_eval_base Upwind2_eval(Upwind2);


        const REAL aDD_dupD000    = Upwind0_eval(UpwindAlgInputaDD_ddnD000, UpwindAlgInputaDD_dupD000);
        const REAL aDD_dupD010    = Upwind0_eval(UpwindAlgInputaDD_ddnD010, UpwindAlgInputaDD_dupD010);
        const REAL aDD_dupD020    = Upwind0_eval(UpwindAlgInputaDD_ddnD020, UpwindAlgInputaDD_dupD020);
        const REAL aDD_dupD110    = Upwind0_eval(UpwindAlgInputaDD_ddnD110, UpwindAlgInputaDD_dupD110);
        const REAL aDD_dupD120    = Upwind0_eval(UpwindAlgInputaDD_ddnD120, UpwindAlgInputaDD_dupD120);
        const REAL aDD_dupD220    = Upwind0_eval(UpwindAlgInputaDD_ddnD220, UpwindAlgInputaDD_dupD220);
        const REAL alpha_dupD0    = Upwind0_eval(UpwindAlgInputalpha_ddnD0, UpwindAlgInputalpha_dupD0);
        const REAL betU_dupD00    = Upwind0_eval(UpwindAlgInputbetU_ddnD00, UpwindAlgInputbetU_dupD00);
        const REAL betU_dupD10    = Upwind0_eval(UpwindAlgInputbetU_ddnD10, UpwindAlgInputbetU_dupD10);
        const REAL betU_dupD20    = Upwind0_eval(UpwindAlgInputbetU_ddnD20, UpwindAlgInputbetU_dupD20);
        const REAL cf_dupD0       = Upwind0_eval(UpwindAlgInputcf_ddnD0, UpwindAlgInputcf_dupD0);
        const REAL hDD_dupD000    = Upwind0_eval(UpwindAlgInputhDD_ddnD000, UpwindAlgInputhDD_dupD000);
        const REAL hDD_dupD010    = Upwind0_eval(UpwindAlgInputhDD_ddnD010, UpwindAlgInputhDD_dupD010);
        const REAL hDD_dupD020    = Upwind0_eval(UpwindAlgInputhDD_ddnD020, UpwindAlgInputhDD_dupD020);
        const REAL hDD_dupD110    = Upwind0_eval(UpwindAlgInputhDD_ddnD110, UpwindAlgInputhDD_dupD110);
        const REAL hDD_dupD120    = Upwind0_eval(UpwindAlgInputhDD_ddnD120, UpwindAlgInputhDD_dupD120);
        const REAL hDD_dupD220    = Upwind0_eval(UpwindAlgInputhDD_ddnD220, UpwindAlgInputhDD_dupD220);
        const REAL lambdaU_dupD00 = Upwind0_eval(UpwindAlgInputlambdaU_ddnD00, UpwindAlgInputlambdaU_dupD00);
        const REAL lambdaU_dupD10 = Upwind0_eval(UpwindAlgInputlambdaU_ddnD10, UpwindAlgInputlambdaU_dupD10);
        const REAL lambdaU_dupD20 = Upwind0_eval(UpwindAlgInputlambdaU_ddnD20, UpwindAlgInputlambdaU_dupD20);
        const REAL trK_dupD0      = Upwind0_eval(UpwindAlgInputtrK_ddnD0, UpwindAlgInputtrK_dupD0);
        const REAL vetU_dupD00    = Upwind0_eval(UpwindAlgInputvetU_ddnD00, UpwindAlgInputvetU_dupD00);
        const REAL vetU_dupD10    = Upwind0_eval(UpwindAlgInputvetU_ddnD10, UpwindAlgInputvetU_dupD10);
        const REAL vetU_dupD20    = Upwind0_eval(UpwindAlgInputvetU_ddnD20, UpwindAlgInputvetU_dupD20);

        const REAL aDD_dupD001    = Upwind1_eval(UpwindAlgInputaDD_ddnD001, UpwindAlgInputaDD_dupD001);
        const REAL aDD_dupD011    = Upwind1_eval(UpwindAlgInputaDD_ddnD011, UpwindAlgInputaDD_dupD011);
        const REAL aDD_dupD021    = Upwind1_eval(UpwindAlgInputaDD_ddnD021, UpwindAlgInputaDD_dupD021);
        const REAL aDD_dupD111    = Upwind1_eval(UpwindAlgInputaDD_ddnD111, UpwindAlgInputaDD_dupD111);
        const REAL aDD_dupD121    = Upwind1_eval(UpwindAlgInputaDD_ddnD121, UpwindAlgInputaDD_dupD121);
        const REAL aDD_dupD221    = Upwind1_eval(UpwindAlgInputaDD_ddnD221, UpwindAlgInputaDD_dupD221);
        const REAL alpha_dupD1    = Upwind1_eval(UpwindAlgInputalpha_ddnD1, UpwindAlgInputalpha_dupD1);
        const REAL betU_dupD01    = Upwind1_eval(UpwindAlgInputbetU_ddnD01, UpwindAlgInputbetU_dupD01);
        const REAL betU_dupD11    = Upwind1_eval(UpwindAlgInputbetU_ddnD11, UpwindAlgInputbetU_dupD11);
        const REAL betU_dupD21    = Upwind1_eval(UpwindAlgInputbetU_ddnD21, UpwindAlgInputbetU_dupD21);
        const REAL cf_dupD1       = Upwind1_eval(UpwindAlgInputcf_ddnD1, UpwindAlgInputcf_dupD1);
        const REAL hDD_dupD001    = Upwind1_eval(UpwindAlgInputhDD_ddnD001, UpwindAlgInputhDD_dupD001);
        const REAL hDD_dupD011    = Upwind1_eval(UpwindAlgInputhDD_ddnD011, UpwindAlgInputhDD_dupD011);
        const REAL hDD_dupD021    = Upwind1_eval(UpwindAlgInputhDD_ddnD021, UpwindAlgInputhDD_dupD021);
        const REAL hDD_dupD111    = Upwind1_eval(UpwindAlgInputhDD_ddnD111, UpwindAlgInputhDD_dupD111);
        const REAL hDD_dupD121    = Upwind1_eval(UpwindAlgInputhDD_ddnD121, UpwindAlgInputhDD_dupD121);
        const REAL hDD_dupD221    = Upwind1_eval(UpwindAlgInputhDD_ddnD221, UpwindAlgInputhDD_dupD221);
        const REAL lambdaU_dupD01 = Upwind1_eval(UpwindAlgInputlambdaU_ddnD01, UpwindAlgInputlambdaU_dupD01);
        const REAL lambdaU_dupD11 = Upwind1_eval(UpwindAlgInputlambdaU_ddnD11, UpwindAlgInputlambdaU_dupD11);
        const REAL lambdaU_dupD21 = Upwind1_eval(UpwindAlgInputlambdaU_ddnD21, UpwindAlgInputlambdaU_dupD21);
        const REAL trK_dupD1      = Upwind1_eval(UpwindAlgInputtrK_ddnD1, UpwindAlgInputtrK_dupD1);
        const REAL vetU_dupD01    = Upwind1_eval(UpwindAlgInputvetU_ddnD01, UpwindAlgInputvetU_dupD01);
        const REAL vetU_dupD11    = Upwind1_eval(UpwindAlgInputvetU_ddnD11, UpwindAlgInputvetU_dupD11);
        const REAL vetU_dupD21    = Upwind1_eval(UpwindAlgInputvetU_ddnD21, UpwindAlgInputvetU_dupD21);

        const REAL aDD_dupD002    = Upwind2_eval(UpwindAlgInputaDD_ddnD002, UpwindAlgInputaDD_dupD002);
        const REAL aDD_dupD012    = Upwind2_eval(UpwindAlgInputaDD_ddnD012, UpwindAlgInputaDD_dupD012);
        const REAL aDD_dupD022    = Upwind2_eval(UpwindAlgInputaDD_ddnD022, UpwindAlgInputaDD_dupD022);
        const REAL aDD_dupD112    = Upwind2_eval(UpwindAlgInputaDD_ddnD112, UpwindAlgInputaDD_dupD112);
        const REAL aDD_dupD122    = Upwind2_eval(UpwindAlgInputaDD_ddnD122, UpwindAlgInputaDD_dupD122);
        const REAL aDD_dupD222    = Upwind2_eval(UpwindAlgInputaDD_ddnD222, UpwindAlgInputaDD_dupD222);
        const REAL alpha_dupD2    = Upwind2_eval(UpwindAlgInputalpha_ddnD2, UpwindAlgInputalpha_dupD2);
        const REAL betU_dupD02    = Upwind2_eval(UpwindAlgInputbetU_ddnD02, UpwindAlgInputbetU_dupD02);
        const REAL betU_dupD12    = Upwind2_eval(UpwindAlgInputbetU_ddnD12, UpwindAlgInputbetU_dupD12);
        const REAL betU_dupD22    = Upwind2_eval(UpwindAlgInputbetU_ddnD22, UpwindAlgInputbetU_dupD22);
        const REAL cf_dupD2       = Upwind2_eval(UpwindAlgInputcf_ddnD2, UpwindAlgInputcf_dupD2);
        const REAL hDD_dupD002    = Upwind2_eval(UpwindAlgInputhDD_ddnD002, UpwindAlgInputhDD_dupD002);
        const REAL hDD_dupD012    = Upwind2_eval(UpwindAlgInputhDD_ddnD012, UpwindAlgInputhDD_dupD012);
        const REAL hDD_dupD022    = Upwind2_eval(UpwindAlgInputhDD_ddnD022, UpwindAlgInputhDD_dupD022);
        const REAL hDD_dupD112    = Upwind2_eval(UpwindAlgInputhDD_ddnD112, UpwindAlgInputhDD_dupD112);
        const REAL hDD_dupD122    = Upwind2_eval(UpwindAlgInputhDD_ddnD122, UpwindAlgInputhDD_dupD122);
        const REAL hDD_dupD222    = Upwind2_eval(UpwindAlgInputhDD_ddnD222, UpwindAlgInputhDD_dupD222);
        const REAL lambdaU_dupD02 = Upwind2_eval(UpwindAlgInputlambdaU_ddnD02, UpwindAlgInputlambdaU_dupD02);
        const REAL lambdaU_dupD12 = Upwind2_eval(UpwindAlgInputlambdaU_ddnD12, UpwindAlgInputlambdaU_dupD12);
        const REAL lambdaU_dupD22 = Upwind2_eval(UpwindAlgInputlambdaU_ddnD22, UpwindAlgInputlambdaU_dupD22);
        const REAL trK_dupD2      = Upwind2_eval(UpwindAlgInputtrK_ddnD2, UpwindAlgInputtrK_dupD2);
        const REAL vetU_dupD02    = Upwind2_eval(UpwindAlgInputvetU_ddnD02, UpwindAlgInputvetU_dupD02);
        const REAL vetU_dupD12    = Upwind2_eval(UpwindAlgInputvetU_ddnD12, UpwindAlgInputvetU_dupD12);
        const REAL vetU_dupD22    = Upwind2_eval(UpwindAlgInputvetU_ddnD22, UpwindAlgInputvetU_dupD22);
        // WARP SYNC HERE

        /*
         * NRPy+-Generated GF Access/FD Code, Step 3 of 3:
         * Evaluate SymPy expressions and write to main memory.
         */
        
        // FDPart3tmp BLOCK 1
        const REAL FDPart3tmp1 = alpha * trK;
        const REAL FDPart3tmp2 = (1.0 / (f0_of_xx0));
        const REAL FDPart3tmp4 = (1.0 / (f1_of_xx1));
        const REAL FDPart3tmp7 = ((f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp12 = 2. * f0_of_xx0;
        const REAL FDPart3tmp18 = aDD02 * f1_of_xx1;
        const REAL FDPart3tmp28 = aDD01 * f0_of_xx0;
        const REAL FDPart3tmp29 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp30 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp31 = hDD00 + 1;
        const REAL FDPart3tmp42 = f1_of_xx1 * hDD02;
        const REAL FDPart3tmp47 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp56 = f0_of_xx0 * hDD01;
        const REAL FDPart3tmp68 = 2. * alpha;
        const REAL FDPart3tmp70 = ((cf) * (cf));
        const REAL FDPart3tmp71 = (1.0 / (cf));
        const REAL FDPart3tmp75 = 2. * hDD01;
        const REAL FDPart3tmp102 = f0_of_xx0 * hDD_dD012;
        const REAL FDPart3tmp105 = f0_of_xx0 * f1_of_xx1;
        const REAL FDPart3tmp166 = f1_of_xx1 * f1_of_xx1__D1;
        const REAL FDPart3tmp200 = f0_of_xx0 * vetU_dD00;
        const REAL FDPart3tmp244 = alpha * trK_dD2;
        const REAL FDPart3tmp245 = alpha * trK_dD1;
        const REAL FDPart3tmp247 = alpha * trK_dD0;
        const REAL FDPart3tmp271 = ((f1_of_xx1__D1) * (f1_of_xx1__D1));
        const REAL FDPart3tmp290 = betU0 * vetU0;
        const REAL FDPart3tmp291 = (3.0 / 4.0) * lambdaU0;
        const REAL FDPart3tmp317 = 3. * alpha;
        const REAL FDPart3tmp351 = FDPart1_Rational_3_2 * alpha;
        const REAL FDPart3tmp440 = (1.0 / ((f1_of_xx1) * (f1_of_xx1) * (f1_of_xx1)));
        const REAL FDPart3tmp457 = 4. * alpha;
        const REAL FDPart3tmp465 = ((vetU0) * (vetU0));
        // END FDPart3tmp BLOCK 1

        // FDPart3tmp BLOCK 2
        const REAL FDPart3tmp3 = FDPart3tmp2 * vetU1;
        const REAL FDPart3tmp5 = FDPart3tmp2 * FDPart3tmp4;
        const REAL FDPart3tmp8 = (1.0 / (FDPart3tmp7));
        const REAL FDPart3tmp13 = FDPart3tmp12 * aDD01;
        const REAL FDPart3tmp14 = FDPart3tmp4 * vetU2;
        const REAL FDPart3tmp19 = FDPart3tmp12 * FDPart3tmp18;
        const REAL FDPart3tmp20 = FDPart3tmp2 * vetU_dD11;
        const REAL FDPart3tmp21 = 2. * FDPart3tmp2;
        const REAL FDPart3tmp24 = FDPart3tmp4 * f1_of_xx1__D1;
        const REAL FDPart3tmp32 = FDPart3tmp29 * FDPart3tmp30 * ((hDD12) * (hDD12));
        const REAL FDPart3tmp33 = FDPart3tmp7 * hDD11;
        const REAL FDPart3tmp35 = FDPart3tmp30 * FDPart3tmp7;
        const REAL FDPart3tmp39 = FDPart3tmp7 * ((hDD01) * (hDD01));
        const REAL FDPart3tmp43 = FDPart3tmp7 * f1_of_xx1;
        const REAL FDPart3tmp49 = FDPart3tmp42 * f0_of_xx0;
        const REAL FDPart3tmp54 = FDPart3tmp18 * f0_of_xx0;
        const REAL FDPart3tmp69 = FDPart3tmp68 * aDD00;
        const REAL FDPart3tmp76 = FDPart3tmp12 * hDD_dD010 + FDPart3tmp75 - hDD_dD001;
        const REAL FDPart3tmp78 = FDPart3tmp12 * f1_of_xx1;
        const REAL FDPart3tmp83 = FDPart1_Rational_1_2 * FDPart3tmp71;
        const REAL FDPart3tmp94 = (1.0 / (FDPart3tmp70));
        const REAL FDPart3tmp95 = FDPart3tmp71 * cf_dD0;
        const REAL FDPart3tmp99 = FDPart3tmp12 * hDD11 + FDPart3tmp12;
        const REAL FDPart3tmp104 = f0_of_xx0 * f1_of_xx1__D1 * hDD02;
        const REAL FDPart3tmp120 = FDPart3tmp71 * cf_dD1;
        const REAL FDPart3tmp132 = FDPart3tmp7 * hDD_dD112;
        const REAL FDPart3tmp149 = FDPart3tmp12 * FDPart3tmp30;
        const REAL FDPart3tmp159 = FDPart3tmp71 * cf_dD2;
        const REAL FDPart3tmp201 = FDPart3tmp7 * aDD11;
        const REAL FDPart3tmp202 = FDPart3tmp2 * vetU2;
        const REAL FDPart3tmp203 = (1.0 / (FDPart3tmp30));
        const REAL FDPart3tmp223 = FDPart1_Rational_2_3 * FDPart3tmp7;
        const REAL FDPart3tmp239 = FDPart3tmp2 * vetU_dD01;
        const REAL FDPart3tmp241 = FDPart3tmp2 * vetU_dD02;
        const REAL FDPart3tmp242 = FDPart3tmp4 * lambdaU2;
        const REAL FDPart3tmp252 = FDPart3tmp30 * f0_of_xx0;
        const REAL FDPart3tmp270 = 2. * FDPart3tmp29;
        const REAL FDPart3tmp273 = 4. * FDPart3tmp30 * FDPart3tmp47;
        const REAL FDPart3tmp274 = FDPart3tmp2 * vetU0;
        const REAL FDPart3tmp292 = FDPart3tmp291 * vetU0;
        const REAL FDPart3tmp294 = FDPart3tmp2 * betU1;
        const REAL FDPart3tmp412 = (1.0 / (FDPart3tmp47));
        // END FDPart3tmp BLOCK 2

        // FDPart3tmp BLOCK 3
        const REAL FDPart3tmp6 = FDPart3tmp5 * vetU2;
        const REAL FDPart3tmp9 = FDPart3tmp8 * vetU1;
        const REAL FDPart3tmp15 = FDPart3tmp14 * FDPart3tmp8;
        const REAL FDPart3tmp34 = FDPart3tmp33 + FDPart3tmp7;
        const REAL FDPart3tmp36 = FDPart3tmp35 * ((hDD02) * (hDD02));
        const REAL FDPart3tmp37 = FDPart3tmp35 * hDD22;
        const REAL FDPart3tmp44 = FDPart3tmp43 * hDD12;
        const REAL FDPart3tmp55 = FDPart3tmp19 * alpha;
        const REAL FDPart3tmp60 = FDPart3tmp13 * alpha;
        const REAL FDPart3tmp79 = 2. * FDPart3tmp42 + FDPart3tmp78 * hDD_dD020 - hDD_dD002;
        const REAL FDPart3tmp84 = FDPart3tmp83 * cf_dD2;
        const REAL FDPart3tmp87 = FDPart3tmp83 * cf_dD1;
        const REAL FDPart3tmp93 = FDPart3tmp83 * cf_dD0;
        const REAL FDPart3tmp100 = FDPart3tmp7 * hDD_dD110 + FDPart3tmp99;
        const REAL FDPart3tmp107 = FDPart3tmp104 + FDPart3tmp105 * hDD_dD021;
        const REAL FDPart3tmp108 = FDPart3tmp78 * hDD12;
        const REAL FDPart3tmp123 = FDPart3tmp71 * alpha_dD1 * cf_dD0;
        const REAL FDPart3tmp130 = FDPart1_Rational_1_2 * FDPart3tmp7 * hDD_dD111;
        const REAL FDPart3tmp135 = FDPart3tmp7 * f1_of_xx1__D1 * hDD12;
        const REAL FDPart3tmp150 = FDPart3tmp149 * hDD22 + FDPart3tmp149;
        const REAL FDPart3tmp168 = 2. * FDPart3tmp166 * FDPart3tmp7;
        const REAL FDPart3tmp179 = FDPart3tmp71 * alpha_dD1 * cf_dD2;
        const REAL FDPart3tmp207 = FDPart3tmp43 * aDD12;
        const REAL FDPart3tmp214 = FDPart3tmp35 * aDD22;
        const REAL FDPart3tmp226 = FDPart3tmp201 * FDPart3tmp68;
        const REAL FDPart3tmp233 = FDPart3tmp4 * FDPart3tmp8;
        const REAL FDPart3tmp237 = FDPart3tmp203 * FDPart3tmp8;
        const REAL FDPart3tmp255 = FDPart3tmp2 * FDPart3tmp24 * vetU_dD12;
        const REAL FDPart3tmp257 = 2. * FDPart3tmp241 + FDPart3tmp5 * vetU_dDD222;
        const REAL FDPart3tmp272 = FDPart1_Rational_1_2 * FDPart3tmp203 / FDPart3tmp29;
        const REAL FDPart3tmp278 = FDPart3tmp2 * FDPart3tmp203;
        const REAL FDPart3tmp293 = FDPart3tmp3 * betU0;
        const REAL FDPart3tmp295 = FDPart3tmp294 * vetU0;
        const REAL FDPart3tmp296 = FDPart3tmp291 * FDPart3tmp3;
        const REAL FDPart3tmp298 = (3.0 / 4.0) * FDPart3tmp274 * lambdaU1;
        const REAL FDPart3tmp302 = FDPart3tmp5 * betU2;
        const REAL FDPart3tmp306 = (3.0 / 4.0) * FDPart3tmp5 * lambdaU2 * vetU0;
        const REAL FDPart3tmp318 = FDPart3tmp317 * FDPart3tmp95 + alpha_dD0;
        const REAL FDPart3tmp367 = FDPart3tmp159 * FDPart3tmp317 + alpha_dD2;
        const REAL FDPart3tmp377 = FDPart3tmp120 * FDPart3tmp317 + alpha_dD1;
        const REAL FDPart3tmp436 = 2. * FDPart3tmp14;
        const REAL FDPart3tmp466 = 2. * FDPart3tmp3 * vetU0;
        const REAL FDPart3tmp467 = FDPart3tmp8 * ((vetU1) * (vetU1));
        // END FDPart3tmp BLOCK 3

        // FDPart3tmp BLOCK 4
        const REAL FDPart3tmp26 = FDPart3tmp20 + FDPart3tmp21 * vetU0 + FDPart3tmp24 * FDPart3tmp3 + FDPart3tmp5 * vetU_dD22 + vetU_dD00;
        const REAL FDPart3tmp38 = FDPart3tmp35 + FDPart3tmp37;
        const REAL FDPart3tmp45 = -FDPart3tmp31 * FDPart3tmp44 + FDPart3tmp42 * FDPart3tmp7 * hDD01;
        const REAL FDPart3tmp50 = -FDPart3tmp34 * FDPart3tmp49 + FDPart3tmp47 * f1_of_xx1 * hDD01 * hDD12;
        const REAL FDPart3tmp52 = FDPart3tmp31 * FDPart3tmp34 - FDPart3tmp39;
        const REAL FDPart3tmp110 = FDPart3tmp108 + FDPart3tmp43 * hDD_dD120;
        const REAL FDPart3tmp131 = -FDPart3tmp100 + 2 * f0_of_xx0 * hDD_dD011;
        const REAL FDPart3tmp136 = -FDPart3tmp132 + 2 * FDPart3tmp135 + 2 * FDPart3tmp43 * hDD_dD121;
        const REAL FDPart3tmp151 = FDPart3tmp150 + FDPart3tmp35 * hDD_dD220;
        const REAL FDPart3tmp169 = FDPart3tmp168 * hDD22 + FDPart3tmp168;
        const REAL FDPart3tmp172 = FDPart3tmp102 + FDPart3tmp107 - FDPart3tmp108 - FDPart3tmp43 * hDD_dD120;
        const REAL FDPart3tmp186 = FDPart1_Rational_1_2 * FDPart3tmp35 * hDD_dD222;
        const REAL FDPart3tmp205 = -FDPart3tmp202 * FDPart3tmp203 * f1_of_xx1__D1;
        const REAL FDPart3tmp222 = 2. * FDPart3tmp43 * aDD12;
        const REAL FDPart3tmp225 = FDPart3tmp207 * FDPart3tmp68;
        const REAL FDPart3tmp258 = FDPart3tmp2 * vetU_dDD112 + FDPart3tmp255 + FDPart3tmp257 + vetU_dDD002;
        const REAL FDPart3tmp277 = FDPart3tmp203 * FDPart3tmp271 * FDPart3tmp3;
        const REAL FDPart3tmp280 = FDPart3tmp278 * f1_of_xx1__D1 * vetU_dD22;
        const REAL FDPart3tmp286 = FDPart3tmp2 * vetU_dDD101 - FDPart3tmp8 * vetU_dD11;
        const REAL FDPart3tmp287 = -FDPart3tmp233 * vetU_dD22 + FDPart3tmp5 * vetU_dDD202;
        const REAL FDPart3tmp299 = FDPart3tmp9 * betU1;
        const REAL FDPart3tmp300 = (3.0 / 4.0) * FDPart3tmp9 * lambdaU1;
        const REAL FDPart3tmp301 = FDPart3tmp6 * betU0;
        const REAL FDPart3tmp303 = FDPart3tmp302 * vetU0;
        const REAL FDPart3tmp304 = FDPart3tmp291 * FDPart3tmp6;
        const REAL FDPart3tmp308 = FDPart3tmp14 * FDPart3tmp8 * betU1;
        const REAL FDPart3tmp312 = (3.0 / 4.0) * FDPart3tmp14 * FDPart3tmp8 * lambdaU1;
        const REAL FDPart3tmp314 = FDPart3tmp237 * vetU2;
        const REAL FDPart3tmp401 = FDPart3tmp8 * lambdaU1 * vetU_dD11;
        const REAL FDPart3tmp403 = FDPart3tmp242 * FDPart3tmp8 * vetU_dD12;
        const REAL FDPart3tmp406 = FDPart3tmp2 * vetU_dD00 - FDPart3tmp8 * vetU0;
        const REAL FDPart3tmp416 = FDPart3tmp4 * FDPart3tmp9 * betU2;
        const REAL FDPart3tmp417 = (3.0 / 4.0) * FDPart3tmp242 * FDPart3tmp9;
        const REAL FDPart3tmp427 = FDPart3tmp237 * lambdaU2 * vetU_dD22;
        const REAL FDPart3tmp429 = FDPart3tmp278 * f1_of_xx1__D1;
        const REAL FDPart3tmp434 = -FDPart3tmp203 * FDPart3tmp271 + FDPart3tmp4 * f1_of_xx1__DD11;
        const REAL FDPart3tmp468 = FDPart3tmp14 * FDPart3tmp21 * vetU0;
        const REAL FDPart3tmp469 = FDPart3tmp237 * ((vetU2) * (vetU2));
        // END FDPart3tmp BLOCK 4


        // FDPart3tmp BLOCK 5
        const REAL FDPart3tmp11 = FDPart3tmp2 * vetU_dD10 - FDPart3tmp9;
        const REAL FDPart3tmp17 = -FDPart3tmp15 + FDPart3tmp5 * vetU_dD20;
        const REAL FDPart3tmp27 = FDPart1_Rational_2_3 * FDPart3tmp26;
        const REAL FDPart3tmp40 = 2. * FDPart3tmp29 * FDPart3tmp30 * hDD01 * hDD02 * hDD12 - FDPart3tmp31 * FDPart3tmp32 +
                                  FDPart3tmp31 * FDPart3tmp34 * FDPart3tmp38 - FDPart3tmp34 * FDPart3tmp36 - FDPart3tmp38 * FDPart3tmp39;
        const REAL FDPart3tmp57 = FDPart3tmp30 * FDPart3tmp47 * hDD02 * hDD12 - FDPart3tmp38 * FDPart3tmp56;
        const REAL FDPart3tmp58 = FDPart3tmp31 * FDPart3tmp38 - FDPart3tmp36;
        const REAL FDPart3tmp65 = -FDPart3tmp32 + FDPart3tmp34 * FDPart3tmp38;
        const REAL FDPart3tmp111 = -FDPart3tmp102 + FDPart3tmp107 + FDPart3tmp110;
        const REAL FDPart3tmp147 = FDPart3tmp102 - FDPart3tmp104 - FDPart3tmp105 * hDD_dD021 + FDPart3tmp110;
        const REAL FDPart3tmp170 = FDPart3tmp169 + FDPart3tmp35 * hDD_dD221;
        const REAL FDPart3tmp189 = -FDPart3tmp151 + 2 * f0_of_xx0 * f1_of_xx1 * hDD_dD022;
        const REAL FDPart3tmp206 = FDPart3tmp205 + FDPart3tmp5 * vetU_dD21;
        const REAL FDPart3tmp259 = FDPart1_Rational_1_4 * FDPart3tmp258;
        const REAL FDPart3tmp281 = -FDPart3tmp280 + FDPart3tmp5 * vetU_dDD212;
        const REAL FDPart3tmp315 = FDPart3tmp314 * betU2;
        const REAL FDPart3tmp316 = (3.0 / 4.0) * FDPart3tmp314 * lambdaU2;
        const REAL FDPart3tmp346 = FDPart3tmp45 * FDPart3tmp50;
        const REAL FDPart3tmp452 = (1.0 / 3.0) * FDPart3tmp258;
        // END FDPart3tmp BLOCK 5

        // FDPart3tmp BLOCK 6
        const REAL FDPart3tmp41 = (1.0 / (FDPart3tmp40));
        const REAL FDPart3tmp187 = -FDPart3tmp170 + 2 * FDPart3tmp7 * f1_of_xx1 * hDD_dD122;
        const REAL FDPart3tmp260 = FDPart3tmp11 * f0_of_xx0;
        const REAL FDPart3tmp266 = -FDPart3tmp206 * FDPart3tmp252 - FDPart3tmp24 * vetU_dD02 - vetU_dD12 + vetU_dDD012;
        const REAL FDPart3tmp282 = FDPart3tmp2 * vetU_dDD111 - 4.* FDPart3tmp24 * FDPart3tmp274 +
                                   FDPart3tmp272 * (8. * FDPart3tmp166 * FDPart3tmp47 * vetU0 + FDPart3tmp273 * vetU_dD01) +
                                   FDPart3tmp272 * (2. * FDPart3tmp166 * FDPart3tmp47 * vetU_dD11 +
                                                    FDPart3tmp3 * (FDPart3tmp270 * FDPart3tmp271 + FDPart3tmp270 * f1_of_xx1 * f1_of_xx1__DD11)) -
                                   2. * FDPart3tmp277 + FDPart3tmp281 + vetU_dDD001;
        const REAL FDPart3tmp288 = -4 * FDPart3tmp24 * FDPart3tmp9 + FDPart3tmp272 * (FDPart3tmp273 * vetU_dD00 + 12. * FDPart3tmp35 * vetU0) +
                                   FDPart3tmp272 * (FDPart3tmp11 * FDPart3tmp166 * FDPart3tmp270 + 8. * FDPart3tmp166 * FDPart3tmp7 * vetU1) +
                                   FDPart3tmp286 + FDPart3tmp287 - 8. * FDPart3tmp8 * vetU0 + vetU_dDD000;
        const REAL FDPart3tmp319 = (1.0 / ((FDPart3tmp40) * (FDPart3tmp40)));
        const REAL FDPart3tmp328 = FDPart3tmp50 * FDPart3tmp57;
        const REAL FDPart3tmp358 = FDPart3tmp45 * FDPart3tmp57;
        const REAL FDPart3tmp359 = FDPart3tmp45 * FDPart3tmp58;
        const REAL FDPart3tmp372 = FDPart3tmp52 * FDPart3tmp57;
        const REAL FDPart3tmp379 = FDPart3tmp50 * FDPart3tmp58;
        const REAL FDPart3tmp408 = -FDPart3tmp166 * FDPart3tmp17 + FDPart3tmp2 * vetU_dDD102 - FDPart3tmp8 * vetU_dD12;
        const REAL FDPart3tmp410 = FDPart3tmp166 * FDPart3tmp206;
        const REAL FDPart3tmp430 = FDPart3tmp2 * FDPart3tmp206 * lambdaU1;
        const REAL FDPart3tmp432 = FDPart3tmp11 * FDPart3tmp24 + FDPart3tmp287 + FDPart3tmp406;
        const REAL FDPart3tmp438 = FDPart3tmp17 * FDPart3tmp24 - FDPart3tmp233 * vetU_dD21 + FDPart3tmp314 * f1_of_xx1__D1 -
                                   FDPart3tmp429 * vetU_dD20 + FDPart3tmp5 * vetU_dDD201;
        // END sFDPart3tmp BLOCK 6
        
        // FDPart3tmp BLOCK 7
        const REAL FDPart3tmp46 = FDPart3tmp41 * FDPart3tmp45;
        const REAL FDPart3tmp53 = FDPart3tmp41 * FDPart3tmp52;
        const REAL FDPart3tmp59 = FDPart3tmp41 * FDPart3tmp58;
        const REAL FDPart3tmp61 = FDPart3tmp41 * FDPart3tmp50;
        const REAL FDPart3tmp63 = FDPart3tmp41 * FDPart3tmp57;
        const REAL FDPart3tmp66 = FDPart3tmp41 * FDPart3tmp65;
        const REAL FDPart3tmp74 = FDPart1_Rational_1_2 * FDPart3tmp41 * hDD_dD000;
        const REAL FDPart3tmp261 = -FDPart3tmp239 - FDPart3tmp260 + vetU_dDD001;
        const REAL FDPart3tmp264 = -FDPart3tmp17 * FDPart3tmp252 - FDPart3tmp241 + vetU_dDD002;
        const REAL FDPart3tmp283 = FDPart1_Rational_1_4 * FDPart3tmp282;
        const REAL FDPart3tmp289 = FDPart1_Rational_1_4 * FDPart3tmp288;
        const REAL FDPart3tmp321 = FDPart3tmp319 * ((FDPart3tmp50) * (FDPart3tmp50));
        const REAL FDPart3tmp324 = FDPart3tmp319 * ((FDPart3tmp57) * (FDPart3tmp57));
        const REAL FDPart3tmp326 = FDPart3tmp319 * aDD00;
        const REAL FDPart3tmp329 = FDPart3tmp207 * FDPart3tmp319;
        const REAL FDPart3tmp331 = FDPart3tmp319 * FDPart3tmp54;
        const REAL FDPart3tmp334 = FDPart3tmp28 * FDPart3tmp319;
        const REAL FDPart3tmp340 = FDPart3tmp201 * FDPart3tmp319;
        const REAL FDPart3tmp343 = FDPart3tmp214 * FDPart3tmp319;
        const REAL FDPart3tmp345 = FDPart3tmp13 * FDPart3tmp319;
        const REAL FDPart3tmp347 = FDPart3tmp222 * FDPart3tmp319;
        const REAL FDPart3tmp349 = FDPart3tmp19 * FDPart3tmp319;
        const REAL FDPart3tmp354 = FDPart3tmp319 * ((FDPart3tmp45) * (FDPart3tmp45));
        const REAL FDPart3tmp453 = (1.0 / 3.0) * FDPart3tmp282;
        const REAL FDPart3tmp454 = (1.0 / 3.0) * FDPart3tmp288;
        // END FDPart3tmp BLOCK 7

        // FDPart3tmp BLOCK 8
        const REAL FDPart3tmp81 = FDPart1_Rational_1_2 * FDPart3tmp46 * FDPart3tmp76 + FDPart3tmp50 * FDPart3tmp74 + FDPart1_Rational_1_2 * FDPart3tmp53 * FDPart3tmp79;
        const REAL FDPart3tmp85 = FDPart1_Rational_1_2 * FDPart3tmp46 * FDPart3tmp79 + FDPart3tmp57 * FDPart3tmp74 + FDPart1_Rational_1_2 * FDPart3tmp59 * FDPart3tmp76;
        const REAL FDPart3tmp89 = FDPart1_Rational_1_2 * FDPart3tmp41 * FDPart3tmp50;
        const REAL FDPart3tmp90 = FDPart1_Rational_1_2 * FDPart3tmp41 * FDPart3tmp57;
        const REAL FDPart3tmp117 = FDPart1_Rational_1_2 * FDPart3tmp66;
        const REAL FDPart3tmp127 = 2. * FDPart3tmp41 * FDPart3tmp57;
        const REAL FDPart3tmp164 = 2. * FDPart3tmp41 * FDPart3tmp50;
        const REAL FDPart3tmp208 = FDPart3tmp201 * FDPart3tmp46 + FDPart3tmp207 * FDPart3tmp53 + FDPart3tmp28 * FDPart3tmp61;
        const REAL FDPart3tmp212 = FDPart3tmp201 * FDPart3tmp63 + FDPart3tmp207 * FDPart3tmp61 + FDPart3tmp28 * FDPart3tmp66;
        const REAL FDPart3tmp217 = FDPart3tmp207 * FDPart3tmp59 + FDPart3tmp214 * FDPart3tmp46 + FDPart3tmp54 * FDPart3tmp63;
        const REAL FDPart3tmp218 = FDPart3tmp207 * FDPart3tmp63 + FDPart3tmp214 * FDPart3tmp61 + FDPart3tmp54 * FDPart3tmp66;
        const REAL FDPart3tmp250 = FDPart3tmp59 * (FDPart3tmp200 - vetU0 - 2 * vetU_dD11 + vetU_dDD011);
        const REAL FDPart3tmp253 = FDPart3tmp53 * (-2. * FDPart3tmp166 * vetU1 + FDPart3tmp166 * vetU_dD01 + FDPart3tmp252 * vetU_dD00 -
                                                   FDPart3tmp30 * vetU0 - 2. * f1_of_xx1 * vetU_dD22 + vetU_dDD022);
        const REAL FDPart3tmp262 = FDPart1_Rational_3_2 * FDPart3tmp63;
        const REAL FDPart3tmp265 = FDPart1_Rational_3_2 * FDPart3tmp61;
        const REAL FDPart3tmp327 = FDPart3tmp326 * ((FDPart3tmp65) * (FDPart3tmp65));
        const REAL FDPart3tmp330 = 4. * FDPart3tmp329;
        const REAL FDPart3tmp332 = FDPart3tmp331 * FDPart3tmp65;
        const REAL FDPart3tmp335 = FDPart3tmp334 * FDPart3tmp65;
        const REAL FDPart3tmp341 = FDPart3tmp340 * ((FDPart3tmp45) * (FDPart3tmp45));
        const REAL FDPart3tmp342 = FDPart3tmp326 * ((FDPart3tmp50) * (FDPart3tmp50));
        const REAL FDPart3tmp344 = FDPart3tmp343 * ((FDPart3tmp52) * (FDPart3tmp52));
        const REAL FDPart3tmp356 = FDPart3tmp326 * ((FDPart3tmp57) * (FDPart3tmp57));
        const REAL FDPart3tmp357 = FDPart3tmp340 * ((FDPart3tmp58) * (FDPart3tmp58));
        const REAL FDPart3tmp362 = FDPart3tmp349 * FDPart3tmp65;
        const REAL FDPart3tmp363 = FDPart3tmp345 * FDPart3tmp65;
        const REAL FDPart3tmp394 = FDPart1_Rational_2_3 * FDPart3tmp66;
        const REAL FDPart3tmp395 = FDPart1_Rational_4_3 * FDPart3tmp63;
        const REAL FDPart3tmp396 = FDPart1_Rational_4_3 * FDPart3tmp46;
        const REAL FDPart3tmp397 = FDPart1_Rational_4_3 * FDPart3tmp61;
        const REAL FDPart3tmp398 = FDPart1_Rational_2_3 * FDPart3tmp59;
        const REAL FDPart3tmp399 = FDPart1_Rational_2_3 * FDPart3tmp53;
        const REAL FDPart3tmp409 = FDPart3tmp59 * (FDPart3tmp2 * vetU_dDD111 + 2 * FDPart3tmp239 + FDPart3tmp260);
        const REAL FDPart3tmp411 = FDPart3tmp46 * (FDPart3tmp2 * vetU_dDD112 + FDPart3tmp241 - FDPart3tmp255 - FDPart3tmp410);
        const REAL FDPart3tmp413 =
            FDPart3tmp66 * (FDPart3tmp11 * FDPart3tmp21 + FDPart3tmp2 * vetU_dDD100 + 2 * FDPart3tmp412 * vetU1 - 2 * FDPart3tmp8 * vetU_dD10);
        const REAL FDPart3tmp414 =
            FDPart3tmp53 * (FDPart3tmp11 * FDPart3tmp252 + FDPart3tmp166 * FDPart3tmp20 + FDPart3tmp2 * vetU_dDD122 -
                            FDPart3tmp21 * f1_of_xx1__D1 * vetU_dD22 - FDPart3tmp271 * FDPart3tmp3 + FDPart3tmp3 * FDPart3tmp30);
        const REAL FDPart3tmp415 =
            FDPart3tmp46 * (FDPart3tmp2 * vetU_dDD112 - FDPart3tmp202 * f1_of_xx1 + FDPart3tmp241 - FDPart3tmp255 + FDPart3tmp271 * FDPart3tmp6 -
                            FDPart3tmp410 + FDPart3tmp6 * (-FDPart3tmp271 - f1_of_xx1 * f1_of_xx1__DD11));
        const REAL FDPart3tmp433 = FDPart3tmp46 * (FDPart3tmp2 * FDPart3tmp4 * f1_of_xx1__D1 * vetU_dD11 + FDPart3tmp2 * FDPart3tmp4 * vetU_dDD212 +
                                                   FDPart3tmp239 - FDPart3tmp277 - FDPart3tmp280 - FDPart3tmp3);
        const REAL FDPart3tmp435 = FDPart3tmp46 * (FDPart3tmp20 * FDPart3tmp24 + FDPart3tmp239 + FDPart3tmp281 + FDPart3tmp3 * FDPart3tmp434);
        const REAL FDPart3tmp437 =
            FDPart3tmp66 * (FDPart3tmp17 * FDPart3tmp21 - 2 * FDPart3tmp233 * vetU_dD20 + FDPart3tmp412 * FDPart3tmp436 + FDPart3tmp5 * vetU_dDD200);
        const REAL FDPart3tmp439 =
            FDPart3tmp53 * (FDPart3tmp17 * FDPart3tmp252 + FDPart3tmp21 * FDPart3tmp24 * vetU_dD12 + FDPart3tmp257 + FDPart3tmp410);
        const REAL FDPart3tmp442 = FDPart3tmp59 * (FDPart3tmp17 * f0_of_xx0 + FDPart3tmp202 * FDPart3tmp271 * FDPart3tmp440 -
                                                   FDPart3tmp203 * FDPart3tmp21 * f1_of_xx1__D1 * vetU_dD21 + 2 * FDPart3tmp206 * FDPart3tmp24 +
                                                   FDPart3tmp434 * FDPart3tmp6 + FDPart3tmp5 * vetU_dDD211 + FDPart3tmp6 +
                                                   vetU2 * (2 * FDPart3tmp2 * FDPart3tmp271 * FDPart3tmp440 - FDPart3tmp278 * f1_of_xx1__DD11));
        // END FDPart3tmp BLOCK 8
        
        // FDPart3tmp BLOCK 9
        const REAL FDPart3tmp91 = FDPart1_Rational_1_2 * FDPart3tmp66 * hDD_dD000 + FDPart3tmp76 * FDPart3tmp90 + FDPart3tmp79 * FDPart3tmp89;
        const REAL FDPart3tmp113 = FDPart1_Rational_1_2 * FDPart3tmp100 * FDPart3tmp46 + FDPart1_Rational_1_2 * FDPart3tmp111 * FDPart3tmp53 + FDPart3tmp89 * hDD_dD001;
        const REAL FDPart3tmp115 = FDPart1_Rational_1_2 * FDPart3tmp100 * FDPart3tmp59 + FDPart1_Rational_1_2 * FDPart3tmp111 * FDPart3tmp46 + FDPart3tmp90 * hDD_dD001;
        const REAL FDPart3tmp118 = FDPart3tmp100 * FDPart3tmp90 + FDPart3tmp111 * FDPart3tmp89 + FDPart3tmp117 * hDD_dD001;
        const REAL FDPart3tmp138 = FDPart3tmp130 * FDPart3tmp46 + FDPart3tmp131 * FDPart3tmp89 + FDPart1_Rational_1_2 * FDPart3tmp136 * FDPart3tmp53;
        const REAL FDPart3tmp140 = FDPart3tmp130 * FDPart3tmp59 + FDPart3tmp131 * FDPart3tmp90 + FDPart1_Rational_1_2 * FDPart3tmp136 * FDPart3tmp46;
        const REAL FDPart3tmp142 = FDPart3tmp117 * FDPart3tmp131 + FDPart3tmp136 * FDPart3tmp89 + FDPart3tmp7 * FDPart3tmp90 * hDD_dD111;
        const REAL FDPart3tmp153 = FDPart1_Rational_1_2 * FDPart3tmp147 * FDPart3tmp46 + FDPart1_Rational_1_2 * FDPart3tmp151 * FDPart3tmp53 + FDPart3tmp89 * hDD_dD002;
        const REAL FDPart3tmp155 = FDPart1_Rational_1_2 * FDPart3tmp147 * FDPart3tmp59 + FDPart1_Rational_1_2 * FDPart3tmp151 * FDPart3tmp46 + FDPart3tmp90 * hDD_dD002;
        const REAL FDPart3tmp157 = FDPart3tmp117 * hDD_dD002 + FDPart3tmp147 * FDPart3tmp90 + FDPart3tmp151 * FDPart3tmp89;
        const REAL FDPart3tmp173 =
            FDPart1_Rational_1_2 * FDPart3tmp132 * FDPart3tmp46 + FDPart1_Rational_1_2 * FDPart3tmp170 * FDPart3tmp53 + FDPart3tmp172 * FDPart3tmp89;
        const REAL FDPart3tmp175 =
            FDPart1_Rational_1_2 * FDPart3tmp132 * FDPart3tmp59 + FDPart1_Rational_1_2 * FDPart3tmp170 * FDPart3tmp46 + FDPart3tmp172 * FDPart3tmp90;
        const REAL FDPart3tmp177 = FDPart3tmp117 * FDPart3tmp172 + FDPart3tmp132 * FDPart3tmp90 + FDPart3tmp170 * FDPart3tmp89;
        const REAL FDPart3tmp190 = FDPart3tmp186 * FDPart3tmp53 + FDPart1_Rational_1_2 * FDPart3tmp187 * FDPart3tmp46 + FDPart3tmp189 * FDPart3tmp89;
        const REAL FDPart3tmp192 = FDPart3tmp186 * FDPart3tmp46 + FDPart1_Rational_1_2 * FDPart3tmp187 * FDPart3tmp59 + FDPart3tmp189 * FDPart3tmp90;
        const REAL FDPart3tmp194 = FDPart3tmp117 * FDPart3tmp189 + FDPart3tmp187 * FDPart3tmp90 + FDPart3tmp35 * FDPart3tmp89 * hDD_dD222;
        const REAL FDPart3tmp211 = FDPart3tmp201 * FDPart3tmp59 + FDPart3tmp207 * FDPart3tmp46 + FDPart3tmp28 * FDPart3tmp63;
        const REAL FDPart3tmp216 = FDPart3tmp207 * FDPart3tmp46 + FDPart3tmp214 * FDPart3tmp53 + FDPart3tmp54 * FDPart3tmp61;
        const REAL FDPart3tmp337 =
            FDPart3tmp318 * (-2. * FDPart3tmp201 * FDPart3tmp324 - 2. * FDPart3tmp214 * FDPart3tmp321 - 2. * FDPart3tmp327 -
                             FDPart3tmp328 * FDPart3tmp330 - 4. * FDPart3tmp332 * FDPart3tmp50 - 4. * FDPart3tmp335 * FDPart3tmp57);
        const REAL FDPart3tmp350 = FDPart3tmp341 + FDPart3tmp342 + FDPart3tmp344 + FDPart3tmp345 * FDPart3tmp346 +
                                   FDPart3tmp347 * FDPart3tmp45 * FDPart3tmp52 + FDPart3tmp349 * FDPart3tmp50 * FDPart3tmp52;
        const REAL FDPart3tmp360 = FDPart3tmp214 * FDPart3tmp354 + FDPart3tmp345 * FDPart3tmp57 * FDPart3tmp58 + FDPart3tmp347 * FDPart3tmp359 +
                                   FDPart3tmp349 * FDPart3tmp358 + FDPart3tmp356 + FDPart3tmp357;
        const REAL FDPart3tmp364 = FDPart3tmp201 * FDPart3tmp324 + FDPart3tmp214 * FDPart3tmp321 + FDPart3tmp327 + FDPart3tmp328 * FDPart3tmp347 +
                                   FDPart3tmp362 * FDPart3tmp50 + FDPart3tmp363 * FDPart3tmp57;
        const REAL FDPart3tmp369 = FDPart3tmp343 * FDPart3tmp50 * FDPart3tmp52;
        const REAL FDPart3tmp371 = FDPart3tmp340 * FDPart3tmp45 * FDPart3tmp57;
        const REAL FDPart3tmp374 = FDPart3tmp326 * FDPart3tmp50 * FDPart3tmp65;
        const REAL FDPart3tmp380 = FDPart3tmp340 * FDPart3tmp57 * FDPart3tmp58;
        const REAL FDPart3tmp381 = FDPart3tmp326 * FDPart3tmp57 * FDPart3tmp65;
        const REAL FDPart3tmp384 = FDPart3tmp343 * FDPart3tmp45 * FDPart3tmp52;
        const REAL FDPart3tmp418 =
            FDPart3tmp377 * (-2. * FDPart3tmp214 * FDPart3tmp354 - FDPart3tmp330 * FDPart3tmp359 - 4. * FDPart3tmp331 * FDPart3tmp358 -
                             4. * FDPart3tmp334 * FDPart3tmp57 * FDPart3tmp58 - 2. * FDPart3tmp356 - 2. * FDPart3tmp357);
        const REAL FDPart3tmp443 = FDPart3tmp367 * (-FDPart3tmp330 * FDPart3tmp45 * FDPart3tmp52 - 4. * FDPart3tmp331 * FDPart3tmp50 * FDPart3tmp52 -
                                                    4. * FDPart3tmp334 * FDPart3tmp346 - 2. * FDPart3tmp341 - 2. * FDPart3tmp342 - 2. * FDPart3tmp344);
        const REAL FDPart3tmp450 = -FDPart3tmp26 + alpha * (FDPart3tmp13 * FDPart3tmp63 + FDPart3tmp19 * FDPart3tmp61 + FDPart3tmp201 * FDPart3tmp59 +
                                                            FDPart3tmp214 * FDPart3tmp53 + FDPart3tmp222 * FDPart3tmp46 + FDPart3tmp66 * aDD00);
        // END FDPart3tmp BLOCK 9
        
        // FDPart3tmp BLOCK 10
        const REAL FDPart3tmp309 = FDPart3tmp177 * FDPart3tmp9;
        const REAL FDPart3tmp338 = FDPart3tmp194 + FDPart3tmp252;
        const REAL FDPart3tmp352 = FDPart3tmp350 * FDPart3tmp351;
        const REAL FDPart3tmp375 = -FDPart3tmp19 * FDPart3tmp321 - FDPart3tmp328 * FDPart3tmp345 - FDPart3tmp346 * FDPart3tmp347 -
                                   FDPart3tmp347 * FDPart3tmp372 - FDPart3tmp362 * FDPart3tmp52 - FDPart3tmp363 * FDPart3tmp45 - 2. * FDPart3tmp369 -
                                   2. * FDPart3tmp371 - 2. * FDPart3tmp374;
        const REAL FDPart3tmp382 = -FDPart3tmp13 * FDPart3tmp324 - FDPart3tmp328 * FDPart3tmp349 - 2 * FDPart3tmp343 * FDPart3tmp346 -
                                   FDPart3tmp347 * FDPart3tmp358 - FDPart3tmp347 * FDPart3tmp379 - FDPart3tmp362 * FDPart3tmp45 -
                                   FDPart3tmp363 * FDPart3tmp58 - 2. * FDPart3tmp380 - 2. * FDPart3tmp381;
        const REAL FDPart3tmp388 = FDPart3tmp207 * FDPart3tmp354 + FDPart3tmp326 * FDPart3tmp328 + FDPart3tmp329 * FDPart3tmp52 * FDPart3tmp58 +
                                   FDPart3tmp331 * FDPart3tmp346 + FDPart3tmp331 * FDPart3tmp372 + FDPart3tmp334 * FDPart3tmp358 +
                                   FDPart3tmp334 * FDPart3tmp379 + FDPart3tmp340 * FDPart3tmp359 + FDPart3tmp384;
        const REAL FDPart3tmp390 = FDPart3tmp321 * FDPart3tmp54 + FDPart3tmp328 * FDPart3tmp334 + FDPart3tmp329 * FDPart3tmp346 +
                                   FDPart3tmp329 * FDPart3tmp372 + FDPart3tmp332 * FDPart3tmp52 + FDPart3tmp335 * FDPart3tmp45 + FDPart3tmp369 +
                                   FDPart3tmp371 + FDPart3tmp374;
        const REAL FDPart3tmp392 = FDPart3tmp28 * FDPart3tmp324 + FDPart3tmp328 * FDPart3tmp331 + FDPart3tmp329 * FDPart3tmp358 +
                                   FDPart3tmp329 * FDPart3tmp379 + FDPart3tmp332 * FDPart3tmp45 + FDPart3tmp335 * FDPart3tmp58 +
                                   FDPart3tmp343 * FDPart3tmp346 + FDPart3tmp380 + FDPart3tmp381;
        const REAL FDPart3tmp419 = FDPart3tmp166 + FDPart3tmp192;
        const REAL FDPart3tmp421 = -FDPart3tmp222 * FDPart3tmp354 - 2. * FDPart3tmp326 * FDPart3tmp328 - 2. * FDPart3tmp340 * FDPart3tmp359 -
                                   FDPart3tmp345 * FDPart3tmp358 - FDPart3tmp345 * FDPart3tmp379 - FDPart3tmp346 * FDPart3tmp349 -
                                   FDPart3tmp347 * FDPart3tmp52 * FDPart3tmp58 - FDPart3tmp349 * FDPart3tmp372 - 2. * FDPart3tmp384;
        const REAL FDPart3tmp425 = FDPart3tmp115 - FDPart3tmp2;
        const REAL FDPart3tmp446 = FDPart3tmp173 - FDPart3tmp24;
        const REAL FDPart3tmp447 = FDPart3tmp153 - FDPart3tmp2;
        const REAL FDPart3tmp455 = FDPart3tmp350 * FDPart3tmp68;
        const REAL FDPart3tmp456 = FDPart3tmp364 * FDPart3tmp68;
        // END FDPart3tmp BLOCK 10
        
        // FDPart3tmp BLOCK 11
        const REAL FDPart3tmp97 = -FDPart3tmp81 * alpha_dD2 - FDPart3tmp85 * alpha_dD1 - FDPart3tmp91 * alpha_dD0 + alpha_dDD00;
        const REAL FDPart3tmp124 = -FDPart3tmp113 * alpha_dD2 - FDPart3tmp115 * alpha_dD1 - FDPart3tmp118 * alpha_dD0 + alpha_dDD01;
        const REAL FDPart3tmp145 = -FDPart3tmp138 * alpha_dD2 - FDPart3tmp140 * alpha_dD1 - FDPart3tmp142 * alpha_dD0 + alpha_dDD11;
        const REAL FDPart3tmp162 = -FDPart3tmp153 * alpha_dD2 - FDPart3tmp155 * alpha_dD1 - FDPart3tmp157 * alpha_dD0 + alpha_dDD02;
        const REAL FDPart3tmp181 = -FDPart3tmp173 * alpha_dD2 - FDPart3tmp175 * alpha_dD1 - FDPart3tmp177 * alpha_dD0 + alpha_dDD12;
        const REAL FDPart3tmp197 = -FDPart3tmp190 * alpha_dD2 - FDPart3tmp192 * alpha_dD1 - FDPart3tmp194 * alpha_dD0 + alpha_dDD22;
        const REAL FDPart3tmp361 = FDPart3tmp360 * (FDPart3tmp142 + f0_of_xx0);
        const REAL FDPart3tmp366 = FDPart1_Rational_3_2 * FDPart3tmp364 * alpha;
        const REAL FDPart3tmp389 = FDPart3tmp317 * FDPart3tmp388;
        const REAL FDPart3tmp391 = FDPart3tmp317 * FDPart3tmp390;
        const REAL FDPart3tmp393 = FDPart3tmp317 * FDPart3tmp392;
        const REAL FDPart3tmp400 =
            FDPart3tmp26 * (FDPart3tmp118 * FDPart3tmp395 + FDPart3tmp157 * FDPart3tmp397 + FDPart3tmp177 * FDPart3tmp396 +
                            FDPart3tmp338 * FDPart3tmp399 + FDPart3tmp394 * FDPart3tmp91 + FDPart3tmp398 * (FDPart3tmp142 + f0_of_xx0));
        const REAL FDPart3tmp426 = FDPart3tmp26 * (FDPart3tmp140 * FDPart3tmp398 + FDPart3tmp155 * FDPart3tmp397 + FDPart3tmp175 * FDPart3tmp396 +
                                                   FDPart3tmp394 * FDPart3tmp85 + FDPart3tmp395 * FDPart3tmp425 + FDPart3tmp399 * FDPart3tmp419);
        const REAL FDPart3tmp448 = FDPart3tmp26 * (FDPart3tmp113 * FDPart3tmp395 + FDPart3tmp138 * FDPart3tmp398 + FDPart3tmp190 * FDPart3tmp399 +
                                                   FDPart3tmp394 * FDPart3tmp81 + FDPart3tmp396 * FDPart3tmp446 + FDPart3tmp397 * FDPart3tmp447);
        const REAL FDPart3tmp458 = FDPart3tmp388 * FDPart3tmp457;
        const REAL FDPart3tmp459 = FDPart3tmp390 * FDPart3tmp457;
        const REAL FDPart3tmp460 = FDPart3tmp392 * FDPart3tmp457;
        // END FDPart3tmp BLOCK 11
        
        // FDPart3tmp BLOCK 12
        const REAL FDPart3tmp98 = FDPart3tmp68 * (FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * ((cf_dD0) * (cf_dD0)) - cf_dDD00) +
                                                  FDPart3tmp81 * FDPart3tmp84 + FDPart3tmp85 * FDPart3tmp87 + FDPart3tmp91 * FDPart3tmp93) -
                                  FDPart3tmp94 * alpha * ((cf_dD0) * (cf_dD0)) + 2. * FDPart3tmp95 * alpha_dD0 + FDPart3tmp97 - RbarDD00 * alpha;
        const REAL FDPart3tmp125 = FDPart3tmp120 * alpha_dD0 + FDPart3tmp123 + FDPart3tmp124 +
                                   FDPart3tmp68 * (FDPart3tmp113 * FDPart3tmp84 + FDPart3tmp115 * FDPart3tmp87 + FDPart3tmp118 * FDPart3tmp93 +
                                                   FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * cf_dD0 * cf_dD1 - cf_dDD01)) -
                                   FDPart3tmp94 * alpha * cf_dD0 * cf_dD1 - RbarDD01 * alpha;
        const REAL FDPart3tmp146 = 2. * FDPart3tmp120 * alpha_dD1 + FDPart3tmp145 +
                                   FDPart3tmp68 * (FDPart3tmp138 * FDPart3tmp84 + FDPart3tmp140 * FDPart3tmp87 + FDPart3tmp142 * FDPart3tmp93 +
                                                   FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * ((cf_dD1) * (cf_dD1)) - cf_dDD11)) -
                                   FDPart3tmp94 * alpha * ((cf_dD1) * (cf_dD1)) - RbarDD11 * alpha;
        const REAL FDPart3tmp163 = FDPart3tmp159 * alpha_dD0 + FDPart3tmp162 +
                                   FDPart3tmp68 * (FDPart3tmp153 * FDPart3tmp84 + FDPart3tmp155 * FDPart3tmp87 + FDPart3tmp157 * FDPart3tmp93 +
                                                   FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * cf_dD0 * cf_dD2 - cf_dDD02)) -
                                   FDPart3tmp94 * alpha * cf_dD0 * cf_dD2 + FDPart3tmp95 * alpha_dD2 - RbarDD02 * alpha;
        const REAL FDPart3tmp182 = FDPart3tmp120 * alpha_dD2 + FDPart3tmp179 + FDPart3tmp181 +
                                   FDPart3tmp68 * (FDPart3tmp173 * FDPart3tmp84 + FDPart3tmp175 * FDPart3tmp87 + FDPart3tmp177 * FDPart3tmp93 +
                                                   FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * cf_dD1 * cf_dD2 - cf_dDD12)) -
                                   FDPart3tmp94 * alpha * cf_dD1 * cf_dD2 - RbarDD12 * alpha;
        const REAL FDPart3tmp198 = 2. * FDPart3tmp159 * alpha_dD2 + FDPart3tmp197 +
                                   FDPart3tmp68 * (FDPart3tmp190 * FDPart3tmp84 + FDPart3tmp192 * FDPart3tmp87 + FDPart3tmp194 * FDPart3tmp93 +
                                                   FDPart1_Rational_1_2 * FDPart3tmp71 * (FDPart3tmp71 * ((cf_dD2) * (cf_dD2)) - cf_dDD22)) -
                                   FDPart3tmp94 * alpha * ((cf_dD2) * (cf_dD2)) - RbarDD22 * alpha;
        const REAL FDPart3tmp199 = -FDPart3tmp125 * FDPart3tmp127 - FDPart3tmp146 * FDPart3tmp59 - FDPart3tmp163 * FDPart3tmp164 -
                                   2. * FDPart3tmp182 * FDPart3tmp46 - FDPart3tmp198 * FDPart3tmp53 - FDPart3tmp66 * FDPart3tmp98;
        // END FDPart3tmp BLOCK 12
        // END ALL FDPart3tmp BLOCKS
        
        rhs_gfs[IDX4(ADD00GF, i0, i1, i2)] =
            FDPart3tmp1 * aDD00 + FDPart3tmp11 * FDPart3tmp13 + FDPart3tmp17 * FDPart3tmp19 - FDPart3tmp27 * aDD00 + FDPart3tmp3 * aDD_dupD001 -
            FDPart3tmp55 * (FDPart3tmp28 * FDPart3tmp46 + FDPart3tmp41 * FDPart3tmp50 * aDD00 + FDPart3tmp53 * FDPart3tmp54) +
            FDPart3tmp6 * aDD_dupD002 -
            FDPart3tmp60 * (FDPart3tmp28 * FDPart3tmp59 + FDPart3tmp41 * FDPart3tmp57 * aDD00 + FDPart3tmp46 * FDPart3tmp54) -
            FDPart3tmp69 * (FDPart3tmp28 * FDPart3tmp63 + FDPart3tmp54 * FDPart3tmp61 + FDPart3tmp66 * aDD00) +
            FDPart3tmp70 * (-FDPart3tmp199 * ((1.0 / 3.0) * hDD00 + 1.0 / 3.0) - FDPart3tmp98) + 2. * aDD00 * vetU_dD00 + aDD_dupD000 * vetU0;
        rhs_gfs[IDX4(ADD01GF, i0, i1, i2)] =
            FDPart3tmp2 *
            (FDPart3tmp1 * FDPart3tmp28 + FDPart3tmp11 * FDPart3tmp201 + FDPart3tmp14 * aDD_dupD012 + FDPart3tmp17 * FDPart3tmp207 +
             FDPart3tmp200 * aDD01 + FDPart3tmp206 * FDPart3tmp54 - FDPart3tmp208 * FDPart3tmp55 - FDPart3tmp211 * FDPart3tmp60 -
             FDPart3tmp212 * FDPart3tmp69 - FDPart3tmp27 * FDPart3tmp28 + FDPart3tmp70 * (-FDPart3tmp125 - 1.0 / 3.0 * FDPart3tmp199 * FDPart3tmp56) +
             aDD00 * vetU_dD01 + aDD01 * vetU_dD11 + aDD_dupD011 * vetU1 + vetU0 * (aDD01 + aDD_dupD010 * f0_of_xx0));
        rhs_gfs[IDX4(ADD02GF, i0, i1, i2)] =
            FDPart3tmp5 * (FDPart3tmp1 * FDPart3tmp54 + FDPart3tmp11 * FDPart3tmp207 + FDPart3tmp17 * FDPart3tmp214 + FDPart3tmp18 * FDPart3tmp200 -
                           FDPart3tmp216 * FDPart3tmp55 - FDPart3tmp217 * FDPart3tmp60 - FDPart3tmp218 * FDPart3tmp69 - FDPart3tmp27 * FDPart3tmp54 +
                           FDPart3tmp3 * (FDPart3tmp105 * aDD_dupD021 + aDD02 * f0_of_xx0 * f1_of_xx1__D1) +
                           FDPart3tmp70 * (-FDPart3tmp163 - 1.0 / 3.0 * FDPart3tmp199 * FDPart3tmp49) + aDD00 * vetU_dD02 + aDD01 * vetU_dD12 +
                           aDD02 * vetU_dD22 + aDD_dupD022 * vetU2 + vetU0 * (FDPart3tmp105 * aDD_dupD020 + FDPart3tmp18));
        rhs_gfs[IDX4(ADD11GF, i0, i1, i2)] =
            FDPart3tmp8 * (FDPart3tmp1 * FDPart3tmp201 + FDPart3tmp12 * aDD11 * vetU_dD11 + FDPart3tmp13 * vetU_dD01 +
                           FDPart3tmp14 * aDD_dupD112 * f0_of_xx0 + FDPart3tmp206 * FDPart3tmp222 - FDPart3tmp208 * FDPart3tmp225 -
                           FDPart3tmp211 * FDPart3tmp226 - FDPart3tmp212 * FDPart3tmp60 - FDPart3tmp223 * FDPart3tmp26 * aDD11 +
                           FDPart3tmp70 * (-FDPart3tmp146 - FDPart3tmp199 * ((1.0 / 3.0) * FDPart3tmp33 + (1.0 / 3.0) * FDPart3tmp7)) +
                           aDD_dupD111 * f0_of_xx0 * vetU1 + vetU0 * (FDPart3tmp12 * aDD11 + FDPart3tmp7 * aDD_dupD110));
        rhs_gfs[IDX4(ADD12GF, i0, i1, i2)] =
            FDPart3tmp233 *
            (FDPart3tmp1 * FDPart3tmp207 + FDPart3tmp105 * aDD12 * vetU_dD11 + FDPart3tmp206 * FDPart3tmp214 - FDPart3tmp216 * FDPart3tmp225 -
             FDPart3tmp217 * FDPart3tmp226 - FDPart3tmp218 * FDPart3tmp60 - FDPart3tmp223 * FDPart3tmp26 * aDD12 * f1_of_xx1 +
             FDPart3tmp28 * vetU_dD02 + FDPart3tmp3 * (FDPart3tmp43 * aDD_dupD121 + FDPart3tmp7 * aDD12 * f1_of_xx1__D1) + FDPart3tmp54 * vetU_dD01 +
             FDPart3tmp70 * (-FDPart3tmp182 - 1.0 / 3.0 * FDPart3tmp199 * FDPart3tmp7 * f1_of_xx1 * hDD12) + aDD11 * f0_of_xx0 * vetU_dD12 +
             aDD12 * f0_of_xx0 * vetU_dD22 + aDD_dupD122 * f0_of_xx0 * vetU2 + vetU0 * (FDPart3tmp43 * aDD_dupD120 + FDPart3tmp78 * aDD12));
        rhs_gfs[IDX4(ADD22GF, i0, i1, i2)] =
            FDPart3tmp237 *
            (FDPart3tmp1 * FDPart3tmp214 + FDPart3tmp105 * aDD_dupD222 * vetU2 + FDPart3tmp12 * aDD22 * f1_of_xx1 * vetU_dD22 +
             FDPart3tmp19 * vetU_dD02 - FDPart3tmp214 * FDPart3tmp216 * FDPart3tmp68 - FDPart3tmp214 * FDPart3tmp27 - FDPart3tmp217 * FDPart3tmp225 -
             FDPart3tmp218 * FDPart3tmp55 + FDPart3tmp3 * (FDPart3tmp168 * aDD22 + FDPart3tmp35 * aDD_dupD221) +
             FDPart3tmp70 * (-FDPart3tmp198 - FDPart3tmp199 * ((1.0 / 3.0) * FDPart3tmp35 + (1.0 / 3.0) * FDPart3tmp37)) +
             FDPart3tmp78 * aDD12 * vetU_dD12 + vetU0 * (FDPart3tmp149 * aDD22 + FDPart3tmp35 * aDD_dupD220));
        rhs_gfs[IDX4(ALPHAGF, i0, i1, i2)] = FDPart3tmp3 * alpha_dupD1 + FDPart3tmp6 * alpha_dupD2 - FDPart3tmp68 * trK + alpha_dupD0 * vetU0;
        rhs_gfs[IDX4(BETU0GF, i0, i1, i2)] =
            FDPart3tmp118 * FDPart3tmp293 + FDPart3tmp118 * FDPart3tmp295 - FDPart3tmp118 * FDPart3tmp296 - FDPart3tmp118 * FDPart3tmp298 +
            FDPart3tmp118 * FDPart3tmp393 + FDPart3tmp142 * FDPart3tmp299 - FDPart3tmp142 * FDPart3tmp300 + FDPart3tmp157 * FDPart3tmp301 +
            FDPart3tmp157 * FDPart3tmp303 - FDPart3tmp157 * FDPart3tmp304 - FDPart3tmp157 * FDPart3tmp306 + FDPart3tmp157 * FDPart3tmp391 +
            FDPart3tmp177 * FDPart3tmp308 - FDPart3tmp177 * FDPart3tmp312 + FDPart3tmp177 * FDPart3tmp389 + FDPart3tmp194 * FDPart3tmp315 -
            FDPart3tmp194 * FDPart3tmp316 - 3.0 / 4.0 * FDPart3tmp239 * lambdaU1 - 3.0 / 4.0 * FDPart3tmp241 * FDPart3tmp242 -
            3.0 / 4.0 * FDPart3tmp242 * FDPart3tmp309 - FDPart3tmp244 * FDPart3tmp61 - FDPart3tmp245 * FDPart3tmp63 - FDPart3tmp247 * FDPart3tmp66 +
            (3.0 / 4.0) * FDPart3tmp250 + (3.0 / 4.0) * FDPart3tmp253 + FDPart3tmp259 * FDPart3tmp61 + FDPart3tmp261 * FDPart3tmp262 +
            FDPart3tmp264 * FDPart3tmp265 + FDPart1_Rational_3_2 * FDPart3tmp266 * FDPart3tmp46 + FDPart3tmp283 * FDPart3tmp63 + FDPart3tmp289 * FDPart3tmp66 +
            FDPart3tmp290 * FDPart3tmp91 - FDPart3tmp292 * FDPart3tmp91 + FDPart3tmp3 * betU_dupD01 + FDPart3tmp309 * FDPart3tmp4 * betU2 +
            (3.0 / 4.0) * FDPart3tmp337 + FDPart3tmp338 * FDPart3tmp352 + FDPart3tmp351 * FDPart3tmp361 + FDPart3tmp366 * FDPart3tmp91 +
            (3.0 / 4.0) * FDPart3tmp367 * FDPart3tmp375 + (3.0 / 4.0) * FDPart3tmp377 * FDPart3tmp382 + (3.0 / 4.0) * FDPart3tmp400 +
            FDPart3tmp6 * betU_dupD02 + (3.0 / 4.0) * FDPart3tmp66 * vetU_dDD000 - betU0 * eta + betU_dupD00 * vetU0 -
            3.0 / 4.0 * lambdaU0 * vetU_dD00;
        rhs_gfs[IDX4(BETU1GF, i0, i1, i2)] =
            f0_of_xx0 *
            (-3.0 / 4.0 * FDPart3tmp11 * lambdaU0 + FDPart3tmp115 * FDPart3tmp293 + FDPart3tmp115 * FDPart3tmp295 - FDPart3tmp115 * FDPart3tmp296 -
             FDPart3tmp115 * FDPart3tmp298 + FDPart3tmp140 * FDPart3tmp299 - FDPart3tmp140 * FDPart3tmp300 +
             FDPart3tmp140 * FDPart3tmp351 * FDPart3tmp360 + FDPart3tmp15 * betU_dupD12 + FDPart3tmp155 * FDPart3tmp301 +
             FDPart3tmp155 * FDPart3tmp303 - FDPart3tmp155 * FDPart3tmp304 - FDPart3tmp155 * FDPart3tmp306 + FDPart3tmp155 * FDPart3tmp391 +
             FDPart3tmp175 * FDPart3tmp308 - FDPart3tmp175 * FDPart3tmp312 + FDPart3tmp175 * FDPart3tmp389 + FDPart3tmp175 * FDPart3tmp416 -
             FDPart3tmp175 * FDPart3tmp417 + FDPart3tmp192 * FDPart3tmp315 - FDPart3tmp192 * FDPart3tmp316 - FDPart3tmp244 * FDPart3tmp46 -
             FDPart3tmp245 * FDPart3tmp59 - FDPart3tmp247 * FDPart3tmp63 + FDPart3tmp259 * FDPart3tmp46 +
             FDPart3tmp262 * (FDPart3tmp286 + FDPart3tmp406) + FDPart3tmp265 * FDPart3tmp408 + FDPart3tmp283 * FDPart3tmp59 +
             FDPart3tmp289 * FDPart3tmp63 + FDPart3tmp290 * FDPart3tmp85 - FDPart3tmp292 * FDPart3tmp85 - FDPart3tmp294 * eta +
             (3.0 / 4.0) * FDPart3tmp318 * FDPart3tmp382 + FDPart3tmp352 * FDPart3tmp419 + FDPart3tmp366 * FDPart3tmp85 +
             (3.0 / 4.0) * FDPart3tmp367 * FDPart3tmp421 + FDPart3tmp393 * FDPart3tmp425 - 3.0 / 4.0 * FDPart3tmp401 - 3.0 / 4.0 * FDPart3tmp403 +
             (3.0 / 4.0) * FDPart3tmp409 + (3.0 / 4.0) * FDPart3tmp411 + (3.0 / 4.0) * FDPart3tmp413 + (3.0 / 4.0) * FDPart3tmp414 +
             (3.0 / 4.0) * FDPart3tmp415 + (3.0 / 4.0) * FDPart3tmp418 + (3.0 / 4.0) * FDPart3tmp426 + FDPart3tmp9 * betU_dupD11 +
             vetU0 * (FDPart3tmp2 * betU_dupD10 - FDPart3tmp8 * betU1));
        rhs_gfs[IDX4(BETU2GF, i0, i1, i2)] =
            FDPart3tmp105 *
            (FDPart3tmp113 * FDPart3tmp293 + FDPart3tmp113 * FDPart3tmp295 - FDPart3tmp113 * FDPart3tmp296 - FDPart3tmp113 * FDPart3tmp298 +
             FDPart3tmp113 * FDPart3tmp393 + FDPart3tmp138 * FDPart3tmp299 - FDPart3tmp138 * FDPart3tmp300 +
             FDPart3tmp138 * FDPart3tmp351 * FDPart3tmp360 + FDPart3tmp153 * FDPart3tmp301 + FDPart3tmp153 * FDPart3tmp303 -
             FDPart3tmp153 * FDPart3tmp304 - FDPart3tmp153 * FDPart3tmp306 - 3.0 / 4.0 * FDPart3tmp17 * lambdaU0 + FDPart3tmp173 * FDPart3tmp308 -
             FDPart3tmp173 * FDPart3tmp312 + FDPart3tmp173 * FDPart3tmp416 - FDPart3tmp173 * FDPart3tmp417 + FDPart3tmp190 * FDPart3tmp315 -
             FDPart3tmp190 * FDPart3tmp316 + FDPart3tmp190 * FDPart3tmp352 - FDPart3tmp244 * FDPart3tmp53 - FDPart3tmp245 * FDPart3tmp46 -
             FDPart3tmp247 * FDPart3tmp61 + FDPart3tmp259 * FDPart3tmp53 + FDPart3tmp262 * FDPart3tmp438 + FDPart3tmp265 * FDPart3tmp432 +
             FDPart3tmp283 * FDPart3tmp46 + FDPart3tmp289 * FDPart3tmp61 + FDPart3tmp290 * FDPart3tmp81 - FDPart3tmp292 * FDPart3tmp81 +
             FDPart3tmp3 * (FDPart3tmp2 * FDPart3tmp4 * betU_dupD21 - FDPart3tmp429 * betU2) - FDPart3tmp302 * eta + FDPart3tmp314 * betU_dupD22 +
             (3.0 / 4.0) * FDPart3tmp318 * FDPart3tmp375 + FDPart3tmp366 * FDPart3tmp81 + (3.0 / 4.0) * FDPart3tmp377 * FDPart3tmp421 +
             FDPart3tmp389 * FDPart3tmp446 + FDPart3tmp391 * FDPart3tmp447 - 3.0 / 4.0 * FDPart3tmp427 - 3.0 / 4.0 * FDPart3tmp430 +
             (3.0 / 4.0) * FDPart3tmp433 + (3.0 / 4.0) * FDPart3tmp435 + (3.0 / 4.0) * FDPart3tmp437 + (3.0 / 4.0) * FDPart3tmp439 +
             (3.0 / 4.0) * FDPart3tmp442 + (3.0 / 4.0) * FDPart3tmp443 + (3.0 / 4.0) * FDPart3tmp448 +
             vetU0 * (FDPart3tmp2 * FDPart3tmp4 * betU_dupD20 - FDPart3tmp233 * betU2));
        rhs_gfs[IDX4(CFGF, i0, i1, i2)] =
            -2. * cf *
            (-1.0 / 6.0 * FDPart3tmp1 + (1.0 / 6.0) * FDPart3tmp20 + (1.0 / 6.0) * FDPart3tmp24 * FDPart3tmp3 + (1.0 / 3.0) * FDPart3tmp274 -
             FDPart3tmp3 * FDPart3tmp83 * cf_dupD1 + (1.0 / 6.0) * FDPart3tmp5 * vetU_dD22 - FDPart3tmp6 * FDPart3tmp83 * cf_dupD2 -
             FDPart3tmp83 * cf_dupD0 * vetU0 + (1.0 / 6.0) * vetU_dD00);
        rhs_gfs[IDX4(HDD00GF, i0, i1, i2)] = FDPart3tmp12 * FDPart3tmp17 * FDPart3tmp42 + FDPart3tmp260 * FDPart3tmp75 + FDPart3tmp3 * hDD_dupD001 +
                                             2. * FDPart3tmp31 * vetU_dD00 + FDPart3tmp450 * (FDPart1_Rational_2_3 * hDD00 + 2.0 / 3.0) +
                                             FDPart3tmp6 * hDD_dupD002 - FDPart3tmp69 + hDD_dupD000 * vetU0;
        rhs_gfs[IDX4(HDD01GF, i0, i1, i2)] =
            FDPart3tmp2 * (FDPart3tmp11 * FDPart3tmp34 + FDPart3tmp14 * hDD_dupD012 + FDPart3tmp17 * FDPart3tmp44 + FDPart3tmp200 * hDD01 +
                           FDPart3tmp206 * FDPart3tmp49 + FDPart3tmp31 * vetU_dD01 + FDPart1_Rational_2_3 * FDPart3tmp450 * FDPart3tmp56 - FDPart3tmp60 +
                           hDD01 * vetU_dD11 + hDD_dupD011 * vetU1 + vetU0 * (f0_of_xx0 * hDD_dupD010 + hDD01));
        rhs_gfs[IDX4(HDD02GF, i0, i1, i2)] =
            FDPart3tmp5 *
            (FDPart3tmp11 * FDPart3tmp44 + FDPart3tmp17 * FDPart3tmp38 + FDPart3tmp200 * FDPart3tmp42 +
             FDPart3tmp3 * (FDPart3tmp104 + FDPart3tmp105 * hDD_dupD021) + FDPart3tmp31 * vetU_dD02 + FDPart1_Rational_2_3 * FDPart3tmp450 * FDPart3tmp49 -
             FDPart3tmp55 + hDD01 * vetU_dD12 + hDD02 * vetU_dD22 + hDD_dupD022 * vetU2 + vetU0 * (FDPart3tmp105 * hDD_dupD020 + FDPart3tmp42));
        rhs_gfs[IDX4(HDD11GF, i0, i1, i2)] =
            FDPart3tmp8 * (FDPart3tmp14 * f0_of_xx0 * hDD_dupD112 + 2. * FDPart3tmp20 * FDPart3tmp34 + 2. * FDPart3tmp206 * FDPart3tmp44 -
                           FDPart3tmp226 + FDPart3tmp450 * (FDPart3tmp223 + FDPart1_Rational_2_3 * FDPart3tmp33) + FDPart3tmp75 * f0_of_xx0 * vetU_dD01 +
                           f0_of_xx0 * hDD_dupD111 * vetU1 + vetU0 * (FDPart3tmp7 * hDD_dupD110 + FDPart3tmp99));
        rhs_gfs[IDX4(HDD12GF, i0, i1, i2)] =
            FDPart3tmp233 * (FDPart3tmp105 * hDD12 * vetU_dD11 + FDPart3tmp2 * FDPart3tmp34 * vetU_dD12 + FDPart3tmp206 * FDPart3tmp38 +
                             FDPart3tmp223 * FDPart3tmp450 * f1_of_xx1 * hDD12 - FDPart3tmp225 +
                             FDPart3tmp3 * (FDPart3tmp135 + FDPart3tmp43 * hDD_dupD121) + FDPart3tmp49 * vetU_dD01 + FDPart3tmp56 * vetU_dD02 +
                             f0_of_xx0 * hDD12 * vetU_dD22 + f0_of_xx0 * hDD_dupD122 * vetU2 + vetU0 * (FDPart3tmp108 + FDPart3tmp43 * hDD_dupD120));
        rhs_gfs[IDX4(HDD22GF, i0, i1, i2)] =
            FDPart3tmp237 *
            (FDPart3tmp105 * hDD_dupD222 * vetU2 + FDPart3tmp108 * vetU_dD12 + FDPart3tmp12 * FDPart3tmp42 * vetU_dD02 +
             FDPart3tmp21 * FDPart3tmp38 * FDPart3tmp4 * vetU_dD22 - FDPart3tmp214 * FDPart3tmp68 +
             FDPart3tmp3 * (FDPart3tmp169 + FDPart3tmp35 * hDD_dupD221) + FDPart3tmp450 * (FDPart1_Rational_2_3 * FDPart3tmp35 + FDPart1_Rational_2_3 * FDPart3tmp37) +
             vetU0 * (FDPart3tmp150 + FDPart3tmp35 * hDD_dupD220));
        rhs_gfs[IDX4(LAMBDAU0GF, i0, i1, i2)] =
            FDPart3tmp118 * FDPart3tmp460 + FDPart3tmp127 * FDPart3tmp261 + FDPart3tmp157 * FDPart3tmp459 + FDPart3tmp164 * FDPart3tmp264 +
            FDPart3tmp177 * FDPart3tmp458 - FDPart3tmp239 * lambdaU1 - FDPart3tmp241 * FDPart3tmp242 - FDPart3tmp244 * FDPart3tmp397 -
            FDPart3tmp245 * FDPart3tmp395 - 4.0 / 3.0 * FDPart3tmp247 * FDPart3tmp66 + FDPart3tmp250 + FDPart3tmp253 +
            2 * FDPart3tmp266 * FDPart3tmp46 + FDPart3tmp3 * lambdaU_dupD01 + FDPart3tmp337 + FDPart3tmp338 * FDPart3tmp455 +
            FDPart3tmp361 * FDPart3tmp68 + FDPart3tmp367 * FDPart3tmp375 + FDPart3tmp377 * FDPart3tmp382 + FDPart3tmp400 +
            FDPart3tmp452 * FDPart3tmp61 + FDPart3tmp453 * FDPart3tmp63 + FDPart3tmp454 * FDPart3tmp66 + FDPart3tmp456 * FDPart3tmp91 +
            FDPart3tmp6 * lambdaU_dupD02 + FDPart3tmp66 * vetU_dDD000 - lambdaU0 * vetU_dD00 + lambdaU_dupD00 * vetU0;
        rhs_gfs[IDX4(LAMBDAU1GF, i0, i1, i2)] =
            f0_of_xx0 *
            (-FDPart3tmp11 * lambdaU0 + FDPart3tmp127 * (FDPart3tmp286 + FDPart3tmp406) + FDPart3tmp140 * FDPart3tmp360 * FDPart3tmp68 +
             FDPart3tmp15 * lambdaU_dupD12 + FDPart3tmp155 * FDPart3tmp459 + FDPart3tmp164 * FDPart3tmp408 + FDPart3tmp175 * FDPart3tmp458 -
             FDPart3tmp244 * FDPart3tmp396 - 4.0 / 3.0 * FDPart3tmp245 * FDPart3tmp59 - FDPart3tmp247 * FDPart3tmp395 +
             FDPart3tmp318 * FDPart3tmp382 + FDPart3tmp367 * FDPart3tmp421 - FDPart3tmp401 - FDPart3tmp403 + FDPart3tmp409 + FDPart3tmp411 +
             FDPart3tmp413 + FDPart3tmp414 + FDPart3tmp415 + FDPart3tmp418 + FDPart3tmp419 * FDPart3tmp455 + FDPart3tmp425 * FDPart3tmp460 +
             FDPart3tmp426 + FDPart3tmp452 * FDPart3tmp46 + FDPart3tmp453 * FDPart3tmp59 + FDPart3tmp454 * FDPart3tmp63 +
             FDPart3tmp456 * FDPart3tmp85 + FDPart3tmp9 * lambdaU_dupD11 + vetU0 * (FDPart3tmp2 * lambdaU_dupD10 - FDPart3tmp8 * lambdaU1));
        rhs_gfs[IDX4(LAMBDAU2GF, i0, i1, i2)] =
            FDPart3tmp105 *
            (FDPart3tmp113 * FDPart3tmp460 + FDPart3tmp127 * FDPart3tmp438 + FDPart3tmp138 * FDPart3tmp360 * FDPart3tmp68 +
             FDPart3tmp164 * FDPart3tmp432 - FDPart3tmp17 * lambdaU0 + FDPart3tmp190 * FDPart3tmp455 - 4.0 / 3.0 * FDPart3tmp244 * FDPart3tmp53 -
             FDPart3tmp245 * FDPart3tmp396 - FDPart3tmp247 * FDPart3tmp397 +
             FDPart3tmp3 * (-FDPart3tmp429 * lambdaU2 + FDPart3tmp5 * lambdaU_dupD21) + FDPart3tmp314 * lambdaU_dupD22 +
             FDPart3tmp318 * FDPart3tmp375 + FDPart3tmp377 * FDPart3tmp421 - FDPart3tmp427 - FDPart3tmp430 + FDPart3tmp433 + FDPart3tmp435 +
             FDPart3tmp437 + FDPart3tmp439 + FDPart3tmp442 + FDPart3tmp443 + FDPart3tmp446 * FDPart3tmp458 + FDPart3tmp447 * FDPart3tmp459 +
             FDPart3tmp448 + FDPart3tmp452 * FDPart3tmp53 + FDPart3tmp453 * FDPart3tmp46 + FDPart3tmp454 * FDPart3tmp61 +
             FDPart3tmp456 * FDPart3tmp81 + vetU0 * (-FDPart3tmp233 * lambdaU2 + FDPart3tmp5 * lambdaU_dupD20));
        rhs_gfs[IDX4(TRKGF, i0, i1, i2)] =
            FDPart3tmp201 * FDPart3tmp360 * alpha + FDPart3tmp214 * FDPart3tmp350 * alpha + FDPart3tmp225 * FDPart3tmp388 + FDPart3tmp3 * trK_dupD1 +
            FDPart3tmp364 * aDD00 * alpha + FDPart3tmp390 * FDPart3tmp55 + FDPart3tmp392 * FDPart3tmp60 -
            FDPart3tmp46 * FDPart3tmp70 * (-FDPart3tmp179 + FDPart3tmp181) -
            FDPart3tmp46 * FDPart3tmp70 * (-FDPart3tmp120 * alpha_dD2 + FDPart3tmp181) -
            FDPart3tmp53 * FDPart3tmp70 * (-FDPart3tmp159 * alpha_dD2 + FDPart3tmp197) -
            FDPart3tmp59 * FDPart3tmp70 * (-FDPart3tmp120 * alpha_dD1 + FDPart3tmp145) + FDPart3tmp6 * trK_dupD2 -
            FDPart3tmp61 * FDPart3tmp70 * (FDPart3tmp162 - FDPart3tmp95 * alpha_dD2) -
            FDPart3tmp61 * FDPart3tmp70 * (-FDPart3tmp159 * alpha_dD0 + FDPart3tmp162) -
            FDPart3tmp63 * FDPart3tmp70 * (-FDPart3tmp123 + FDPart3tmp124) -
            FDPart3tmp63 * FDPart3tmp70 * (-FDPart3tmp120 * alpha_dD0 + FDPart3tmp124) -
            FDPart3tmp66 * FDPart3tmp70 * (-FDPart3tmp95 * alpha_dD0 + FDPart3tmp97) + (1.0 / 3.0) * alpha * ((trK) * (trK)) + trK_dupD0 * vetU0;
        rhs_gfs[IDX4(VETU0GF, i0, i1, i2)] = FDPart3tmp118 * FDPart3tmp466 + FDPart3tmp142 * FDPart3tmp467 + FDPart3tmp157 * FDPart3tmp468 +
                                             FDPart3tmp194 * FDPart3tmp469 + FDPart3tmp3 * vetU_dupD01 + FDPart3tmp309 * FDPart3tmp436 +
                                             FDPart3tmp465 * FDPart3tmp91 + FDPart3tmp6 * vetU_dupD02 + betU0 + vetU0 * vetU_dupD00;
        rhs_gfs[IDX4(VETU1GF, i0, i1, i2)] =
            f0_of_xx0 * (FDPart3tmp115 * FDPart3tmp466 + FDPart3tmp140 * FDPart3tmp467 + FDPart3tmp15 * vetU_dupD12 + FDPart3tmp155 * FDPart3tmp468 +
                         FDPart3tmp175 * FDPart3tmp436 * FDPart3tmp9 + FDPart3tmp192 * FDPart3tmp469 + FDPart3tmp294 + FDPart3tmp465 * FDPart3tmp85 +
                         FDPart3tmp9 * vetU_dupD11 + vetU0 * (FDPart3tmp2 * vetU_dupD10 - FDPart3tmp9));
        rhs_gfs[IDX4(VETU2GF, i0, i1, i2)] =
            FDPart3tmp105 *
            (FDPart3tmp113 * FDPart3tmp466 + FDPart3tmp138 * FDPart3tmp467 + FDPart3tmp153 * FDPart3tmp468 +
             FDPart3tmp173 * FDPart3tmp436 * FDPart3tmp9 + FDPart3tmp190 * FDPart3tmp469 + FDPart3tmp3 * (FDPart3tmp205 + FDPart3tmp5 * vetU_dupD21) +
             FDPart3tmp302 + FDPart3tmp314 * vetU_dupD22 + FDPart3tmp465 * FDPart3tmp81 + vetU0 * (-FDPart3tmp15 + FDPart3tmp5 * vetU_dupD20));
        // if(IDX4(VETU2GF, i0, i1, i2) == IDX4(VETU2GF, 17, 6 , 6)) {
        //     // printf("VETU2GF: %f, %f, %f, %f, %f, %f, %f\n",
        //     // FDPart3tmp105, FDPart3tmp113, FDPart3tmp466, FDPart3tmp138, FDPart3tmp467, FDPart3tmp153, FDPart3tmp468);
        //     printf("VETU2GF: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f,\n",
        //     FDPart3tmp173, FDPart3tmp436, FDPart3tmp9, FDPart3tmp190, FDPart3tmp469, FDPart3tmp3, FDPart3tmp205, FDPart3tmp5, vetU_dupD21);
        // }
            // int const iref0 = 3;
            // int const iref1 = (int)(NGHOSTS + FDPart1_Rational_1_4 * Nxx2 - 1.0 / 2.0);
            // int const iref2 = (int)(NGHOSTS + (3.0 / 4.0) * Nxx2 - 1.0 / 2.0);
            // if(IDX4(ADD22GF, i0, i1, i2) == IDX4(ADD22GF, iref0, iref1 , iref2)) {
            //     printf("ADD22GF:\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e \n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e\n ", 
            //     FDPart3tmp237, FDPart3tmp1, FDPart3tmp214, FDPart3tmp105, aDD_dupD222,
            //     vetU2, FDPart3tmp12, aDD22, f1_of_xx1, vetU_dD22, 
            //     FDPart3tmp19, vetU_dD02, FDPart3tmp216, FDPart3tmp68, FDPart3tmp27, 
            //     FDPart3tmp217, FDPart3tmp225, FDPart3tmp218, FDPart3tmp55, FDPart3tmp3, 
            //     FDPart3tmp168, FDPart3tmp35, aDD_dupD221, FDPart3tmp70, FDPart3tmp198, 
            //     FDPart3tmp199, FDPart3tmp37, FDPart3tmp78, aDD12, vetU_dD12, 
            //     vetU0, FDPart3tmp149, aDD_dupD220);
            //     printf("FDPart3tmp199: \n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n",
            //     FDPart3tmp125, FDPart3tmp127, FDPart3tmp146, FDPart3tmp59, FDPart3tmp163,
            //     FDPart3tmp164, FDPart3tmp182, FDPart3tmp46, FDPart3tmp53, FDPart3tmp66, FDPart3tmp98);

            //     printf("FDPart3tmp98: \n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n"
            //     " %1.13e, %1.13e, %1.13e, %1.13e, %1.13e, %1.13e\n",
            //     FDPart3tmp68, FDPart3tmp71, cf_dD0, cf_dDD00,FDPart3tmp81, 
            //     FDPart3tmp84, FDPart3tmp85, FDPart3tmp87, FDPart3tmp91, FDPart3tmp93,
            //     FDPart3tmp94, alpha, FDPart3tmp95, alpha_dD0, FDPart3tmp97, RbarDD00);
            // }
      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    }   // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  }     // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}
void rhs_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                              const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                              REAL *restrict rhs_gfs) {
#include "../set_CodeParameters.h"
  int threads_in_x_dir = MIN(1024, params->Nxx0 / 32);
  int threads_in_y_dir = MIN(1024 / threads_in_x_dir, params->Nxx1);
  int threads_in_z_dir = 1;

    // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
//   dim3 grid_blocks(params->Nxx1 / threads_in_y_dir, params->Nxx2, 1);
  dim3 grid_blocks(
    (Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
    (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
    (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir
  );
    rhs_eval__rfm__Spherical_gpu<<<grid_blocks, block_threads>>>(eta, rfmstruct->f0_of_xx0, rfmstruct->f1_of_xx1, 
    // rhs_eval__rfm__Spherical_gpu<<<1,1>>>(eta, rfmstruct->f0_of_xx0, rfmstruct->f1_of_xx1, 
    rfmstruct->f1_of_xx1__D1, rfmstruct->f1_of_xx1__DD11, auxevol_gfs, in_gfs, rhs_gfs);
      // print_params<<<1,1>>>();
    cudaDeviceSynchronize();
    for(int i = 0; i < NUM_EVOL_GFS; ++i)
        print_var<<<1,1>>>(rhs_gfs, IDX4(i, 34, 18 , 18));
    printf("**************************\n");
}