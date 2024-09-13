#include "BHaH_defines.h"
/*
 * GPU Kernel: rhs_eval_gpu.
 * GPU Kernel to evaluate RHS on the interior.
 */

// Baseline
// __global__ static void rhs_eval_gpu(const REAL *restrict rfm_f0_of_xx0, const REAL *restrict rfm_f0_of_xx0__D0,
//                                     const REAL *restrict rfm_f0_of_xx0__DD00, const REAL *restrict rfm_f0_of_xx0__DDD000,
//                                     const REAL *restrict rfm_f1_of_xx1, const REAL *restrict rfm_f1_of_xx1__D1,
//                                     const REAL *restrict rfm_f1_of_xx1__DD11, const REAL *restrict rfm_f2_of_xx0,
//                                     const REAL *restrict rfm_f2_of_xx0__D0, const REAL *restrict rfm_f2_of_xx0__DD00,
//                                     const REAL *restrict rfm_f4_of_xx1, const REAL *restrict rfm_f4_of_xx1__D1,
//                                     const REAL *restrict rfm_f4_of_xx1__DD11, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
//                                     REAL *restrict rhs_gfs, const REAL eta_damping) {

//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

//   [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
//   [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
//   [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

//   const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
//   const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
//   const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

//   const int stride0 = blockDim.x * gridDim.x;
//   const int stride1 = blockDim.y * gridDim.y;
//   const int stride2 = blockDim.z * gridDim.z;

//   for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
//     for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
//       [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1];
//       [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1];

//       for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
//         [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
//         [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
//          * Read gridfunction(s) from main memory and compute FD stencils as needed.
//          */
//         const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
//         const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
//         const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1, i2 - 5)];
//         const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1, i2 - 4)];
//         const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1, i2 - 3)];
//         const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
//         const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
//         const REAL uu_i1m5 = in_gfs[IDX4(UUGF, i0, i1 - 5, i2)];
//         const REAL uu_i1m4 = in_gfs[IDX4(UUGF, i0, i1 - 4, i2)];
//         const REAL uu_i1m3 = in_gfs[IDX4(UUGF, i0, i1 - 3, i2)];
//         const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
//         const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
//         const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1, i2)];
//         const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1, i2)];
//         const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1, i2)];
//         const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
//         const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
//         const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
//         const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
//         const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
//         const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1, i2)];
//         const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1, i2)];
//         const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1, i2)];
//         const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
//         const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
//         const REAL uu_i1p3 = in_gfs[IDX4(UUGF, i0, i1 + 3, i2)];
//         const REAL uu_i1p4 = in_gfs[IDX4(UUGF, i0, i1 + 4, i2)];
//         const REAL uu_i1p5 = in_gfs[IDX4(UUGF, i0, i1 + 5, i2)];
//         const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
//         const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
//         const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1, i2 + 3)];
//         const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1, i2 + 4)];
//         const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1, i2 + 5)];
//         const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)];
//         const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
//         static constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
//         static constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
//         static constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
//         static constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
//         static constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
//         static constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
//         static constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
//         static constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
//         static constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
//         static constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;
//         const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
//         const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
//         const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
//         const REAL uu_dDD00 =
//             ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
//                                        FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
//         const REAL uu_dDD11 =
//             ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
//                                        FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
//         const REAL uu_dDD22 =
//             ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
//                                        FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
//                                        FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
//          * Evaluate SymPy expressions and write to main memory.
//          */
//         const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
//         const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
//         const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
//         const REAL FDPart3tmp7 = 2 / FDPart3tmp4;
//         const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
//         const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
//         rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
//             ((variable_wavespeed) * (variable_wavespeed)) *
//             ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
//              FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
//              uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
//                        (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
//                            (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
//                             2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
//                             FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
//                            ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
//              uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));

//       } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
//     } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
//   } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
// }

// Revision 1
// __global__ static void rhs_eval_gpu(const REAL *restrict rfm_f0_of_xx0, const REAL *restrict rfm_f0_of_xx0__D0,
//                                     const REAL *restrict rfm_f0_of_xx0__DD00, const REAL *restrict rfm_f0_of_xx0__DDD000,
//                                     const REAL *restrict rfm_f1_of_xx1, const REAL *restrict rfm_f1_of_xx1__D1,
//                                     const REAL *restrict rfm_f1_of_xx1__DD11, const REAL *restrict rfm_f2_of_xx0,
//                                     const REAL *restrict rfm_f2_of_xx0__D0, const REAL *restrict rfm_f2_of_xx0__DD00,
//                                     const REAL *restrict rfm_f4_of_xx1, const REAL *restrict rfm_f4_of_xx1__D1,
//                                     const REAL *restrict rfm_f4_of_xx1__DD11, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
//                                     REAL *restrict rhs_gfs, const REAL eta_damping) {

//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

//   [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
//   [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
//   [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

//   const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
//   const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
//   const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

//   // const int stride0 = blockDim.x * gridDim.x;
//   // const int stride1 = blockDim.y * gridDim.y;
//   // const int stride2 = blockDim.z * gridDim.z;

//   int i2 = tid2 + NGHOSTS;
//   int i1 = tid1 + NGHOSTS;
//   int i0 = tid0 + NGHOSTS;

//   if( (i2 >= Nxx_plus_2NGHOSTS2 - NGHOSTS)
//     ||(i1 >= Nxx_plus_2NGHOSTS1 - NGHOSTS)
//     ||(i0 >= Nxx_plus_2NGHOSTS0 - NGHOSTS) ) {

//     return;
//   }

//   // for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
//   //   for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
//       [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1];
//       [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1];

//       // for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
//         [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
//         [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
//          * Read gridfunction(s) from main memory and compute FD stencils as needed.
//          */
//         const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
//         const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
//         const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1, i2 - 5)];
//         const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1, i2 - 4)];
//         const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1, i2 - 3)];
//         const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
//         const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
//         const REAL uu_i1m5 = in_gfs[IDX4(UUGF, i0, i1 - 5, i2)];
//         const REAL uu_i1m4 = in_gfs[IDX4(UUGF, i0, i1 - 4, i2)];
//         const REAL uu_i1m3 = in_gfs[IDX4(UUGF, i0, i1 - 3, i2)];
//         const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
//         const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
//         const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1, i2)];
//         const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1, i2)];
//         const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1, i2)];
//         const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
//         const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
//         const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
//         const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
//         const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
//         const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1, i2)];
//         const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1, i2)];
//         const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1, i2)];
//         const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
//         const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
//         const REAL uu_i1p3 = in_gfs[IDX4(UUGF, i0, i1 + 3, i2)];
//         const REAL uu_i1p4 = in_gfs[IDX4(UUGF, i0, i1 + 4, i2)];
//         const REAL uu_i1p5 = in_gfs[IDX4(UUGF, i0, i1 + 5, i2)];
//         const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
//         const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
//         const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1, i2 + 3)];
//         const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1, i2 + 4)];
//         const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1, i2 + 5)];
//         const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)];
//         const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
//         static constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
//         static constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
//         static constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
//         static constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
//         static constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
//         static constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
//         static constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
//         static constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
//         static constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
//         static constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;
//         const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
//         const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
//         const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
//         const REAL uu_dDD00 =
//             ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
//                                        FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
//         const REAL uu_dDD11 =
//             ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
//                                        FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
//         const REAL uu_dDD22 =
//             ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
//                                        FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
//                                        FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
//          * Evaluate SymPy expressions and write to main memory.
//          */
//         const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
//         const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
//         const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
//         const REAL FDPart3tmp7 = 2 / FDPart3tmp4;
//         const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
//         const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
//         rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
//             ((variable_wavespeed) * (variable_wavespeed)) *
//             ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
//              FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
//              uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
//                        (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
//                            (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
//                             2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
//                             FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
//                            ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
//              uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));

//   //     } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
//   //   } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
//   // } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
// }

// Revision 2
#define TILE_M 8
__global__ static void rhs_eval_gpu(const REAL *restrict rfm_f0_of_xx0, const REAL *restrict rfm_f0_of_xx0__D0,
                                    const REAL *restrict rfm_f0_of_xx0__DD00, const REAL *restrict rfm_f0_of_xx0__DDD000,
                                    const REAL *restrict rfm_f1_of_xx1, const REAL *restrict rfm_f1_of_xx1__D1,
                                    const REAL *restrict rfm_f1_of_xx1__DD11, const REAL *restrict rfm_f2_of_xx0,
                                    const REAL *restrict rfm_f2_of_xx0__D0, const REAL *restrict rfm_f2_of_xx0__DD00,
                                    const REAL *restrict rfm_f4_of_xx1, const REAL *restrict rfm_f4_of_xx1__D1,
                                    const REAL *restrict rfm_f4_of_xx1__DD11, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                                    REAL *restrict rhs_gfs, const REAL eta_damping) {

  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
  [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
  [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

  const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

  int i2 = tid2 + NGHOSTS;
  int i1 = TILE_M * (tid1 + NGHOSTS);
  int i0 = tid0 + NGHOSTS;

  if( (i2 >= Nxx_plus_2NGHOSTS2 - NGHOSTS)
    ||(i1 >= Nxx_plus_2NGHOSTS1 - NGHOSTS)
    ||(i0 >= Nxx_plus_2NGHOSTS0 - NGHOSTS) ) {

    return;
  }
  // Move these out of the way
  static constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
  static constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
  static constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
  static constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
  static constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
  static constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
  static constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
  static constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
  static constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
  static constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;

  REAL uu_dDD22_part[TILE_M] = {0};
  // Function of xx2
  for(int m = 0; m < TILE_M; ++m) {
    if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }

    const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 - 5)];
    const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 - 4)];
    const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 - 3)];
    const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 - 2)];
    const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 - 1)];

    const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 + 1)];
    const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 + 2)];
    const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 + 3)];
    const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 + 4)];
    const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1 + m, i2 + 5)];

    uu_dDD22_part[m] = FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
                                        FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
                                        FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1);
  }

  // Function of xx1
  REAL uu_dDD11_part[TILE_M] = {0};
  REAL uu_dD1[TILE_M] = {0};
  {

    REAL uu_GF_ary[2U * NGHOSTS + 1] = {0};
    for(int i = 0; i < 2U * NGHOSTS + 1; ++i) {
      uu_GF_ary[i] = in_gfs[IDX4(UUGF, i0, i1 - NGHOSTS + i, i2)];
    }
    for(int m = 0; m < TILE_M; ++m) {
      if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
        break;
      }

      const REAL uu_i1m5 = uu_GF_ary[0] ; //in_gfs[IDX4(UUGF, i0, i1+m - 5, i2)];
      const REAL uu_i1m4 = uu_GF_ary[1] ; //in_gfs[IDX4(UUGF, i0, i1+m - 4, i2)];
      const REAL uu_i1m3 = uu_GF_ary[2] ; //in_gfs[IDX4(UUGF, i0, i1+m - 3, i2)];
      const REAL uu_i1m2 = uu_GF_ary[3] ; //in_gfs[IDX4(UUGF, i0, i1+m - 2, i2)];
      const REAL uu_i1m1 = uu_GF_ary[4] ; //in_gfs[IDX4(UUGF, i0, i1+m - 1, i2)];

      const REAL uu_i1p1 = uu_GF_ary[6] ; //in_gfs[IDX4(UUGF, i0, i1+m + 1, i2)];
      const REAL uu_i1p2 = uu_GF_ary[7] ; //in_gfs[IDX4(UUGF, i0, i1+m + 2, i2)];
      const REAL uu_i1p3 = uu_GF_ary[8] ; //in_gfs[IDX4(UUGF, i0, i1+m + 3, i2)];
      const REAL uu_i1p4 = uu_GF_ary[9] ; //in_gfs[IDX4(UUGF, i0, i1+m + 4, i2)];
      const REAL uu_i1p5 = uu_GF_ary[10] ; //in_gfs[IDX4(UUGF, i0, i1+m + 5, i2)];

      uu_dD1[m] = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
                                    FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
                                    FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));

      uu_dDD11_part[m] = FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
                                          FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
                                          FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1);
      if(m < TILE_M - 1) {
        for(int i = 0; i < 2U * NGHOSTS; ++i) {
          uu_GF_ary[i] = uu_GF_ary[i+1];
        }
        uu_GF_ary[2U * NGHOSTS] = in_gfs[IDX4(UUGF, i0, i1 + NGHOSTS + m + 1, i2)];
      }
    }
  }

  /*
    * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
    * Read gridfunction(s) from main memory and compute FD stencils as needed.
    */

  // Function of xx0
  [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
  [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
  [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
  [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
  [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
  [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
  [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
  /*
    * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
    * Evaluate SymPy expressions and write to main memory.
    */
  const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
  const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
  const REAL FDPart3tmp7 = 2 / FDPart3tmp4;

  REAL ADD_times_AUU[TILE_M] = {0};
  REAL psi_background[TILE_M] = {0};

  for(int m = 0; m < TILE_M; ++m) {
    if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }
    ADD_times_AUU[m] = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1 + m, i2)];
  }
  for(int m = 0; m < TILE_M; ++m) {
    if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }
    psi_background[m] = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1 + m, i2)];
  }

  // Initialize Local arrays
  REAL uugf[TILE_M] = {0};

  for(int m = 0; m < TILE_M; ++m) {
    if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }
    // rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
    uugf[m] = in_gfs[IDX4(VVGF, i0, i1 + m, i2)];
  }
  REAL vvgf[TILE_M] = {0};

  // Only needed here for VVGF
  // const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
  // const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
  for(int m = 0; m < TILE_M; ++m) {
    if((i1 + m >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }
    const int i1pm = i1 + m;

    [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1pm];
    [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1pm];
    [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1pm];
    [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1pm];
    [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1pm];
    [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1pm];

    const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
    const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
    const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));

    // Function of xx0
    const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1pm, i2)];
    const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1pm, i2)];
    const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1pm, i2)];
    const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1pm, i2)];
    const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1pm, i2)];
    const REAL uu =      in_gfs[IDX4(UUGF, i0    , i1pm, i2)];
    const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1pm, i2)];
    const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1pm, i2)];
    const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1pm, i2)];
    const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1pm, i2)];
    const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1pm, i2)];

    const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
    const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
                                    FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
                                    FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));

    const REAL uu_dDD00 =
        ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
                                    FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
                                    FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);


    //     ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
    //                                 FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
    //                                 FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
    const REAL uu_dDD11 = ((invdxx1) * (invdxx1)) * (uu_dDD11_part[m] + FDPart1tmp0);

    // const REAL uu_dDD22 = ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
    //                                  FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
    //                                  FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);
    const REAL uu_dDD22 = ((invdxx2) * (invdxx2)) * (uu_dDD22_part[m] + FDPart1tmp0);




    const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1pm, i2)];
    vvgf[m] =
          (1.0 / 8.0) * ADD_times_AUU[m] / pow(psi_background[m] + uu, 7) +
          FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
          FDPart3tmp2 * uu_dDD11 +
          FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1[m] / f1_of_xx1 -
          uu_dD0 * (
            -FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 -
            FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
            (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) * (
                FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
                2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
                FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) / ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))
          ) +
          uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1)));
    vvgf[m] *= ((variable_wavespeed) * (variable_wavespeed));
    // rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
    uugf[m] -= eta_damping * uu;
  }

  for(int m = 0; m < TILE_M; ++m) {
    const int i1pm = i1 + m;
    if((i1pm >= Nxx_plus_2NGHOSTS1 - NGHOSTS)) {
      break;
    }

    rhs_gfs[IDX4(UUGF, i0, i1pm, i2)] = uugf[m];
    rhs_gfs[IDX4(VVGF, i0, i1pm, i2)] = vvgf[m];
  }
}

// Baseline
// __global__ static void rhs_eval_gpu(const REAL *restrict rfm_f0_of_xx0, const REAL *restrict rfm_f0_of_xx0__D0,
//                                     const REAL *restrict rfm_f0_of_xx0__DD00, const REAL *restrict rfm_f0_of_xx0__DDD000,
//                                     const REAL *restrict rfm_f1_of_xx1, const REAL *restrict rfm_f1_of_xx1__D1,
//                                     const REAL *restrict rfm_f1_of_xx1__DD11, const REAL *restrict rfm_f2_of_xx0,
//                                     const REAL *restrict rfm_f2_of_xx0__D0, const REAL *restrict rfm_f2_of_xx0__DD00,
//                                     const REAL *restrict rfm_f4_of_xx1, const REAL *restrict rfm_f4_of_xx1__D1,
//                                     const REAL *restrict rfm_f4_of_xx1__DD11, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
//                                     REAL *restrict rhs_gfs, const REAL eta_damping) {

//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

//   [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
//   [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
//   [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

//   const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
//   const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
//   const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

//   const int stride0 = blockDim.x * gridDim.x;
//   const int stride1 = blockDim.y * gridDim.y;
//   const int stride2 = blockDim.z * gridDim.z;

//   for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
//     for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
//       [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1];
//       [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1];

//       for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
//         [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
//         [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
//          * Read gridfunction(s) from main memory and compute FD stencils as needed.
//          */
//         const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
//         const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
//         const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1, i2 - 5)];
//         const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1, i2 - 4)];
//         const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1, i2 - 3)];
//         const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
//         const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
//         const REAL uu_i1m5 = in_gfs[IDX4(UUGF, i0, i1 - 5, i2)];
//         const REAL uu_i1m4 = in_gfs[IDX4(UUGF, i0, i1 - 4, i2)];
//         const REAL uu_i1m3 = in_gfs[IDX4(UUGF, i0, i1 - 3, i2)];
//         const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
//         const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
//         const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1, i2)];
//         const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1, i2)];
//         const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1, i2)];
//         const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
//         const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
//         const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
//         const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
//         const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
//         const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1, i2)];
//         const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1, i2)];
//         const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1, i2)];
//         const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
//         const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
//         const REAL uu_i1p3 = in_gfs[IDX4(UUGF, i0, i1 + 3, i2)];
//         const REAL uu_i1p4 = in_gfs[IDX4(UUGF, i0, i1 + 4, i2)];
//         const REAL uu_i1p5 = in_gfs[IDX4(UUGF, i0, i1 + 5, i2)];
//         const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
//         const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
//         const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1, i2 + 3)];
//         const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1, i2 + 4)];
//         const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1, i2 + 5)];
//         const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)];
//         const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
//         static constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
//         static constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
//         static constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
//         static constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
//         static constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
//         static constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
//         static constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
//         static constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
//         static constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
//         static constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;
//         const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
//         const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
//         const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
//         const REAL uu_dDD00 =
//             ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
//                                        FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
//         const REAL uu_dDD11 =
//             ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
//                                        FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
//         const REAL uu_dDD22 =
//             ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
//                                        FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
//                                        FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
//          * Evaluate SymPy expressions and write to main memory.
//          */
//         const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
//         const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
//         const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
//         const REAL FDPart3tmp7 = 2 / FDPart3tmp4;
//         const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
//         const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
//         rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
//             ((variable_wavespeed) * (variable_wavespeed)) *
//             ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
//              FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
//              uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
//                        (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
//                            (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
//                             2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
//                             FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
//                            ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
//              uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));

//       } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
//     } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
//   } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
// }

// Revision 1.2
// #define TILE_M 1
// __global__ static void rhs_eval_gpu(const REAL *restrict rfm_f0_of_xx0, const REAL *restrict rfm_f0_of_xx0__D0,
//                                     const REAL *restrict rfm_f0_of_xx0__DD00, const REAL *restrict rfm_f0_of_xx0__DDD000,
//                                     const REAL *restrict rfm_f1_of_xx1, const REAL *restrict rfm_f1_of_xx1__D1,
//                                     const REAL *restrict rfm_f1_of_xx1__DD11, const REAL *restrict rfm_f2_of_xx0,
//                                     const REAL *restrict rfm_f2_of_xx0__D0, const REAL *restrict rfm_f2_of_xx0__DD00,
//                                     const REAL *restrict rfm_f4_of_xx1, const REAL *restrict rfm_f4_of_xx1__D1,
//                                     const REAL *restrict rfm_f4_of_xx1__DD11, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
//                                     REAL *restrict rhs_gfs, const REAL eta_damping) {

//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

//   [[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
//   [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
//   [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;

//   const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
//   const int tid1 = TILE_M * (blockIdx.y * blockDim.y + threadIdx.y + NGHOSTS);
//   const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

//   int i2 = tid2 + NGHOSTS;
//   int i1 = tid1; // + NGHOSTS;
//   int i0 = tid0 + NGHOSTS;

//   if( (i2 >= Nxx_plus_2NGHOSTS2 - NGHOSTS)
//     // ||(i1 >= Nxx_plus_2NGHOSTS1 - NGHOSTS)
//     ||(i0 >= Nxx_plus_2NGHOSTS0 - NGHOSTS) ) {

//     return;
//   }
//   REAL uu_gf_ary[2U * NGHOSTS + 1U + TILE_M] = {0};

//       [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1];
//       [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1];
//       [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1];

//       // for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
//         [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
//         [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
//         [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
//         [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
//          * Read gridfunction(s) from main memory and compute FD stencils as needed.
//          */
//         const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
//         const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
//         const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1, i2 - 5)];
//         const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1, i2 - 4)];
//         const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1, i2 - 3)];
//         const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
//         const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
//         const REAL uu_i1m5 = in_gfs[IDX4(UUGF, i0, i1 - 5, i2)];
//         const REAL uu_i1m4 = in_gfs[IDX4(UUGF, i0, i1 - 4, i2)];
//         const REAL uu_i1m3 = in_gfs[IDX4(UUGF, i0, i1 - 3, i2)];
//         const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
//         const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
//         const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1, i2)];
//         const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1, i2)];
//         const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1, i2)];
//         const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
//         const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
//         const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
//         const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
//         const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
//         const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1, i2)];
//         const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1, i2)];
//         const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1, i2)];
//         const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
//         const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
//         const REAL uu_i1p3 = in_gfs[IDX4(UUGF, i0, i1 + 3, i2)];
//         const REAL uu_i1p4 = in_gfs[IDX4(UUGF, i0, i1 + 4, i2)];
//         const REAL uu_i1p5 = in_gfs[IDX4(UUGF, i0, i1 + 5, i2)];
//         const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
//         const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
//         const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1, i2 + 3)];
//         const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1, i2 + 4)];
//         const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1, i2 + 5)];
//         const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)];
//         const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
//         static constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
//         static constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
//         static constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
//         static constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
//         static constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
//         static constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
//         static constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
//         static constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
//         static constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
//         static constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;
//         const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
//         const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
//         const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
//                                        FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
//         const REAL uu_dDD00 =
//             ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
//                                        FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
//                                        FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
//         const REAL uu_dDD11 =
//             ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
//                                        FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
//                                        FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
//         const REAL uu_dDD22 =
//             ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
//                                        FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
//                                        FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
//          * Evaluate SymPy expressions and write to main memory.
//          */
//         const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
//         const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
//         const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
//         const REAL FDPart3tmp7 = 2 / FDPart3tmp4;
//         const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
//         const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
//         rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
//             ((variable_wavespeed) * (variable_wavespeed)) *
//             ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
//              FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
//              uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
//                        (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
//                            (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
//                             2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
//                             FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
//                            ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
//              uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));

//   //     } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
//   //   } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
//   // } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
// }

/*
 * Set RHSs for hyperbolic relaxation equation.
 */
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
              const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
#include "set_CodeParameters.h"
  const REAL *restrict rfm_f0_of_xx0 = rfmstruct->f0_of_xx0;
  const REAL *restrict rfm_f0_of_xx0__D0 = rfmstruct->f0_of_xx0__D0;
  const REAL *restrict rfm_f0_of_xx0__DD00 = rfmstruct->f0_of_xx0__DD00;
  const REAL *restrict rfm_f0_of_xx0__DDD000 = rfmstruct->f0_of_xx0__DDD000;
  const REAL *restrict rfm_f1_of_xx1 = rfmstruct->f1_of_xx1;
  const REAL *restrict rfm_f1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1;
  const REAL *restrict rfm_f1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11;
  const REAL *restrict rfm_f2_of_xx0 = rfmstruct->f2_of_xx0;
  const REAL *restrict rfm_f2_of_xx0__D0 = rfmstruct->f2_of_xx0__D0;
  const REAL *restrict rfm_f2_of_xx0__DD00 = rfmstruct->f2_of_xx0__DD00;
  const REAL *restrict rfm_f4_of_xx1 = rfmstruct->f4_of_xx1;
  const REAL *restrict rfm_f4_of_xx1__D1 = rfmstruct->f4_of_xx1__D1;
  const REAL *restrict rfm_f4_of_xx1__DD11 = rfmstruct->f4_of_xx1__DD11;

  const size_t threads_in_x_dir = 256;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                       (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir / TILE_M,
                       (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rhs_eval_gpu<<<blocks_per_grid,threads_per_block, sm, streams[streamid]>>>(
      rfm_f0_of_xx0, rfm_f0_of_xx0__D0, rfm_f0_of_xx0__DD00, rfm_f0_of_xx0__DDD000, rfm_f1_of_xx1, rfm_f1_of_xx1__D1, rfm_f1_of_xx1__DD11,
      rfm_f2_of_xx0, rfm_f2_of_xx0__D0, rfm_f2_of_xx0__DD00, rfm_f4_of_xx1, rfm_f4_of_xx1__D1, rfm_f4_of_xx1__DD11, auxevol_gfs, in_gfs, rhs_gfs,
      eta_damping);
  cudaCheckErrors(cudaKernel, "rhs_eval_gpu failure");
}
