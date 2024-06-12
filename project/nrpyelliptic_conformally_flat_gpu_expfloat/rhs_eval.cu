#include "BHaH_defines.h"
#include "expansion_math.h"
#define vecsize 1
/*
 * GPU Kernel: rhs_eval_gpu expansion.
 * GPU Kernel to evaluate RHS on the interior.
 */
// __global__ static void rhs_eval_expansion_gpu(const float *restrict rfm_f0_of_xx0, const float *restrict rfm_f0_of_xx0__D0,
//                                     const float *restrict rfm_f0_of_xx0__DD00, const float *restrict rfm_f0_of_xx0__DDD000,
//                                     const float *restrict rfm_f1_of_xx1, const float *restrict rfm_f1_of_xx1__D1,
//                                     const float *restrict rfm_f1_of_xx1__DD11, const float *restrict rfm_f2_of_xx0,
//                                     const float *restrict rfm_f2_of_xx0__D0, const float *restrict rfm_f2_of_xx0__DD00,
//                                     const float *restrict rfm_f4_of_xx1, const float *restrict rfm_f4_of_xx1__D1,
//                                     const float *restrict rfm_f4_of_xx1__DD11, const float *restrict auxevol_gfs, const float *restrict in_gfs,
//                                     float *restrict rhs_gfs, const float eta_damping) {

//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

//   [[maybe_unused]] const REAL invdxx0_ = d_params.invdxx0;
//   constexpr expansion_math::float2<float> invdxx0 = expansion_math::split<float>(invdxx0_);
//   [[maybe_unused]] const REAL invdxx1_ = d_params.invdxx1_;
//   constexpr expansion_math::float2<float> invdxx1 = expansion_math::split<float>(invdxx1_);
//   [[maybe_unused]] const REAL invdxx2_ = d_params.invdxx2;
//   constexpr expansion_math::float2<float> invdxx2 = expansion_math::split<float>(invdxx2_);

//   const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
//   const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
//   const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

//   const int stride0 = blockDim.x * gridDim.x;
//   const int stride1 = blockDim.y * gridDim.y;
//   const int stride2 = blockDim.z * gridDim.z;

//   for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
//     for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
//       [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1(rfm_f1_of_xx1[i1], rfm_f1_of_xx1[i1+1]);
//       [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1__D1(rfm_f1_of_xx1__D1[i1], rfm_f1_of_xx1__D1[i1+1]);
//       [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1__DD11(rfm_f1_of_xx1__DD11[i1], rfm_f1_of_xx1__DD11[i1+1]);
//       [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1(rfm_f4_of_xx1[i1], rfm_f4_of_xx1[i1+1]);
//       [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1__D1(rfm_f4_of_xx1__D1[i1], rfm_f4_of_xx1__D1[i1+1]);
//       [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1__DD11(rfm_f4_of_xx1__DD11[i1], rfm_f4_of_xx1__DD11[i1+1]);

//       for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
//         [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0(rfm_f0_of_xx0[i0], rfm_f0_of_xx0[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__D0(rfm_f0_of_xx0__D0[i0], rfm_f0_of_xx0__D0[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__DD00(rfm_f0_of_xx0__DD00[i0], rfm_f0_of_xx0__DD00[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__DDD000(rfm_f0_of_xx0__DDD000[i0], rfm_f0_of_xx0__DDD000[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0(rfm_f2_of_xx0[i0], rfm_f2_of_xx0[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0__D0(rfm_f2_of_xx0__D0[i0], rfm_f2_of_xx0__D0[i0+1]);
//         [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0__DD00(rfm_f2_of_xx0__DD00[i0], rfm_f2_of_xx0__DD00[i0+1]);
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
//          * Read gridfunction(s) from main memory and compute FD stencils as needed.
//          */
        
        
//         const expansion_math::float2<float> uu_i2m5(in_gfs[IDX4(UUGF, i0, i1, i2 - 5)], in_gfs[IDX4(UUGF, i0, i1, i2 - 5)+1]);
//         const expansion_math::float2<float> uu_i2m4(in_gfs[IDX4(UUGF, i0, i1, i2 - 4)], in_gfs[IDX4(UUGF, i0, i1, i2 - 4)+1]);
//         const expansion_math::float2<float> uu_i2m3(in_gfs[IDX4(UUGF, i0, i1, i2 - 3)], in_gfs[IDX4(UUGF, i0, i1, i2 - 3)+1]);
//         const expansion_math::float2<float> uu_i2m2(in_gfs[IDX4(UUGF, i0, i1, i2 - 2)], in_gfs[IDX4(UUGF, i0, i1, i2 - 2)+1]);
//         const expansion_math::float2<float> uu_i2m1(in_gfs[IDX4(UUGF, i0, i1, i2 - 1)], in_gfs[IDX4(UUGF, i0, i1, i2 - 1)+1]);
//         const expansion_math::float2<float> uu_i1m5(in_gfs[IDX4(UUGF, i0, i1 - 5, i2)], in_gfs[IDX4(UUGF, i0, i1 - 5, i2)+1]);
//         const expansion_math::float2<float> uu_i1m4(in_gfs[IDX4(UUGF, i0, i1 - 4, i2)], in_gfs[IDX4(UUGF, i0, i1 - 4, i2)+1]);
//         const expansion_math::float2<float> uu_i1m3(in_gfs[IDX4(UUGF, i0, i1 - 3, i2)], in_gfs[IDX4(UUGF, i0, i1 - 3, i2)+1]);
//         const expansion_math::float2<float> uu_i1m2(in_gfs[IDX4(UUGF, i0, i1 - 2, i2)], in_gfs[IDX4(UUGF, i0, i1 - 2, i2)+1]);
//         const expansion_math::float2<float> uu_i1m1(in_gfs[IDX4(UUGF, i0, i1 - 1, i2)], in_gfs[IDX4(UUGF, i0, i1 - 1, i2)+1]);
//         const expansion_math::float2<float> uu_i0m5(in_gfs[IDX4(UUGF, i0 - 5, i1, i2)], in_gfs[IDX4(UUGF, i0 - 5, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0m4(in_gfs[IDX4(UUGF, i0 - 4, i1, i2)], in_gfs[IDX4(UUGF, i0 - 4, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0m3(in_gfs[IDX4(UUGF, i0 - 3, i1, i2)], in_gfs[IDX4(UUGF, i0 - 3, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0m2(in_gfs[IDX4(UUGF, i0 - 2, i1, i2)], in_gfs[IDX4(UUGF, i0 - 2, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0m1(in_gfs[IDX4(UUGF, i0 - 1, i1, i2)], in_gfs[IDX4(UUGF, i0 - 1, i1, i2)+1]);
//         const expansion_math::float2<float> uu(in_gfs[IDX4(UUGF, i0, i1, i2)], in_gfs[IDX4(UUGF, i0, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0p1(in_gfs[IDX4(UUGF, i0 + 1, i1, i2)], in_gfs[IDX4(UUGF, i0 + 1, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0p2(in_gfs[IDX4(UUGF, i0 + 2, i1, i2)], in_gfs[IDX4(UUGF, i0 + 2, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0p3(in_gfs[IDX4(UUGF, i0 + 3, i1, i2)], in_gfs[IDX4(UUGF, i0 + 3, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0p4(in_gfs[IDX4(UUGF, i0 + 4, i1, i2)], in_gfs[IDX4(UUGF, i0 + 4, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i0p5(in_gfs[IDX4(UUGF, i0 + 5, i1, i2)], in_gfs[IDX4(UUGF, i0 + 5, i1, i2)+1]);
//         const expansion_math::float2<float> uu_i1p1(in_gfs[IDX4(UUGF, i0, i1 + 1, i2)], in_gfs[IDX4(UUGF, i0, i1 + 1, i2)+1]);
//         const expansion_math::float2<float> uu_i1p2(in_gfs[IDX4(UUGF, i0, i1 + 2, i2)], in_gfs[IDX4(UUGF, i0, i1 + 2, i2)+1]);
//         const expansion_math::float2<float> uu_i1p3(in_gfs[IDX4(UUGF, i0, i1 + 3, i2)], in_gfs[IDX4(UUGF, i0, i1 + 3, i2)+1]);
//         const expansion_math::float2<float> uu_i1p4(in_gfs[IDX4(UUGF, i0, i1 + 4, i2)], in_gfs[IDX4(UUGF, i0, i1 + 4, i2)+1]);
//         const expansion_math::float2<float> uu_i1p5(in_gfs[IDX4(UUGF, i0, i1 + 5, i2)], in_gfs[IDX4(UUGF, i0, i1 + 5, i2)+1]);
//         const expansion_math::float2<float> uu_i2p1(in_gfs[IDX4(UUGF, i0, i1, i2 + 1)], in_gfs[IDX4(UUGF, i0, i1, i2 + 1)+1]);
//         const expansion_math::float2<float> uu_i2p2(in_gfs[IDX4(UUGF, i0, i1, i2 + 2)], in_gfs[IDX4(UUGF, i0, i1, i2 + 2)+1]);
//         const expansion_math::float2<float> uu_i2p3(in_gfs[IDX4(UUGF, i0, i1, i2 + 3)], in_gfs[IDX4(UUGF, i0, i1, i2 + 3)+1]);
//         const expansion_math::float2<float> uu_i2p4(in_gfs[IDX4(UUGF, i0, i1, i2 + 4)], in_gfs[IDX4(UUGF, i0, i1, i2 + 4)+1]);
//         const expansion_math::float2<float> uu_i2p5(in_gfs[IDX4(UUGF, i0, i1, i2 + 5)], in_gfs[IDX4(UUGF, i0, i1, i2 + 5)+1]);
        
//         const expansion_math::float2<float> variable_wavespeed(auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)], auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)+1]);
//         const expansion_math::float2<float> vv(in_gfs[IDX4(VVGF, i0, i1, i2)], in_gfs[IDX4(VVGF, i0, i1, i2)+1]);
        
//         constexpr REAL FDPart1_Rational_5_6_ = 5.0 / 6.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_6 = expansion_math::split<float>(FDPart1_Rational_5_6_);
//         constexpr REAL FDPart1_Rational_5_21_ = 5.0 / 21.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_21 = expansion_math::split<float>(FDPart1_Rational_5_21_);
//         constexpr REAL FDPart1_Rational_5_84_ = 5.0 / 84.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_84 = expansion_math::split<float>(FDPart1_Rational_5_84_);
//         constexpr REAL FDPart1_Rational_5_504_ = 5.0 / 504.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_504 = expansion_math::split<float>(FDPart1_Rational_5_504_);
//         constexpr REAL FDPart1_Rational_1_1260_ = 1.0 / 1260.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_1_1260 = expansion_math::split<float>(FDPart1_Rational_1_1260);
//         constexpr REAL FDPart1_Rational_5269_1800_ = 5269.0 / 1800.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5269_1800 = expansion_math::split<float>(FDPart1_Rational_5269_1800_);
//         constexpr REAL FDPart1_Rational_5_1008_ = 5.0 / 1008.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_1008 = expansion_math::split<float>(FDPart1_Rational_5_1008_);
//         constexpr REAL FDPart1_Rational_1_3150_ = 1.0 / 3150.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_1_3150 = expansion_math::split<float>(FDPart1_Rational_1_3150_);
//         constexpr REAL FDPart1_Rational_5_3_ = 5.0 / 3.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_3 = expansion_math::split<float>(FDPart1_Rational_5_3_);
//         constexpr REAL FDPart1_Rational_5_126_ = 5.0 / 126.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_5_126 = expansion_math::split<float>(FDPart1_Rational_5_126_);
        
//         const expansion_math::float2<float> FDPart1tmp0 = expansion_math::scale_expansion(-FDPart1_Rational_5269_1800, uu);
        
//         // const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
//         //                         FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
//         //                         FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
//         const expansion_math::float2<float> uu_dD0 = expansion_math::scale_expansion(invdxx0,
//           expansion_math::grow_expansion(
//             expansion_math::scale_expansion(FDPart1_Rational_1_1260, expansion_math::grow_expansion(-uu_i0m5, uu_i0p5)), 
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(FDPart1_Rational_5_21, expansion_math::grow_expansion(uu_i0m2, -uu_i0p2)),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_504, expansion_math::grow_expansion(uu_i0m4, -uu_i0p4)), 
//                 expansion_math::grow_expansion(
//                   expansion_math::scale_expansion(FDPart1_Rational_5_6, expansion_math::grow_expansion(-uu_i0m1, uu_i0p1)),
//                   expansion_math::scale_expansion(FDPart1_Rational_5_84  , expansion_math::grow_expansion(-uu_i0m3, uu_i0p3))
//                 )
//               )
//             )
//           )
//         );
        
//         // const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
//         //                                FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
//         //                                FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
//         const expansion_math::float2<float> uu_dD1 = expansion_math::scale_expansion(invdxx1,
//           expansion_math::grow_expansion(
//             expansion_math::scale_expansion(FDPart1_Rational_1_1260, expansion_math::grow_expansion(-uu_i1m5, uu_i1p5)), 
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(FDPart1_Rational_5_21, expansion_math::grow_expansion(uu_i1m2, -uu_i1p2)),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_504, expansion_math::grow_expansion(uu_i1m4, -uu_i1p4)), 
//                 expansion_math::grow_expansion(
//                   expansion_math::scale_expansion(FDPart1_Rational_5_6, expansion_math::grow_expansion(-uu_i1m1, uu_i1p1)),
//                   expansion_math::scale_expansion(FDPart1_Rational_5_84  , expansion_math::grow_expansion(-uu_i1m3, uu_i1p3))
//                 )
//               )
//             )
//           )
//         );

//         // const REAL uu_dDD00 =
//         //     ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
//         //                                FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
//         //                                FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
//         const expansion_math::float2<float> uu_dDD00 = expansion_math::scale_expansion(
//           expansion_math::scale_expansion(invdxx0, invdxx0),
//           expansion_math::grow_expansion(
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(FDPart1_Rational_1_3150, expansion_math::grow_expansion(uu_i0m5, uu_i0p5)),
//               expansion_math::scale_expansion(FDPart1_Rational_5_1008, expansion_math::grow_expansion(-uu_i0m4, -uu_i0p4))
//             ),
//             expansion_math::grow_expansion(
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_126, expansion_math::grow_expansion(uu_i0m3, uu_i0p3)),
//                 expansion_math::scale_expansion(FDPart1_Rational_5_21, expansion_math::grow_expansion(-uu_i0m2, -uu_i0p2))
//               ),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_3, expansion_math::grow_expansion(uu_i0m1, uu_i0p1)), 
//                 FDPart1tmp0
//               )
//             )
//           )
//         );

//         // const REAL uu_dDD11 =
//         //     ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
//         //                                FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
//         //                                FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
//         const expansion_math::float2<float> uu_dDD11 = expansion_math::scale_expansion(
//           expansion_math::scale_expansion(invdxx1, invdxx1),
//           expansion_math::grow_expansion(
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(FDPart1_Rational_1_3150, expansion_math::grow_expansion(uu_i1m5, uu_i1p5)),
//               expansion_math::scale_expansion(FDPart1_Rational_5_1008, expansion_math::grow_expansion(-uu_i1m4, -uu_i1p4))
//             ),
//             expansion_math::grow_expansion(
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_126, expansion_math::grow_expansion(uu_i1m3, uu_i1p3)),
//                 expansion_math::scale_expansion(FDPart1_Rational_5_21, expansion_math::grow_expansion(-uu_i1m2, -uu_i1p2))
//               ),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_3, expansion_math::grow_expansion(uu_i1m1, uu_i1p1)), 
//                 FDPart1tmp0
//               )
//             )
//           )
//         );        
//         // const REAL uu_dDD22 =
//         //     ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
//         //                                FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
//         //                                FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);
//         const expansion_math::float2<float> uu_dDD11 = expansion_math::scale_expansion(
//           expansion_math::scale_expansion(invdxx1, invdxx1),
//           expansion_math::grow_expansion(
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(FDPart1_Rational_1_3150, expansion_math::grow_expansion(uu_i2m5, uu_i2p5)),
//               expansion_math::scale_expansion(FDPart1_Rational_5_1008, expansion_math::grow_expansion(-uu_i2m4, -uu_i2p4))
//             ),
//             expansion_math::grow_expansion(
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_126, expansion_math::grow_expansion(uu_i2m3, uu_i2p3)),
//                 expansion_math::scale_expansion(FDPart1_Rational_5_21, expansion_math::grow_expansion(-uu_i2m2, -uu_i2p2))
//               ),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(FDPart1_Rational_5_3, expansion_math::grow_expansion(uu_i2m1, uu_i2p1)), 
//                 FDPart1tmp0
//               )
//             )
//           )
//         ); 
//         /*
//          * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
//          * Evaluate SymPy expressions and write to main memory.
//          */
//         const expansion_math::float2<float> FDPart3tmp4 = expansion_math::scale_expansion((f2_of_xx0), (f2_of_xx0));
//         const expansion_math::float2<float> FDPart3tmp1 = expansion_math::grow_expansion(expansion_math::scale_expansion((f0_of_xx0), (f0_of_xx0)), expansion_math::scale_expansion((f4_of_xx1), (f4_of_xx1)));
//         const expansion_math::float2<float> FDPart3tmp6 = division_expansion(FDPart3tmp4, f0_of_xx0__D0);
//         const expansion_math::float2<float> FDPart3tmp7 = division_expansion(expansion_math::float2<float>(2.0,0), FDPart3tmp4);
//         const expansion_math::float2<float> FDPart3tmp2 = division_expansion(expansion_math::float2<float>(1.0,0), FDPart3tmp1);
//         const expansion_math::float2<float> FDPart3tmp5 = division_expansion(expansion_math::float2<float>(1.0,0), expansion_math::scale_expansion(((FDPart3tmp1), (FDPart3tmp1))));
//         const expansion_math::float2<float> resUU = expansion_math::grow_expansion(expansion_math::scale_expansion(-eta_damping, uu), vv);
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)] = resUU.value;
//         rhs_gfs[IDX4(UUGF, i0, i1, i2)+1] = resUU.remainder;

//         constexpr REAL FDPart1_Rational_1_8_ = 1.0 / 8.0;
//         constexpr expansion_math::float2<float> FDPart1_Rational_1_8 = expansion_math::split<float>(FDPart1_Rational_1_8_);
//         const expansion_math::float2<float> ADD_times_AUU(auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)], auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)+1]);
//         const expansion_math::float2<float> psi_background(auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)], auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)+1]);
//         // rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
//         //     ((variable_wavespeed) * (variable_wavespeed)) *
//         //     ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
//         //      FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
//         //      uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
//         //                (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
//         //                    (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
//         //                     2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
//         //                     FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
//         //                    ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
//         //      uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));        
//         const expansion_math::float2<float> resVV = expansion_math::scale_expansion(
//             expansion_math::scale_expansion(variable_wavespeed, variable_wavespeed),
//             expansion_math::grow_expansion(
//               expansion_math::scale_expansion(
//                 FDPart1_Rational_1_8,
//                 expansion_math::division_expansion(
//                   ADD_times_AUU, 
//                   expansion_math::pow_expansion(
//                     expansion_math::grow_expansion(psi_background + uu), 
//                     7
//                   )
//                 )
//               ),
//               expansion_math::grow_expansion(
//                 expansion_math::scale_expansion(
//                   FDPart3tmp2, 
//                   expansion_math::scale_expansion(
//                     FDPart3tmp4, 
//                     expansion_math::division_expansion(
//                       uu_dDD00, 
//                       expansion_math::scale_expansion(f0_of_xx0__D0, f0_of_xx0__D0)
//                     )
//                   )
//                 ), +
//                 expansion_math::scale_expansion(
//                   FDPart3tmp2,
//                   uu_dDD11
//                 ) + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
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

/*
 * GPU Kernel: rhs_eval_gpu.
 * GPU Kernel to evaluate RHS on the interior.
 */
__global__ static void rhs_eval_expansion_gpu(const float *restrict rfm_f0_of_xx0, const float *restrict rfm_f0_of_xx0__D0,
                                    const float *restrict rfm_f0_of_xx0__DD00, const float *restrict rfm_f0_of_xx0__DDD000,
                                    const float *restrict rfm_f1_of_xx1, const float *restrict rfm_f1_of_xx1__D1,
                                    const float *restrict rfm_f1_of_xx1__DD11, const float *restrict rfm_f2_of_xx0,
                                    const float *restrict rfm_f2_of_xx0__D0, const float *restrict rfm_f2_of_xx0__DD00,
                                    const float *restrict rfm_f4_of_xx1, const float *restrict rfm_f4_of_xx1__D1,
                                    const float *restrict rfm_f4_of_xx1__DD11, const float *restrict auxevol_gfs, const float *restrict in_gfs,
                                    float *restrict rhs_gfs, const expansion_math::float2<float> eta_damping) {

  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  [[maybe_unused]] const REAL invdxx0_ = d_params.invdxx0;
  const expansion_math::float2<float> invdxx0 = expansion_math::split<float>(invdxx0_);
  [[maybe_unused]] const REAL invdxx1_ = d_params.invdxx1;
  const expansion_math::float2<float> invdxx1 = expansion_math::split<float>(invdxx1_);
  [[maybe_unused]] const REAL invdxx2_ = d_params.invdxx2;
  const expansion_math::float2<float> invdxx2 = expansion_math::split<float>(invdxx2_);

  const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
    for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
      [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1(rfm_f1_of_xx1[i1], rfm_f1_of_xx1[i1+1]);
      [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1__D1(rfm_f1_of_xx1__D1[i1], rfm_f1_of_xx1__D1[i1+1]);
      [[maybe_unused]] const expansion_math::float2<float> f1_of_xx1__DD11(rfm_f1_of_xx1__DD11[i1], rfm_f1_of_xx1__DD11[i1+1]);
      [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1(rfm_f4_of_xx1[i1], rfm_f4_of_xx1[i1+1]);
      [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1__D1(rfm_f4_of_xx1__D1[i1], rfm_f4_of_xx1__D1[i1+1]);
      [[maybe_unused]] const expansion_math::float2<float> f4_of_xx1__DD11(rfm_f4_of_xx1__DD11[i1], rfm_f4_of_xx1__DD11[i1+1]);

      for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
        [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0(rfm_f0_of_xx0[i0], rfm_f0_of_xx0[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__D0(rfm_f0_of_xx0__D0[i0], rfm_f0_of_xx0__D0[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__DD00(rfm_f0_of_xx0__DD00[i0], rfm_f0_of_xx0__DD00[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f0_of_xx0__DDD000(rfm_f0_of_xx0__DDD000[i0], rfm_f0_of_xx0__DDD000[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0(rfm_f2_of_xx0[i0], rfm_f2_of_xx0[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0__D0(rfm_f2_of_xx0__D0[i0], rfm_f2_of_xx0__D0[i0+1]);
        [[maybe_unused]] const expansion_math::float2<float> f2_of_xx0__DD00(rfm_f2_of_xx0__DD00[i0], rfm_f2_of_xx0__DD00[i0+1]);
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        #define IDX4F(g, i, j, k, p) (2 * ((i) + Nxx_plus_2NGHOSTS0 * ((j) + Nxx_plus_2NGHOSTS1 * ((k) + Nxx_plus_2NGHOSTS2 * (g)))) + p)
        const expansion_math::float2<float> uu_i2m5(in_gfs[IDX4F(UUGF, i0, i1, i2 - 5, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 - 5, 1)]);
        const expansion_math::float2<float> uu_i2m4(in_gfs[IDX4F(UUGF, i0, i1, i2 - 4, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 - 4, 1)]);
        const expansion_math::float2<float> uu_i2m3(in_gfs[IDX4F(UUGF, i0, i1, i2 - 3, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 - 3, 1)]);
        const expansion_math::float2<float> uu_i2m2(in_gfs[IDX4F(UUGF, i0, i1, i2 - 2, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 - 2, 1)]);
        const expansion_math::float2<float> uu_i2m1(in_gfs[IDX4F(UUGF, i0, i1, i2 - 1, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 - 1, 1)]);
        const expansion_math::float2<float> uu_i1m5(in_gfs[IDX4F(UUGF, i0, i1 - 5, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 - 5, i2, 1)]);
        const expansion_math::float2<float> uu_i1m4(in_gfs[IDX4F(UUGF, i0, i1 - 4, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 - 4, i2, 1)]);
        const expansion_math::float2<float> uu_i1m3(in_gfs[IDX4F(UUGF, i0, i1 - 3, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 - 3, i2, 1)]);
        const expansion_math::float2<float> uu_i1m2(in_gfs[IDX4F(UUGF, i0, i1 - 2, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 - 2, i2, 1)]);
        const expansion_math::float2<float> uu_i1m1(in_gfs[IDX4F(UUGF, i0, i1 - 1, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 - 1, i2, 1)]);
        const expansion_math::float2<float> uu_i0m5(in_gfs[IDX4F(UUGF, i0 - 5, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 - 5, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0m4(in_gfs[IDX4F(UUGF, i0 - 4, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 - 4, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0m3(in_gfs[IDX4F(UUGF, i0 - 3, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 - 3, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0m2(in_gfs[IDX4F(UUGF, i0 - 2, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 - 2, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0m1(in_gfs[IDX4F(UUGF, i0 - 1, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 - 1, i1, i2, 1)]);
        const expansion_math::float2<float> uu(in_gfs[IDX4F(UUGF, i0, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0p1(in_gfs[IDX4F(UUGF, i0 + 1, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 + 1, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0p2(in_gfs[IDX4F(UUGF, i0 + 2, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 + 2, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0p3(in_gfs[IDX4F(UUGF, i0 + 3, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 + 3, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0p4(in_gfs[IDX4F(UUGF, i0 + 4, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 + 4, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i0p5(in_gfs[IDX4F(UUGF, i0 + 5, i1, i2, 0)], in_gfs[IDX4F(UUGF, i0 + 5, i1, i2, 1)]);
        const expansion_math::float2<float> uu_i1p1(in_gfs[IDX4F(UUGF, i0, i1 + 1, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 + 1, i2, 1)]);
        const expansion_math::float2<float> uu_i1p2(in_gfs[IDX4F(UUGF, i0, i1 + 2, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 + 2, i2, 1)]);
        const expansion_math::float2<float> uu_i1p3(in_gfs[IDX4F(UUGF, i0, i1 + 3, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 + 3, i2, 1)]);
        const expansion_math::float2<float> uu_i1p4(in_gfs[IDX4F(UUGF, i0, i1 + 4, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 + 4, i2, 1)]);
        const expansion_math::float2<float> uu_i1p5(in_gfs[IDX4F(UUGF, i0, i1 + 5, i2, 0)], in_gfs[IDX4F(UUGF, i0, i1 + 5, i2, 1)]);
        const expansion_math::float2<float> uu_i2p1(in_gfs[IDX4F(UUGF, i0, i1, i2 + 1, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 + 1, 1)]);
        const expansion_math::float2<float> uu_i2p2(in_gfs[IDX4F(UUGF, i0, i1, i2 + 2, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 + 2, 1)]);
        const expansion_math::float2<float> uu_i2p3(in_gfs[IDX4F(UUGF, i0, i1, i2 + 3, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 + 3, 1)]);
        const expansion_math::float2<float> uu_i2p4(in_gfs[IDX4F(UUGF, i0, i1, i2 + 4, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 + 4, 1)]);
        const expansion_math::float2<float> uu_i2p5(in_gfs[IDX4F(UUGF, i0, i1, i2 + 5, 0)], in_gfs[IDX4F(UUGF, i0, i1, i2 + 5, 1)]);
        
        constexpr REAL FDPart1_Rational_5_6_ = 5.0 / 6.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_6 = expansion_math::split<float>(FDPart1_Rational_5_6_);
        constexpr REAL FDPart1_Rational_5_21_ = 5.0 / 21.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_21 = expansion_math::split<float>(FDPart1_Rational_5_21_);
        constexpr REAL FDPart1_Rational_5_84_ = 5.0 / 84.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_84 = expansion_math::split<float>(FDPart1_Rational_5_84_);
        constexpr REAL FDPart1_Rational_5_504_ = 5.0 / 504.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_504 = expansion_math::split<float>(FDPart1_Rational_5_504_);
        constexpr REAL FDPart1_Rational_1_1260_ = 1.0 / 1260.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_1_1260 = expansion_math::split<float>(FDPart1_Rational_1_1260_);
        constexpr REAL FDPart1_Rational_5269_1800_ = 5269.0 / 1800.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5269_1800 = expansion_math::split<float>(FDPart1_Rational_5269_1800_);
        constexpr REAL FDPart1_Rational_5_1008_ = 5.0 / 1008.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_1008 = expansion_math::split<float>(FDPart1_Rational_5_1008_);
        constexpr REAL FDPart1_Rational_1_3150_ = 1.0 / 3150.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_1_3150 = expansion_math::split<float>(FDPart1_Rational_1_3150_);
        constexpr REAL FDPart1_Rational_5_3_ = 5.0 / 3.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_3 = expansion_math::split<float>(FDPart1_Rational_5_3_);
        constexpr REAL FDPart1_Rational_5_126_ = 5.0 / 126.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_5_126 = expansion_math::split<float>(FDPart1_Rational_5_126_);

        const expansion_math::float2<float> FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
        const expansion_math::float2<float> uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
                                       FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
                                       FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
        const expansion_math::float2<float> uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
                                       FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
                                       FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
        const expansion_math::float2<float> uu_dDD00 =
            ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
                                       FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
                                       FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
        const expansion_math::float2<float> uu_dDD11 =
            ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
                                       FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
                                       FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
        const expansion_math::float2<float> uu_dDD22 =
            ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
                                       FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
                                       FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const expansion_math::float2<float> FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
        const expansion_math::float2<float> FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
        const expansion_math::float2<float> FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
        const expansion_math::float2<float> FDPart3tmp7 = expansion_math::float2<float>(2.0,0) / FDPart3tmp4;
        const expansion_math::float2<float> FDPart3tmp2 = (expansion_math::float2<float>(1.0,0) / (FDPart3tmp1));
        const expansion_math::float2<float> FDPart3tmp5 = (expansion_math::float2<float>(1.0,0) / ((FDPart3tmp1) * (FDPart3tmp1)));
        const expansion_math::float2<float> vv(in_gfs[IDX4F(VVGF, i0, i1, i2, 0)], in_gfs[IDX4F(VVGF, i0, i1, i2, 1)]);
        const expansion_math::float2<float> resUU = -eta_damping * uu + vv;
        rhs_gfs[IDX4F(UUGF, i0, i1, i2, 0)] = resUU.value;
        rhs_gfs[IDX4F(UUGF, i0, i1, i2, 1)] = resUU.remainder;
        
        constexpr REAL FDPart1_Rational_1_8_ = 1.0 / 8.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_1_8 = expansion_math::split<float>(FDPart1_Rational_1_8_);
        constexpr REAL FDPart1_Rational_1_2_ = 1.0 / 2.0;
        constexpr expansion_math::float2<float> FDPart1_Rational_1_2 = expansion_math::split<float>(FDPart1_Rational_1_2_);
        const expansion_math::float2<float> ADD_times_AUU(
          auxevol_gfs[IDX4F(ADD_TIMES_AUUGF, i0, i1, i2, 0)], 
          auxevol_gfs[IDX4F(ADD_TIMES_AUUGF, i0, i1, i2, 1)]
        );
        const expansion_math::float2<float> psi_background(
          auxevol_gfs[IDX4F(PSI_BACKGROUNDGF, i0, i1, i2, 0)], 
          auxevol_gfs[IDX4F(PSI_BACKGROUNDGF, i0, i1, i2, 1)]
        );
        const expansion_math::float2<float> variable_wavespeed(
          auxevol_gfs[IDX4F(VARIABLE_WAVESPEEDGF, i0, i1, i2, 0)], 
          auxevol_gfs[IDX4F(VARIABLE_WAVESPEEDGF, i0, i1, i2, 1)]
        );

        const expansion_math::float2<float> resVV =
            ((variable_wavespeed) * (variable_wavespeed)) *
            (FDPart1_Rational_1_8 * ADD_times_AUU / expansion_math::pow_expansion(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
             FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
             uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
                       FDPart1_Rational_1_2 * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
                           (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
                            2.0f * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
                            FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
                           ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
             uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));
        rhs_gfs[IDX4F(VVGF, i0, i1, i2, 0)] = resVV.value;
        rhs_gfs[IDX4F(VVGF, i0, i1, i2, 1)] = resVV.remainder;
      } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
    } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
  } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
}

/*
 * GPU Kernel: rhs_eval_gpu.
 * GPU Kernel to evaluate RHS on the interior.
 */
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

  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  for (int i2 = tid2 + NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2) {
    for (int i1 = tid1 + NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1) {
      [[maybe_unused]] const REAL f1_of_xx1 = rfm_f1_of_xx1[i1];
      [[maybe_unused]] const REAL f1_of_xx1__D1 = rfm_f1_of_xx1__D1[i1];
      [[maybe_unused]] const REAL f1_of_xx1__DD11 = rfm_f1_of_xx1__DD11[i1];
      [[maybe_unused]] const REAL f4_of_xx1 = rfm_f4_of_xx1[i1];
      [[maybe_unused]] const REAL f4_of_xx1__D1 = rfm_f4_of_xx1__D1[i1];
      [[maybe_unused]] const REAL f4_of_xx1__DD11 = rfm_f4_of_xx1__DD11[i1];

      for (int i0 = tid0 + NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0) {
        [[maybe_unused]] const REAL f0_of_xx0 = rfm_f0_of_xx0[i0];
        [[maybe_unused]] const REAL f0_of_xx0__D0 = rfm_f0_of_xx0__D0[i0];
        [[maybe_unused]] const REAL f0_of_xx0__DD00 = rfm_f0_of_xx0__DD00[i0];
        [[maybe_unused]] const REAL f0_of_xx0__DDD000 = rfm_f0_of_xx0__DDD000[i0];
        [[maybe_unused]] const REAL f2_of_xx0 = rfm_f2_of_xx0[i0];
        [[maybe_unused]] const REAL f2_of_xx0__D0 = rfm_f2_of_xx0__D0[i0];
        [[maybe_unused]] const REAL f2_of_xx0__DD00 = rfm_f2_of_xx0__DD00[i0];
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL ADD_times_AUU = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
        const REAL psi_background = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
        const REAL uu_i2m5 = in_gfs[IDX4(UUGF, i0, i1, i2 - 5)];
        const REAL uu_i2m4 = in_gfs[IDX4(UUGF, i0, i1, i2 - 4)];
        const REAL uu_i2m3 = in_gfs[IDX4(UUGF, i0, i1, i2 - 3)];
        const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
        const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
        const REAL uu_i1m5 = in_gfs[IDX4(UUGF, i0, i1 - 5, i2)];
        const REAL uu_i1m4 = in_gfs[IDX4(UUGF, i0, i1 - 4, i2)];
        const REAL uu_i1m3 = in_gfs[IDX4(UUGF, i0, i1 - 3, i2)];
        const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
        const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
        const REAL uu_i0m5 = in_gfs[IDX4(UUGF, i0 - 5, i1, i2)];
        const REAL uu_i0m4 = in_gfs[IDX4(UUGF, i0 - 4, i1, i2)];
        const REAL uu_i0m3 = in_gfs[IDX4(UUGF, i0 - 3, i1, i2)];
        const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
        const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
        const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
        const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
        const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
        const REAL uu_i0p3 = in_gfs[IDX4(UUGF, i0 + 3, i1, i2)];
        const REAL uu_i0p4 = in_gfs[IDX4(UUGF, i0 + 4, i1, i2)];
        const REAL uu_i0p5 = in_gfs[IDX4(UUGF, i0 + 5, i1, i2)];
        const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
        const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
        const REAL uu_i1p3 = in_gfs[IDX4(UUGF, i0, i1 + 3, i2)];
        const REAL uu_i1p4 = in_gfs[IDX4(UUGF, i0, i1 + 4, i2)];
        const REAL uu_i1p5 = in_gfs[IDX4(UUGF, i0, i1 + 5, i2)];
        const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
        const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
        const REAL uu_i2p3 = in_gfs[IDX4(UUGF, i0, i1, i2 + 3)];
        const REAL uu_i2p4 = in_gfs[IDX4(UUGF, i0, i1, i2 + 4)];
        const REAL uu_i2p5 = in_gfs[IDX4(UUGF, i0, i1, i2 + 5)];
        const REAL variable_wavespeed = auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)];
        const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
        constexpr REAL FDPart1_Rational_5_6 = 5.0 / 6.0;
        constexpr REAL FDPart1_Rational_5_21 = 5.0 / 21.0;
        constexpr REAL FDPart1_Rational_5_84 = 5.0 / 84.0;
        constexpr REAL FDPart1_Rational_5_504 = 5.0 / 504.0;
        constexpr REAL FDPart1_Rational_1_1260 = 1.0 / 1260.0;
        constexpr REAL FDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
        constexpr REAL FDPart1_Rational_5_1008 = 5.0 / 1008.0;
        constexpr REAL FDPart1_Rational_1_3150 = 1.0 / 3150.0;
        constexpr REAL FDPart1_Rational_5_3 = 5.0 / 3.0;
        constexpr REAL FDPart1_Rational_5_126 = 5.0 / 126.0;
        const REAL FDPart1tmp0 = -FDPart1_Rational_5269_1800 * uu;
        const REAL uu_dD0 = invdxx0 * (FDPart1_Rational_1_1260 * (-uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_21 * (uu_i0m2 - uu_i0p2) +
                                       FDPart1_Rational_5_504 * (uu_i0m4 - uu_i0p4) + FDPart1_Rational_5_6 * (-uu_i0m1 + uu_i0p1) +
                                       FDPart1_Rational_5_84 * (-uu_i0m3 + uu_i0p3));
        const REAL uu_dD1 = invdxx1 * (FDPart1_Rational_1_1260 * (-uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_21 * (uu_i1m2 - uu_i1p2) +
                                       FDPart1_Rational_5_504 * (uu_i1m4 - uu_i1p4) + FDPart1_Rational_5_6 * (-uu_i1m1 + uu_i1p1) +
                                       FDPart1_Rational_5_84 * (-uu_i1m3 + uu_i1p3));
        const REAL uu_dDD00 =
            ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_3150 * (uu_i0m5 + uu_i0p5) + FDPart1_Rational_5_1008 * (-uu_i0m4 - uu_i0p4) +
                                       FDPart1_Rational_5_126 * (uu_i0m3 + uu_i0p3) + FDPart1_Rational_5_21 * (-uu_i0m2 - uu_i0p2) +
                                       FDPart1_Rational_5_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
        const REAL uu_dDD11 =
            ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_3150 * (uu_i1m5 + uu_i1p5) + FDPart1_Rational_5_1008 * (-uu_i1m4 - uu_i1p4) +
                                       FDPart1_Rational_5_126 * (uu_i1m3 + uu_i1p3) + FDPart1_Rational_5_21 * (-uu_i1m2 - uu_i1p2) +
                                       FDPart1_Rational_5_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
        const REAL uu_dDD22 =
            ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_3150 * (uu_i2m5 + uu_i2p5) + FDPart1_Rational_5_1008 * (-uu_i2m4 - uu_i2p4) +
                                       FDPart1_Rational_5_126 * (uu_i2m3 + uu_i2p3) + FDPart1_Rational_5_21 * (-uu_i2m2 - uu_i2p2) +
                                       FDPart1_Rational_5_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp4 = ((f2_of_xx0) * (f2_of_xx0));
        const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0)) + ((f4_of_xx1) * (f4_of_xx1));
        const REAL FDPart3tmp6 = FDPart3tmp4 / f0_of_xx0__D0;
        const REAL FDPart3tmp7 = 2 / FDPart3tmp4;
        const REAL FDPart3tmp2 = (1.0 / (FDPart3tmp1));
        const REAL FDPart3tmp5 = (1.0 / ((FDPart3tmp1) * (FDPart3tmp1)));
        rhs_gfs[IDX4(UUGF, i0, i1, i2)] = -eta_damping * uu + vv;
        rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
            ((variable_wavespeed) * (variable_wavespeed)) *
            ((1.0 / 8.0) * ADD_times_AUU / pow(psi_background + uu, 7) + FDPart3tmp2 * FDPart3tmp4 * uu_dDD00 / ((f0_of_xx0__D0) * (f0_of_xx0__D0)) +
             FDPart3tmp2 * uu_dDD11 + FDPart3tmp2 * f1_of_xx1__D1 * uu_dD1 / f1_of_xx1 -
             uu_dD0 * (-FDPart3tmp2 * FDPart3tmp6 / f0_of_xx0 - FDPart3tmp5 * FDPart3tmp6 * f0_of_xx0 +
                       (1.0 / 2.0) * FDPart3tmp5 * ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) *
                           (FDPart3tmp1 * FDPart3tmp7 * f0_of_xx0__D0 * f0_of_xx0__DD00 -
                            2 * FDPart3tmp1 * ((f0_of_xx0__D0) * (f0_of_xx0__D0)) * f2_of_xx0__D0 / ((f2_of_xx0) * (f2_of_xx0) * (f2_of_xx0)) +
                            FDPart3tmp7 * f0_of_xx0 * ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) /
                           ((f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0) * (f0_of_xx0__D0))) +
             uu_dDD22 / (((f0_of_xx0) * (f0_of_xx0)) * ((f1_of_xx1) * (f1_of_xx1))));

      } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += stride0)
    } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1 += stride1)
  } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2 += stride2)
}

__global__ static void compare(const float *restrict auxgfs_exp, const float *restrict ingfs_exp, const float *restrict rhsgfs_exp, 
  const REAL *restrict auxgfs, const REAL *restrict ingfs, const REAL *restrict rhsgfs) {
 
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;

  size_t vec_size = vecsize;
  size_t i = tid0 * vec_size;
  
  if(i < Ntot) {
    size_t j = 2U * i;
    REAL ref = auxgfs[i];
    REAL cmp = expansion_math::recast_sum<double>(expansion_math::float2<float>(auxgfs_exp[j], auxgfs_exp[j+1]));
    // if(std::fabs(ref) > 0) {
    //   REAL rel = std::fabs(1.0 - cmp / ref);
    //   if(rel > 1e-12) {
    //     printf("ausgfs failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
    //   }
    // }

    // ref = ingfs[i];
    // cmp = expansion_math::recast_sum<double>(expansion_math::float2<float>(ingfs_exp[j], ingfs_exp[j+1]));
    // if(std::fabs(ref) > 0) {
    //   REAL rel = std::fabs(1.0 - cmp / ref);
    //   if(rel > 1e-12) {
    //     printf("ingfs failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
    //   }
    // } else if(!(std::fabs(cmp) > 0)) {
    //   REAL rel = std::fabs(ref);
    //   if(rel > 1e-12) {
    //     printf("ingfs failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
    //   }
    // } else {
    //   REAL rel = std::fabs(cmp);
    //   if(rel > 1e-12) {
    //     printf("ingfs failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
    //   }
    // }

    ref = rhsgfs[i];
    cmp = expansion_math::recast_sum<double>(expansion_math::float2<float>(rhsgfs_exp[j], rhsgfs_exp[j+1]));
    if(std::fabs(ref) > 0 && std::fabs(cmp) > 0) {
      REAL rel = std::fabs(1.0 - cmp / ref);
      if(rel > 1e-12) {
        printf("rhsgfs failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
      }
    } else if(!(std::fabs(cmp) > 0)) {
      REAL rel = std::fabs(ref);
      if(rel > 1e-12) {
        printf("rhsgfs failure with cmp0 at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
      }
    } else {
      REAL rel = std::fabs(cmp);
      if(rel > 1e-12) {
        printf("rhsgfs failure with ref0 at %d with; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
      }
    }
  }
}

__global__ static void cpy_back(const float *restrict gf_in, REAL *restrict gf_out) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const uint tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const uint stride0 = blockDim.x * gridDim.x;

  for (uint i = tid0; i < Ntot; i += stride0) {
    uint j = 2U * i;
    gf_out[i] = expansion_math::recast_sum<double>(
      expansion_math::float2<float>(gf_in[j], gf_in[j+1])
    );
  }
}

__global__ static void decompose_gf(const REAL *restrict gf_in, float *restrict gf_out, int const Ntot) {

  // Kernel thread/stride setup
  const uint tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const uint stride0 = blockDim.x * gridDim.x;

  for (uint i = tid0; i < Ntot; i += stride0) {
    uint j = 2U * i;
    const expansion_math::float2<float> gf_exp_c = expansion_math::split<float>(gf_in[i]);
    gf_out[j] = gf_exp_c.value;
    gf_out[j+1] = gf_exp_c.remainder;
  }
}

__host__ static void setup_expansion_gfs( const params_struct *restrict params, 
  const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
  float * auxgfs_exp, float * ingfs_exp) {

  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  {
  int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_AUXEVOL_GFS;

  const size_t threads_in_x_dir = 96;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir / vecsize, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(auxevol_gfs, auxgfs_exp, Ntot);
  cudaCheckErrors(cudaKernel, "decompose auxgf failure");
  }

  {
  int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 96;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir / vecsize, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(in_gfs, ingfs_exp, Ntot);
  cudaCheckErrors(cudaKernel, "decompose ingf failure");
  }
  
  // Decompose REAL into floats and store in the new arrays


}

__host__ static void setup_expansion_rfm(const params_struct *restrict params, 
  float* exp_f0_of_xx0,
  float* exp_f0_of_xx0__D0,
  float* exp_f0_of_xx0__DD00,
  float* exp_f0_of_xx0__DDD000,
  float* exp_f1_of_xx1,
  float* exp_f1_of_xx1__D1,
  float* exp_f1_of_xx1__DD11,
  float* exp_f2_of_xx0,
  float* exp_f2_of_xx0__D0,
  float* exp_f2_of_xx0__DD00,
  float* exp_f4_of_xx1,
  float* exp_f4_of_xx1__D1,
  float* exp_f4_of_xx1__DD11,
  const REAL *restrict rfm_f0_of_xx0,
  const REAL *restrict rfm_f0_of_xx0__D0,
  const REAL *restrict rfm_f0_of_xx0__DD00,
  const REAL *restrict rfm_f0_of_xx0__DDD000,
  const REAL *restrict rfm_f1_of_xx1,
  const REAL *restrict rfm_f1_of_xx1__D1,
  const REAL *restrict rfm_f1_of_xx1__DD11,
  const REAL *restrict rfm_f2_of_xx0,
  const REAL *restrict rfm_f2_of_xx0__D0,
  const REAL *restrict rfm_f2_of_xx0__DD00,
  const REAL *restrict rfm_f4_of_xx1,
  const REAL *restrict rfm_f4_of_xx1__D1,
  const REAL *restrict rfm_f4_of_xx1__DD11
  ) {

  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 96;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir / vecsize, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  
  // Decompose REAL into floats and store in the new arrays
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f0_of_xx0, exp_f0_of_xx0, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f0_of_xx0 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f0_of_xx0__D0, exp_f0_of_xx0__D0, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f0_of_xx0__D0 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f0_of_xx0__DD00, exp_f0_of_xx0__DD00, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f0_of_xx0__DD00 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f0_of_xx0__DDD000, exp_f0_of_xx0__DDD000, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f0_of_xx0__DDD000 failure");

  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f1_of_xx1, exp_f1_of_xx1, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f1_of_xx1 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f1_of_xx1__D1, exp_f1_of_xx1__D1, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f1_of_xx1__D1 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f1_of_xx1__DD11, exp_f1_of_xx1__DD11, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f1_of_xx1__DD11 failure");

  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f2_of_xx0, exp_f2_of_xx0, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f2_of_xx0 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f2_of_xx0__D0, exp_f2_of_xx0__D0, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f2_of_xx0__D0 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f2_of_xx0__DD00, exp_f2_of_xx0__DD00, Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(cudaKernel, "decompose rfm_f2_of_xx0__DD00 failure");

  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f4_of_xx1, exp_f4_of_xx1, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f4_of_xx1 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f4_of_xx1__D1, exp_f4_of_xx1__D1, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f4_of_xx1__D1 failure");
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(rfm_f4_of_xx1__DD11, exp_f4_of_xx1__DD11, Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(cudaKernel, "decompose rfm_f4_of_xx1__DD11 failure");
}

/*
 * Set RHSs for hyperbolic relaxation equation.
 */
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
              const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
#include "set_CodeParameters.h"
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;
  
  // expansion GF arrays
  float* rhsgfs_exp;
  float* ingfs_exp;
  float* auxgfs_exp;
  cudaMalloc(&auxgfs_exp, sizeof(float) * NUM_AUXEVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&ingfs_exp, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rhsgfs_exp, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  setup_expansion_gfs(params, auxevol_gfs, in_gfs, auxgfs_exp, ingfs_exp);

  // expansion RFM arrays
  float* exp_f0_of_xx0;
  float* exp_f0_of_xx0__D0;
  float* exp_f0_of_xx0__DD00;
  float* exp_f0_of_xx0__DDD000;

  float* exp_f1_of_xx1;
  float* exp_f1_of_xx1__D1;
  float* exp_f1_of_xx1__DD11;

  float* exp_f2_of_xx0;
  float* exp_f2_of_xx0__D0;
  float* exp_f2_of_xx0__DD00;

  float* exp_f4_of_xx1;
  float* exp_f4_of_xx1__D1;
  float* exp_f4_of_xx1__DD11;
  
  cudaMalloc(&exp_f0_of_xx0, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f0_of_xx0__D0, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f0_of_xx0__DD00, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f0_of_xx0__DDD000, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");

  cudaMalloc(&exp_f1_of_xx1, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f1_of_xx1__D1, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f1_of_xx1__DD11, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");

  cudaMalloc(&exp_f2_of_xx0, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f2_of_xx0__D0, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f2_of_xx0__DD00, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");

  cudaMalloc(&exp_f4_of_xx1, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f4_of_xx1__D1, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&exp_f4_of_xx1__DD11, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");

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

setup_expansion_rfm(params,
  exp_f0_of_xx0,
  exp_f0_of_xx0__D0,
  exp_f0_of_xx0__DD00,
  exp_f0_of_xx0__DDD000,
  exp_f1_of_xx1,
  exp_f1_of_xx1__D1,
  exp_f1_of_xx1__DD11,
  exp_f2_of_xx0,
  exp_f2_of_xx0__D0,
  exp_f2_of_xx0__DD00,
  exp_f4_of_xx1,
  exp_f4_of_xx1__D1,
  exp_f4_of_xx1__DD11,
  rfm_f0_of_xx0,
  rfm_f0_of_xx0__D0,
  rfm_f0_of_xx0__DD00,
  rfm_f0_of_xx0__DDD000,
  rfm_f1_of_xx1,
  rfm_f1_of_xx1__D1,
  rfm_f1_of_xx1__DD11,
  rfm_f2_of_xx0,
  rfm_f2_of_xx0__D0,
  rfm_f2_of_xx0__DD00,
  rfm_f4_of_xx1,
  rfm_f4_of_xx1__D1,
  rfm_f4_of_xx1__DD11
);
  {
  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = NGHOSTS;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir, (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                       (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rhs_eval_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(
      rfm_f0_of_xx0, rfm_f0_of_xx0__D0, rfm_f0_of_xx0__DD00, rfm_f0_of_xx0__DDD000, rfm_f1_of_xx1, rfm_f1_of_xx1__D1, rfm_f1_of_xx1__DD11,
      rfm_f2_of_xx0, rfm_f2_of_xx0__D0, rfm_f2_of_xx0__DD00, rfm_f4_of_xx1, rfm_f4_of_xx1__D1, rfm_f4_of_xx1__DD11, auxevol_gfs, in_gfs, rhs_gfs,
      eta_damping);
  cudaCheckErrors(cudaKernel, "rhs_eval_gpu failure");
  expansion_math::float2<float> eta = expansion_math::split<float>(eta_damping);
  rhs_eval_expansion_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(
      exp_f0_of_xx0, exp_f0_of_xx0__D0, exp_f0_of_xx0__DD00, exp_f0_of_xx0__DDD000, exp_f1_of_xx1, exp_f1_of_xx1__D1, exp_f1_of_xx1__DD11,
      exp_f2_of_xx0, exp_f2_of_xx0__D0, exp_f2_of_xx0__DD00, exp_f4_of_xx1, exp_f4_of_xx1__D1, exp_f4_of_xx1__DD11, auxgfs_exp, ingfs_exp, rhsgfs_exp,
      eta);
  cudaCheckErrors(cudaKernel, "rhs_eval_expansion_gpu failure");
  }
  cudaDeviceSynchronize();
  {
  const size_t threads_in_x_dir = 96;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir / vecsize, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  compare<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(auxgfs_exp, ingfs_exp, rhsgfs_exp, auxevol_gfs, in_gfs, rhs_gfs);
  }
  // Free expansion GFs
  cudaFree(rhsgfs_exp);
  cudaFree(ingfs_exp);
  cudaFree(auxgfs_exp);

  // Free expansion RFM
  cudaFree(exp_f0_of_xx0);
  cudaFree(exp_f0_of_xx0__D0);
  cudaFree(exp_f0_of_xx0__DD00);
  cudaFree(exp_f0_of_xx0__DDD000);

  cudaFree(exp_f1_of_xx1);
  cudaFree(exp_f1_of_xx1__D1);
  cudaFree(exp_f1_of_xx1__DD11);

  cudaFree(exp_f2_of_xx0);
  cudaFree(exp_f2_of_xx0__D0);
  cudaFree(exp_f2_of_xx0__DD00);

  cudaFree(exp_f4_of_xx1);
  cudaFree(exp_f4_of_xx1__D1);
  cudaFree(exp_f4_of_xx1__DD11);
}
