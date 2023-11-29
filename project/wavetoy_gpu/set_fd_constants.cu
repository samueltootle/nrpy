#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
__host__
void set_fd_constants() {
    const REAL h_FDPart1_Rational_5_2 = 5.0 / 2.0;
    const REAL h_FDPart1_Rational_1_12 = 1.0 / 12.0;
    const REAL h_FDPart1_Rational_4_3 = 4.0 / 3.0;

    cudaMemcpyToSymbol(FDPart1_Rational_5_2, &h_FDPart1_Rational_5_2, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_1_12, &h_FDPart1_Rational_1_12, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_4_3, &h_FDPart1_Rational_4_3, sizeof(REAL));
//     const REAL h_FDPart1_Rational_2_3 = 2.0/3.0;
//     const REAL h_FDPart1_Rational_1_12 = 1.0/12.0;
//     const REAL h_FDPart1_Rational_5_2 = 5.0/2.0;
//     const REAL h_FDPart1_Rational_4_3 = 4.0/3.0;
//     const REAL h_FDPart3_0 = (1.0/(sinh_width));
//     const REAL h_FDPart3_6 = (1.0/((domain_size)*(domain_size)));
//     const REAL h_FDPart3_7 = exp(h_FDPart3_0) - exp(-h_FDPart3_0);
//     const REAL h_FDPart3_8 = h_FDPart3_6*((h_FDPart3_7)*(h_FDPart3_7));
//     const REAL h_FDPart3_13 = 2*h_FDPart3_0;
//     const REAL h_FDPart3_14 = exp(h_FDPart3_13) - 1;
//     const REAL h_FDPart3_18 = ((h_FDPart3_7)*(h_FDPart3_7)*(h_FDPart3_7)*(h_FDPart3_7));
//     const REAL h_FDPart3_22 = (1.0/((h_FDPart3_14)*(h_FDPart3_14)));
//     cudaMemcpyToSymbol(FDPart1_Rational_2_3, &h_FDPart1_Rational_2_3, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart1_Rational_1_12, &h_FDPart1_Rational_1_12, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart1_Rational_5_2, &h_FDPart1_Rational_5_2, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart1_Rational_4_3, &h_FDPart1_Rational_4_3, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_0, &h_FDPart3_0, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_6, &h_FDPart3_6, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_7, &h_FDPart3_7, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_8, &h_FDPart3_8, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_13, &h_FDPart3_13, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_14, &h_FDPart3_14, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_18, &h_FDPart3_18, sizeof(REAL));
//     cudaMemcpyToSymbol(FDPart3_22, &h_FDPart3_22, sizeof(REAL));
}