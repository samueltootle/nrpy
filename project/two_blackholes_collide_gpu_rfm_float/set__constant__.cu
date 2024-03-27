#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
__host__
void set_fd_constants() {
    const REAL h_FDPart1_Rational_1_2 = 1.0F / 2.0F;
    const REAL h_FDPart1_Rational_1_4 = 1.0F / 4.0F;
    const REAL h_FDPart1_Rational_1_12 = 1.0F / 12.0F;
    const REAL h_FDPart1_Rational_1_18 = 1.0F / 18.0F;
    const REAL h_FDPart1_Rational_1_144 = 1.0F / 144.0F;
    const REAL h_FDPart1_Rational_2_3 = 2.0F / 3.0F;    
    const REAL h_FDPart1_Rational_3_2 = 3.0F / 2.0F;
    const REAL h_FDPart1_Rational_4_3 = 4.0F / 3.0F;
    const REAL h_FDPart1_Rational_4_9 = 4.0F / 9.0F;
    const REAL h_FDPart1_Rational_5_2 = 5.0F / 2.0F;
    const REAL h_FDPart1_Rational_5_6 = 5.0F / 6.0F;

    cudaMemcpyToSymbol(FDPart1_Rational_1_2,&h_FDPart1_Rational_1_2, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_1_4,&h_FDPart1_Rational_1_4, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_1_12, &h_FDPart1_Rational_1_12, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_1_18, &h_FDPart1_Rational_1_18, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_1_144,&h_FDPart1_Rational_1_144, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_2_3,  &h_FDPart1_Rational_2_3, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_3_2,&h_FDPart1_Rational_3_2, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_4_3,  &h_FDPart1_Rational_4_3, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_4_9,  &h_FDPart1_Rational_4_9, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_5_2,  &h_FDPart1_Rational_5_2, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_5_6,&h_FDPart1_Rational_5_6, sizeof(REAL));
}

__host__
void set_param_constants(const params_struct *restrict params) {
    cudaMemcpyToSymbol(d_params, params, sizeof(params_struct));
}