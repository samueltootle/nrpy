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
}

__host__
void set_param_constants(params_struct *restrict params) {
    cudaMemcpyToSymbol(d_params, params, sizeof(params_struct));
}

__host__
void set_commondata_constants(commondata_struct *restrict commondata) {
    cudaMemcpyToSymbol(wavespeed, &commondata->wavespeed, sizeof(REAL));
}