#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
__host__
void set_fd_constants() {
    const REAL h_FDPart1_Rational_2_3 = 2.0 / 3.0;
    const REAL h_FDPart1_Rational_1_12 = 1. / 12.;

    cudaMemcpyToSymbol(FDPart1_Rational_1_12, &h_FDPart1_Rational_1_12, sizeof(REAL));
    cudaMemcpyToSymbol(FDPart1_Rational_2_3, &h_FDPart1_Rational_2_3, sizeof(REAL));
}

__host__
void set_param_constants(params_struct *restrict params) {
    cudaMemcpyToSymbol(d_params, params, sizeof(params_struct));
}