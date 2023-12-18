#include "BHaH_gpu_defines.h"
// __constant__ REAL wavespeed; 
// __constant__ REAL FDPart1_Rational_5_2; 
// __constant__ REAL FDPart1_Rational_1_12;
// __constant__ REAL FDPart1_Rational_4_3;
__constant__ params_struct d_params;
__constant__ int8_t d_evol_gf_parity[24];
__constant__ REAL FDPart1_Rational_2_3;
__constant__ REAL FDPart1_Rational_1_12;
cudaStream_t stream1, stream2, stream3;