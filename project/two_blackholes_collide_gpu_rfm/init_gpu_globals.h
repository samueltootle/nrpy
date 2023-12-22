#include "BHaH_gpu_defines.h"
// __constant__ REAL wavespeed; 
__constant__ params_struct d_params;
__constant__ int8_t d_evol_gf_parity[24];
__constant__ REAL FDPart1_Rational_1_2;
__constant__ REAL FDPart1_Rational_1_4;
__constant__ REAL FDPart1_Rational_1_12;
__constant__ REAL FDPart1_Rational_1_18;
__constant__ REAL FDPart1_Rational_1_144;
__constant__ REAL FDPart1_Rational_2_3;
__constant__ REAL FDPart1_Rational_3_2;
__constant__ REAL FDPart1_Rational_4_3;
__constant__ REAL FDPart1_Rational_4_9;
__constant__ REAL FDPart1_Rational_5_2;
__constant__ REAL FDPart1_Rational_5_6;
__constant__ REAL d_gridfunctions_wavespeed[NUM_EVOL_GFS];
__constant__ REAL d_gridfunctions_f_infinity[NUM_EVOL_GFS];
cudaStream_t stream1, stream2, stream3;