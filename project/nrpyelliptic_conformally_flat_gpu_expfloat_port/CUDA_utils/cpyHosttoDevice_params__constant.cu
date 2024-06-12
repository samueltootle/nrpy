#include "../BHaH_defines.h"
/*
 * Copy parameters to GPU __constant__.
 */
__host__ void cpyHosttoDevice_params__constant(const params_struct *restrict params) { cudaMemcpyToSymbol(d_params, params, sizeof(params_struct)); }
