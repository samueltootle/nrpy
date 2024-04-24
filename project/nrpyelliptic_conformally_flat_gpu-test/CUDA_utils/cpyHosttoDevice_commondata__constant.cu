#include "../BHaH_defines.h"
/*
 * Copy parameters to GPU __constant__.
 */
__host__ void cpyHosttoDevice_commondata__constant(const commondata_struct *restrict commondata) {
  cudaMemcpyToSymbol(d_commondata, commondata, sizeof(commondata_struct));
}
