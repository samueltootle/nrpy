#include "../BHaH_defines.h"
/*
 * Free Host storage for diagnostics GFs.
 */
__host__ void CUDA__free_host_gfs(MoL_gridfunctions_struct *gridfuncs) {

  cudaFreeHost(gridfuncs->y_n_gfs);
  cudaCheckErrors(free, "Host-ynFree failed");
  cudaFreeHost(gridfuncs->diagnostic_output_gfs);
  cudaCheckErrors(free, "Host-non-ynFree failed");
}
