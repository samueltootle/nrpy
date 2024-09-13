#include "../BHaH_defines.h"
/*
 * Allocate Host storage for diagnostics GFs.
 */
__host__ void CUDA__malloc_host_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                    MoL_gridfunctions_struct *restrict gridfuncs) {

  int const &Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const &Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const &Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  cudaMallocHost((void **)&gridfuncs->y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_EVOL_GFS);
  cudaCheckErrors(cudaMallocHost, "Malloc y_n diagnostic GFs failed.");

  cudaMallocHost((void **)&gridfuncs->diagnostic_output_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_AUX_GFS);
  cudaCheckErrors(cudaMallocHost, "Malloc diagnostic GFs failed.")
}
