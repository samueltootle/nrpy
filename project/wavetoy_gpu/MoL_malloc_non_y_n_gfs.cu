#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
/*
 * Method of Lines (MoL) for "RK4" method: Allocate memory for "non_y_n_gfs" gridfunctions
 * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 */
void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                            MoL_gridfunctions_struct *restrict gridfuncs) {
#include "set_CodeParameters.h"
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  
  cudaMallocManaged(&gridfuncs->y_nplus1_running_total_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaCheckErrors(gridfuncs->y_nplus1_running_total_gfs, "Malloc failed");
  cudaMemPrefetchAsync(gridfuncs->y_nplus1_running_total_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, 0);
  cudaCheckErrors(gridfuncs->y_nplus1_running_total_gfs, "prefetch failed");

  cudaMallocManaged(&gridfuncs->k_odd_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaCheckErrors(gridfuncs->k_odd_gfs, "Malloc failed");
  cudaMemPrefetchAsync(gridfuncs->k_odd_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, 0);
  cudaCheckErrors(gridfuncs->k_odd_gfs, "prefetch failed");

  cudaMallocManaged(&gridfuncs->k_even_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaCheckErrors(gridfuncs->k_even_gfs, "Malloc failed");
  cudaMemPrefetchAsync(gridfuncs->k_even_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, 0);
  cudaCheckErrors(gridfuncs->k_even_gfs, "prefetch failed");

  // gridfuncs->y_nplus1_running_total_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  // gridfuncs->k_odd_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  // gridfuncs->k_even_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  if (NUM_AUXEVOL_GFS > 0) {
    cudaMallocManaged(&gridfuncs->auxevol_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
    cudaCheckErrors(gridfuncs->auxevol_gfs, "Malloc failed");
    cudaMemPrefetchAsync(gridfuncs->auxevol_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, 0);
    cudaCheckErrors(gridfuncs->auxevol_gfs, "prefetch failed");
  }

  gridfuncs->diagnostic_output_gfs = gridfuncs->y_nplus1_running_total_gfs;
  gridfuncs->diagnostic_output_gfs2 = gridfuncs->k_odd_gfs;
}
