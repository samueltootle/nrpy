#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Free all memory within the griddata struct,
 * except perhaps non_y_n_gfs (e.g., after a regrid, in which non_y_n_gfs are freed first).
 */
void griddata_free(const commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host,
                   const bool enable_free_non_y_n_gfs) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    rfm_precompute_free(commondata, &griddata[grid].params, &griddata[grid].rfmstruct);
    cudaCheckErrors(free, "rfmFree failed");
    cudaFree(griddata[grid].bcstruct.inner_bc_array);
    cudaCheckErrors(free, "bcstruct inner_bc_arrayFree failed");
    for (int ng = 0; ng < NGHOSTS * 3; ng++) {
      cudaFree(griddata[grid].bcstruct.pure_outer_bc_array[ng]);
      cudaCheckErrors(free, "bcstruct pure_outer_bc_arrayFree failed");
    }

    MoL_free_memory_y_n_gfs(&griddata[grid].gridfuncs);
    cudaCheckErrors(free, "MoLFree failed");
    CUDA__free_host_gfs(&griddata_host[grid].gridfuncs);
    if (enable_free_non_y_n_gfs) {
      for (int i = 0; i < 3; i++) {
        cudaFree(griddata[grid].xx[i]);
        cudaCheckErrors(free, "griddata-xxFree failed");
        free(griddata_host[grid].xx[i]);
      }
    }

  } // END for(int grid=0;grid<commondata->NUMGRIDS;grid++)

  free(griddata);
  free(griddata_host);
}
