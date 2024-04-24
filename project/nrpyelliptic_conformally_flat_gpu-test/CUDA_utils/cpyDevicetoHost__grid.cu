#include "../BHaH_defines.h"
/*
 * GPU Kernel: mallocHostgrid.
 * Allocate griddata_struct[grid].xx for host.
 */
__host__ static void mallocHostgrid(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                    griddata_struct *restrict gd_host, const griddata_struct *restrict gd_gpu) {

  int const &Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const &Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const &Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Set up cell-centered Cartesian coordinate grid, centered at the origin.
  gd_host->xx[0] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  gd_host->xx[1] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  gd_host->xx[2] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);
}

/*
 * Copy griddata_struct[grid].xx from GPU to host.
 */
__host__ void cpyDevicetoHost__grid(const commondata_struct *restrict commondata, griddata_struct *restrict gd_host,
                                    const griddata_struct *restrict gd_gpu) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    const params_struct *restrict params = &gd_gpu[grid].params;
    int const &Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const &Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const &Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    mallocHostgrid(commondata, params, &gd_host[grid], &gd_gpu[grid]);
    cudaMemcpy(gd_host[grid].xx[0], gd_gpu[grid].xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[1], gd_gpu[grid].xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[2], gd_gpu[grid].xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2, cudaMemcpyDeviceToHost);
  }
}
