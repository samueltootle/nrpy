#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"

__host__
void mallocHostgrid(const commondata_struct *restrict commondata, 
                    const params_struct *restrict params,
                    griddata_struct *restrict gd_host,
                    const griddata_struct *restrict gd_gpu) {
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Set up cell-centered Cartesian coordinate grid, centered at the origin.
  gd_host->xx[0] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  gd_host->xx[1] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  gd_host->xx[2] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);

}

__host__
void cpyDevicetoHost__grid(const commondata_struct *restrict commondata,
                          griddata_struct *restrict gd_host,
                          const griddata_struct *restrict gd_gpu) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    const params_struct *restrict params = &gd_gpu[grid].params;
    int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    mallocHostgrid(commondata, params, &gd_host[grid], gd_gpu);

    cudaMemcpy(gd_host[grid].xx[0], gd_gpu[grid].xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[1], gd_gpu[grid].xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[2], gd_gpu[grid].xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2, cudaMemcpyDeviceToHost);
  }
}

__host__
void cpyDevicetoHost__malloc_y_n_gfs(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs) {
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  cudaMallocHost((void**)&gridfuncs->y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_DIAG_YN);
  cudaCheckErrors(cudaMallocHost, "Malloc y_n diagnostic GFs failed.")
}

__host__
void cpyDevicetoHost__malloc_diag_gfs(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs) {
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  cudaMallocHost((void**)&gridfuncs->diagnostic_output_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_AUX_GFS);
  cudaCheckErrors(cudaMallocHost, "Malloc diagnostic GFs failed.")
}

__host__
void cpyDevicetoHost__gf(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        REAL * gf_host,
                        const REAL * gf_gpu,
                        const int host_GF_IDX,
                        const int gpu_GF_IDX) {
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  
  int streamid = (gpu_GF_IDX < nstreams) ? gpu_GF_IDX : int(gpu_GF_IDX / nstreams) - 1;
  int offset_gpu  = Nxx_plus_2NGHOSTS_tot * gpu_GF_IDX;
  int offset_host = Nxx_plus_2NGHOSTS_tot * host_GF_IDX;
  cudaMemcpyAsync(&gf_host[offset_host], 
                  &gf_gpu[offset_gpu], 
                  sizeof(REAL) * Nxx_plus_2NGHOSTS_tot, 
                  cudaMemcpyDeviceToHost, streams[streamid]);
  cudaCheckErrors(cudaMemcpyAsync, "Copy of gf data failed")
}

__host__
void cpyDevicetoHost__free_gfs(MoL_gridfunctions_struct *restrict gfs_host) {
  cudaFreeHost(gfs_host->y_n_gfs);
  cudaFreeHost(gfs_host->diagnostic_output_gfs);
}

__host__
void freeHostgrid(griddata_struct *restrict gd_host) {
  free(gd_host->xx[0]);
  free(gd_host->xx[1]);
  free(gd_host->xx[2]);
}