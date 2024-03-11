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

    mallocHostgrid(commondata, params, gd_host, gd_gpu);

    cudaMemcpy(gd_host[grid].xx[0], gd_gpu[grid].xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[1], gd_gpu[grid].xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd_host[grid].xx[2], gd_gpu[grid].xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2, cudaMemcpyDeviceToHost);
  }
}

__host__
void cpyDevicetoHost__gf(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        REAL *restrict gf_host,
                        const REAL *restrict gf_gpu,
                        const int GF_IDX) {
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  
  int streamid = (GF_IDX < nstreams) ? GF_IDX : int(GF_IDX / nstreams) - 1;
  int offset = Nxx_plus_2NGHOSTS_tot * GF_IDX;
  cudaMemcpyAsync(gf_host, &gf_gpu[offset], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot, cudaMemcpyDeviceToHost, streams[GF_IDX]);
}

// __host__
// void MoL_malloc_y_n_gfs__host(const commondata_struct *restrict commondata, const params_struct *restrict params,
//                         MoL_gridfunctions_struct *restrict gf_host) {
// #include "set_CodeParameters.h"
//   const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
//   const int NUM_DIAG_OUT
//   cudaMallocHost(&gf_host->y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * )
//   // cudaMalloc(&gridfuncs->y_n_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);

//   // gridfuncs->diagnostic_output_gfs = gridfuncs->y_nplus1_running_total_gfs;
//   // gridfuncs->diagnostic_output_gfs2 = gridfuncs->k_odd_gfs;
// }