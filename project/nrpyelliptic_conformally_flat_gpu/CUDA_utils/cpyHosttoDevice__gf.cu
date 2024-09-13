#include "../BHaH_defines.h"
/*
 * Asynchronously copying a grid function from host to device.
 */
__host__ size_t cpyHosttoDevice__gf(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *gf_host,
                                    REAL *gf_gpu, const int host_GF_IDX, const int gpu_GF_IDX) {

  int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  size_t streamid = (params->grid_idx + gpu_GF_IDX) % nstreams;
  int offset_gpu = Nxx_plus_2NGHOSTS_tot * gpu_GF_IDX;
  int offset_host = Nxx_plus_2NGHOSTS_tot * host_GF_IDX;
  cudaMemcpyAsync(&gf_gpu[offset_host], &gf_host[offset_gpu], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot, cudaMemcpyHostToDevice, streams[streamid]);
  cudaCheckErrors(cudaMemcpyAsync, "Copy of gf data failed");
  return streamid;
}
