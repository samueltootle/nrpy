#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
/*
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void rfm_precompute_malloc__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct) {
  int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
  cudaMemcpy(&Nxx_plus_2NGHOSTS0, &params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS1, &params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS2, &params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")

  rfmstruct->f0_of_xx0 = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  rfmstruct->f1_of_xx1 = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  rfmstruct->f1_of_xx1__D1 = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  rfmstruct->f1_of_xx1__DD11 = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
}
