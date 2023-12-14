#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
/*
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void rfm_precompute_malloc__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct) {
  [[maybe_unused]] int const & Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  [[maybe_unused]] int const & Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  [[maybe_unused]] int const & Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  cudaMalloc(&rfmstruct->f0_of_xx0, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&rfmstruct->f1_of_xx1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&rfmstruct->f1_of_xx1__D1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed")
  cudaMalloc(&rfmstruct->f1_of_xx1__DD11, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed")
}
