#include "../BHaH_defines.h"
/*
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void rfm_precompute_malloc__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct) {
#include "../set_CodeParameters.h"
  cudaMalloc(&rfmstruct->f0_of_xx0, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f0_of_xx0__D0, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f0_of_xx0__DD00, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f0_of_xx0__DDD000, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f1_of_xx1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f1_of_xx1__D1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f1_of_xx1__DD11, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f2_of_xx0, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f2_of_xx0__D0, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f2_of_xx0__DD00, sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f4_of_xx1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f4_of_xx1__D1, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&rfmstruct->f4_of_xx1__DD11, sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  cudaCheckErrors(malloc, "Malloc failed");
}
