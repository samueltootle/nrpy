#include "../BHaH_defines.h"
/*
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void rfm_precompute_free__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                         rfm_struct *restrict rfmstruct) {
#include "../set_CodeParameters.h"
  cudaFree(rfmstruct->f0_of_xx0);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f0_of_xx0__D0);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f0_of_xx0__DD00);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f0_of_xx0__DDD000);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f1_of_xx1);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f1_of_xx1__D1);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f1_of_xx1__DD11);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f2_of_xx0);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f2_of_xx0__D0);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f2_of_xx0__DD00);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f4_of_xx1);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f4_of_xx1__D1);
  cudaCheckErrors(free, "cudaFree failed");
  cudaFree(rfmstruct->f4_of_xx1__DD11);
  cudaCheckErrors(free, "cudaFree failed");
}
