#include "../BHaH_defines.h"
/**
 * GPU Kernel: rfm_precompute_free__deallocate.
 * Kernel to deallocate rfmstruct arrays.
 */
static void rfm_precompute_free__deallocate(rfm_struct *restrict rfmstruct) {
  // Temporary parameters
  free(rfmstruct->f0_of_xx0);
  free(rfmstruct->f0_of_xx0__D0);
  free(rfmstruct->f0_of_xx0__DD00);
  free(rfmstruct->f0_of_xx0__DDD000);
  free(rfmstruct->f1_of_xx1);
  free(rfmstruct->f1_of_xx1__D1);
  free(rfmstruct->f1_of_xx1__DD11);
} // END FUNCTION rfm_precompute_free__deallocate

/**
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void rfm_precompute_free__rfm__RingHoleySinhSpherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                      rfm_struct *restrict rfmstruct) {
  rfm_precompute_free__deallocate(rfmstruct);
} // END FUNCTION rfm_precompute_free__rfm__RingHoleySinhSpherical
