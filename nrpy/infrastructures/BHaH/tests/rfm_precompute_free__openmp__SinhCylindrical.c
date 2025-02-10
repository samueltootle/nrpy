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
  free(rfmstruct->f3_of_xx2);
  free(rfmstruct->f3_of_xx2__D2);
  free(rfmstruct->f3_of_xx2__DD22);
} // END FUNCTION rfm_precompute_free__deallocate

/**
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void rfm_precompute_free__rfm__SinhCylindrical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                               rfm_struct *restrict rfmstruct) {
  rfm_precompute_free__deallocate(rfmstruct);
} // END FUNCTION rfm_precompute_free__rfm__SinhCylindrical
