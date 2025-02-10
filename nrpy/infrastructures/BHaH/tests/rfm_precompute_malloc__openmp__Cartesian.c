#include "../BHaH_defines.h"
/**
 * GPU Kernel: rfm_precompute_malloc__allocate.
 * Kernel to allocate rfmstruct arrays.
 */
static void rfm_precompute_malloc__allocate(const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  // Temporary parameters
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
} // END FUNCTION rfm_precompute_malloc__allocate

/**
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void rfm_precompute_malloc__rfm__Cartesian(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct) {
  rfm_precompute_malloc__allocate(params, rfmstruct);
} // END FUNCTION rfm_precompute_malloc__rfm__Cartesian
