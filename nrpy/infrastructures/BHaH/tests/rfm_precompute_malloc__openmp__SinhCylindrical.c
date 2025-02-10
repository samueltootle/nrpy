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
  rfmstruct->f0_of_xx0 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  rfmstruct->f0_of_xx0__D0 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  rfmstruct->f0_of_xx0__DD00 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  rfmstruct->f0_of_xx0__DDD000 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  rfmstruct->f3_of_xx2 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS2);
  rfmstruct->f3_of_xx2__D2 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS2);
  rfmstruct->f3_of_xx2__DD22 = (REAL *)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS2);
} // END FUNCTION rfm_precompute_malloc__allocate

/**
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void rfm_precompute_malloc__rfm__SinhCylindrical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                 rfm_struct *restrict rfmstruct) {
  rfm_precompute_malloc__allocate(params, rfmstruct);
} // END FUNCTION rfm_precompute_malloc__rfm__SinhCylindrical
