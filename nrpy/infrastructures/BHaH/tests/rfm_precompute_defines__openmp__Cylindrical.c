#include "../BHaH_defines.h"
/**
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0_host.
 * Host Kernel to precompute metric quantity f0_of_xx0.
 */
static void rfm_precompute_defines__f0_of_xx0_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct, const REAL *restrict x0) {
  // Temporary parameters
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;

  for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0 += 1) {
    const REAL xx0 = x0[i0];
    rfmstruct->f0_of_xx0[i0] = xx0;
  }
} // END FUNCTION rfm_precompute_defines__f0_of_xx0_host

/**
 * rfm_precompute_defines: reference metric precomputed lookup arrays: defines
 */
void rfm_precompute_defines__rfm__Cylindrical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                              rfm_struct *restrict rfmstruct, REAL *restrict xx[3]) {
  MAYBE_UNUSED const REAL *restrict x0 = xx[0];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  MAYBE_UNUSED const REAL *restrict x1 = xx[1];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  MAYBE_UNUSED const REAL *restrict x2 = xx[2];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  {
    rfm_precompute_defines__f0_of_xx0_host(params, rfmstruct, x0);
  }
} // END FUNCTION rfm_precompute_defines__rfm__Cylindrical
