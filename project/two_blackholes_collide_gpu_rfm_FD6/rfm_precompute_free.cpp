#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  switch (params->CoordSystem_hash) {
  case SPHERICAL:
    rfm_precompute_free__rfm__Spherical(commondata, params, rfmstruct);
    break;
  default:
    fprintf(stderr, "ERROR in rfm_precompute_free(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
}
