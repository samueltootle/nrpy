#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Output minimum gridspacing ds_min on a SinhSymTP numerical grid.
 */
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    cfl_limited_timestep__rfm__SinhSymTP(commondata, params, xx);
    break;
  default:
    fprintf(stderr, "ERROR in cfl_limited_timestep(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
}
