#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Output minimum gridspacing ds_min on a SinhSymTP numerical grid.
 */
void numerical_grid_params_Nxx_dxx_xx(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3], const int Nx[3],
                                      const bool grid_is_resized) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(commondata, params, xx, Nx, grid_is_resized);
    break;
  default:
    fprintf(stderr, "ERROR in numerical_grid_params_Nxx_dxx_xx(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
}
