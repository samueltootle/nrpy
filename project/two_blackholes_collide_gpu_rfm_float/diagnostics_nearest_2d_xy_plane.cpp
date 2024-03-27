#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Output diagnostic quantities at gridpoints closest to xy plane.
 */
void diagnostics_nearest_2d_xy_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL * xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs) {
  switch (params->CoordSystem_hash) {
  case SPHERICAL:
    diagnostics_nearest_2d_xy_plane__rfm__Spherical(commondata, params, xx, gridfuncs);
    break;
  default:
    fprintf(stderr, "ERROR in diagnostics_nearest_2d_xy_plane(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
}
