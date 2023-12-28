#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
/*
 * Output minimum gridspacing ds_min on a Spherical numerical grid.
 */
void cfl_limited_timestep__rfm__Spherical(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                          bc_struct *restrict bcstruct) {
#include "../set_CodeParameters.h"

  REAL ds_min = 1e38;
  LOOP_NOOMP(i0, NGHOSTS, Nxx_plus_2NGHOSTS0 - NGHOSTS, i1, NGHOSTS, Nxx_plus_2NGHOSTS1 - NGHOSTS, i2, NGHOSTS, Nxx_plus_2NGHOSTS2 - NGHOSTS) {
    const REAL xx0 = xx[0][i0];
    const REAL xx1 = xx[1][i1];
    const REAL xx2 = xx[2][i2];
    REAL dsmin0, dsmin1, dsmin2;
    /*
     *  Original SymPy expressions:
     *  "[dsmin0 = dxx0]"
     *  "[dsmin1 = dxx1*xx0]"
     *  "[dsmin2 = dxx2*xx0*sin(xx1)]"
     */
    dsmin0 = dxx0;
    dsmin1 = dxx1 * xx0;
    dsmin2 = dxx2 * xx0 * sin(xx1);
    ds_min = MIN(ds_min, MIN(dsmin0, MIN(dsmin1, dsmin2)));
  }
  commondata->dt = MIN(commondata->dt, ds_min * commondata->CFL_FACTOR);
}
