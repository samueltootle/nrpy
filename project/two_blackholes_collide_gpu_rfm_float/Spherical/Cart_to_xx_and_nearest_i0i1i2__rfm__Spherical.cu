#include "../BHaH_defines.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Given Cartesian point (x,y,z), this function outputs the corresponding
 * (xx0,xx1,xx2) and the "closest" (i0,i1,i2) for the given grid
 */
__device__
void Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical(const commondata_struct *restrict commondata,
                                                   const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]) {

  // See comments for description on how coordinates are computed relative to the local grid center.
  const REAL Cartx = xCart[0] - d_params.Cart_originx;
  const REAL Carty = xCart[1] - d_params.Cart_originy;
  const REAL Cartz = xCart[2] - d_params.Cart_originz;
  /*
   *  Original SymPy expressions:
   *  "[xx[0] = sqrt(Cartx**2 + Carty**2 + Cartz**2)]"
   *  "[xx[1] = acos(Cartz/sqrt(Cartx**2 + Carty**2 + Cartz**2))]"
   *  "[xx[2] = atan2(Carty, Cartx)]"
   */
  const REAL tmp0 = sqrtf(((Cartx) * (Cartx)) + ((Carty) * (Carty)) + ((Cartz) * (Cartz)));
  xx[0] = tmp0;
  xx[1] = acosf(Cartz / tmp0);
  xx[2] = atan2f(Carty, Cartx);

  // Then find the nearest index (i0,i1,i2) on underlying grid to (x,y,z)
  Cart_to_i0i1i2[0] = (int)((xx[0] - (0)) / d_params.dxx0 + 0.5 + NGHOSTS - 0.5);     // Account for (int) typecast rounding down
  Cart_to_i0i1i2[1] = (int)((xx[1] - (0)) / d_params.dxx1 + 0.5 + NGHOSTS - 0.5);     // Account for (int) typecast rounding down
  Cart_to_i0i1i2[2] = (int)((xx[2] - (-M_PI)) / d_params.dxx2 + 0.5 + NGHOSTS - 0.5); // Account for (int) typecast rounding down
}
