#include "../BHaH_defines.h"
/*
 * Compute Cartesian coordinates given local grid coordinate (xx0,xx1,xx2),   accounting for the origin of this grid being possibly off-center.
 */
void xx_to_Cart__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                const int i0, const int i1, const int i2, REAL xCart[3]) {
#include "../set_CodeParameters.h"

  REAL xx0 = xx[0][i0];
  REAL xx1 = xx[1][i1];
  REAL xx2 = xx[2][i2];
  /*
   *  Original SymPy expressions:
   *  "[xCart[0] = Cart_originx + xx0*sin(xx1)*cos(xx2)]"
   *  "[xCart[1] = Cart_originy + xx0*sin(xx1)*sin(xx2)]"
   *  "[xCart[2] = Cart_originz + xx0*cos(xx1)]"
   */
  {
    const REAL tmp0 = xx0 * sin(xx1);
    xCart[0] = Cart_originx + tmp0 * cos(xx2);
    xCart[1] = Cart_originy + tmp0 * sin(xx2);
    xCart[2] = Cart_originz + xx0 * cos(xx1);
  }
}
