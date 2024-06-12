#include "../BHaH_defines.h"
/*
 * Compute Cartesian coordinates given local grid coordinate (xx0,xx1,xx2),   accounting for the origin of this grid being possibly off-center.
 */
__host__ void xx_to_Cart__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                         const int i0, const int i1, const int i2, REAL xCart[3]) {
  const REAL AMAX = params->AMAX;
  const REAL Cart_originx = params->Cart_originx;
  const REAL Cart_originy = params->Cart_originy;
  const REAL Cart_originz = params->Cart_originz;
  const REAL SINHWAA = params->SINHWAA;
  const REAL bScale = params->bScale;

  REAL xx0 = xx[0][i0];
  REAL xx1 = xx[1][i1];
  REAL xx2 = xx[2][i2];
  /*
   *  Original SymPy expressions:
   *  "[xCart[0] = AMAX*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))*sin(xx1)*cos(xx2)/(exp(1/SINHWAA) - exp(-1/SINHWAA)) + Cart_originx]"
   *  "[xCart[1] = AMAX*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))*sin(xx1)*sin(xx2)/(exp(1/SINHWAA) - exp(-1/SINHWAA)) + Cart_originy]"
   *  "[xCart[2] = Cart_originz + sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2)*cos(xx1)]"
   */
  {
    const REAL tmp0 = (1.0 / (SINHWAA));
    const REAL tmp1 = exp(tmp0) - exp(-tmp0);
    const REAL tmp3 = exp(tmp0 * xx0) - exp(-tmp0 * xx0);
    const REAL tmp4 = AMAX * tmp3 * sin(xx1) / tmp1;
    xCart[0] = Cart_originx + tmp4 * cos(xx2);
    xCart[1] = Cart_originy + tmp4 * sin(xx2);
    xCart[2] = Cart_originz + sqrt(((AMAX) * (AMAX)) * ((tmp3) * (tmp3)) / ((tmp1) * (tmp1)) + ((bScale) * (bScale))) * cos(xx1);
  }
}
