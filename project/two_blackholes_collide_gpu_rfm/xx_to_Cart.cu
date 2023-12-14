#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Compute Cartesian coordinates given local grid coordinate (xx0,xx1,xx2),   accounting for the origin of this grid being possibly off-center.
 */
__device__
void xx_to_Cart(REAL * xx[3], const int i0, const int i1, const int i2, REAL xCart[3]) {
  switch (d_params.CoordSystem_hash) {
  case SPHERICAL:
    xx_to_Cart__rfm__Spherical(xx, i0, i1, i2, xCart);
    break;
  default:
    printf("ERROR in xx_to_Cart(): CoordSystem hash = %d not #define'd!\n", d_params.CoordSystem_hash);
  }
}
