#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Given Cartesian point (x,y,z), this function outputs the corresponding
 * (xx0,xx1,xx2) and the "closest" (i0,i1,i2) for the given grid
 */
__device__
void Cart_to_xx_and_nearest_i0i1i2(const commondata_struct *restrict commondata, const REAL xCart[3],
                                   REAL xx[3], int Cart_to_i0i1i2[3]) {
  switch (d_params.CoordSystem_hash) {
  case SPHERICAL:
    Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical(commondata, xCart, xx, Cart_to_i0i1i2);
    break;
  default:
    printf("ERROR in Cart_to_xx_and_nearest_i0i1i2(): CoordSystem hash = %d not #define'd!\n", d_params.CoordSystem_hash);
    // exit(1);
  }
}