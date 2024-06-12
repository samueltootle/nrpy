#include "BHaH_defines.h"
/*
 * Compute initial guess at a single point.
 */
__device__ __host__ void initial_guess_single_point(const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID) {
  *uu_ID = 0;
  *vv_ID = 0;
}
