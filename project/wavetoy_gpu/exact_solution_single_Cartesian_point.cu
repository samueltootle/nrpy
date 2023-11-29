#include "BHaH_defines.h"
/*
 * Exact solution at a single Cartesian point (x, y, z) = (xCart0, xCart1, xCart2).
 */
__device__ __host__
void exact_solution_single_Cartesian_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xCart0,
                                           const REAL xCart1, const REAL xCart2, REAL *restrict exact_soln_UUGF, REAL *restrict exact_soln_VVGF) {
#include "set_CodeParameters.h"
  const REAL tmp1 = sqrt(((xCart0) * (xCart0)) + ((xCart1) * (xCart1)) + ((xCart2) * (xCart2)));
  const REAL tmp5 = (1.0 / ((sigma) * (sigma)));
  const REAL tmp2 = time * wavespeed + tmp1;
  const REAL tmp3 = (1.0 / (tmp1));
  const REAL tmp8 = -time * wavespeed + tmp1;
  const REAL tmp7 = tmp3 * exp(-1.0 / 2.0 * ((tmp2) * (tmp2)) * tmp5);
  const REAL tmp10 = tmp3 * exp(-1.0 / 2.0 * tmp5 * ((tmp8) * (tmp8)));
  *exact_soln_UUGF = tmp10 * tmp8 + tmp2 * tmp7 + 2;
  *exact_soln_VVGF =
      tmp10 * tmp5 * ((tmp8) * (tmp8)) * wavespeed - tmp10 * wavespeed - ((tmp2) * (tmp2)) * tmp5 * tmp7 * wavespeed + tmp7 * wavespeed;
}
