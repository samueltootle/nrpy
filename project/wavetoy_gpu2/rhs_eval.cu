#include "BHaH_defines.h"
/*
 * Set RHSs for wave equation.
 */
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *restrict in_gfs,
              REAL *restrict rhs_gfs) {
#include "set_CodeParameters.h"
#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < NGHOSTS + Nxx2; i2++) {
    for (int i1 = NGHOSTS; i1 < NGHOSTS + Nxx1; i1++) {
      for (int i0 = NGHOSTS; i0 < NGHOSTS + Nxx0; i0++) {
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL uu_i2m2 = in_gfs[IDX4(UUGF, i0, i1, i2 - 2)];
        const REAL uu_i2m1 = in_gfs[IDX4(UUGF, i0, i1, i2 - 1)];
        const REAL uu_i1m2 = in_gfs[IDX4(UUGF, i0, i1 - 2, i2)];
        const REAL uu_i1m1 = in_gfs[IDX4(UUGF, i0, i1 - 1, i2)];
        const REAL uu_i0m2 = in_gfs[IDX4(UUGF, i0 - 2, i1, i2)];
        const REAL uu_i0m1 = in_gfs[IDX4(UUGF, i0 - 1, i1, i2)];
        const REAL uu = in_gfs[IDX4(UUGF, i0, i1, i2)];
        const REAL uu_i0p1 = in_gfs[IDX4(UUGF, i0 + 1, i1, i2)];
        const REAL uu_i0p2 = in_gfs[IDX4(UUGF, i0 + 2, i1, i2)];
        const REAL uu_i1p1 = in_gfs[IDX4(UUGF, i0, i1 + 1, i2)];
        const REAL uu_i1p2 = in_gfs[IDX4(UUGF, i0, i1 + 2, i2)];
        const REAL uu_i2p1 = in_gfs[IDX4(UUGF, i0, i1, i2 + 1)];
        const REAL uu_i2p2 = in_gfs[IDX4(UUGF, i0, i1, i2 + 2)];
        const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
        const REAL FDPart1_Rational_5_2 = 5.0 / 2.0;
        const REAL FDPart1_Rational_1_12 = 1.0 / 12.0;
        const REAL FDPart1_Rational_4_3 = 4.0 / 3.0;
        const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;
        const REAL uu_dDD00 =
            ((invdxx0) * (invdxx0)) * (FDPart1_Rational_1_12 * (-uu_i0m2 - uu_i0p2) + FDPart1_Rational_4_3 * (uu_i0m1 + uu_i0p1) + FDPart1tmp0);
        const REAL uu_dDD11 =
            ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_12 * (-uu_i1m2 - uu_i1p2) + FDPart1_Rational_4_3 * (uu_i1m1 + uu_i1p1) + FDPart1tmp0);
        const REAL uu_dDD22 =
            ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_12 * (-uu_i2m2 - uu_i2p2) + FDPart1_Rational_4_3 * (uu_i2m1 + uu_i2p1) + FDPart1tmp0);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = ((wavespeed) * (wavespeed));
        rhs_gfs[IDX4(UUGF, i0, i1, i2)] = vv;
        rhs_gfs[IDX4(VVGF, i0, i1, i2)] = FDPart3tmp0 * uu_dDD00 + FDPart3tmp0 * uu_dDD11 + FDPart3tmp0 * uu_dDD22;

      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    }   // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  }     // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}
