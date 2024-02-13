#include "../BHaH_defines.h"
/*
 * Enforce det(gammabar) = det(gammahat) constraint. Required for strong hyperbolicity.
 */
void enforce_detgammabar_equals_detgammahat__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                            const rfm_struct *restrict rfmstruct, REAL *restrict in_gfs) {
#include "../set_CodeParameters.h"
#pragma omp parallel for
  for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
    for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
      const REAL f1_of_xx1 = rfmstruct->f1_of_xx1[i1];
      const REAL f1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1[i1];
      const REAL f1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11[i1];

      for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
        const REAL f0_of_xx0 = rfmstruct->f0_of_xx0[i0];
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = hDD00 + 1;
        const REAL FDPart3tmp1 = ((f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp2 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp4 = ((f0_of_xx0) * (f0_of_xx0));
        const REAL FDPart3tmp5 = FDPart3tmp4 * hDD11 + FDPart3tmp4;
        const REAL FDPart3tmp6 = FDPart3tmp2 * FDPart3tmp4;
        const REAL FDPart3tmp7 = FDPart3tmp6 * hDD22 + FDPart3tmp6;
        const REAL FDPart3tmp8 = cbrt(fabs(FDPart3tmp1 * FDPart3tmp2) /
                                      (-FDPart3tmp0 * FDPart3tmp1 * FDPart3tmp2 * ((hDD12) * (hDD12)) + FDPart3tmp0 * FDPart3tmp5 * FDPart3tmp7 +
                                       2 * FDPart3tmp1 * FDPart3tmp2 * hDD01 * hDD02 * hDD12 - FDPart3tmp4 * FDPart3tmp7 * ((hDD01) * (hDD01)) -
                                       FDPart3tmp5 * FDPart3tmp6 * ((hDD02) * (hDD02))));
        in_gfs[IDX4(HDD00GF, i0, i1, i2)] = FDPart3tmp0 * FDPart3tmp8 - 1;
        in_gfs[IDX4(HDD01GF, i0, i1, i2)] = FDPart3tmp8 * hDD01;
        in_gfs[IDX4(HDD02GF, i0, i1, i2)] = FDPart3tmp8 * hDD02;
        in_gfs[IDX4(HDD11GF, i0, i1, i2)] = FDPart3tmp8 * (hDD11 + 1) - 1;
        in_gfs[IDX4(HDD12GF, i0, i1, i2)] = FDPart3tmp8 * hDD12;
        in_gfs[IDX4(HDD22GF, i0, i1, i2)] = FDPart3tmp8 * (hDD22 + 1) - 1;

      } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
    }   // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
  }     // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
          
  // for(int i = 0; i < NUM_EVOL_GFS; ++i)
  //   print_var(in_gfs, IDX4(i, 34, 18 , 18));
  // printf("**************************_enforce\n");
}
