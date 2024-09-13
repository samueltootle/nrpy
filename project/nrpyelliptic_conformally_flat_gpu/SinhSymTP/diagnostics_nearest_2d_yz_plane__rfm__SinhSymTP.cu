#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
/*
 * Output diagnostic quantities at gridpoints closest to yz plane.
 */
void diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL *restrict xx[3], MoL_gridfunctions_struct *restrict gridfuncs) {
#include "../set_CodeParameters.h"

  // Unpack gridfuncs struct:
  [[maybe_unused]] const REAL *restrict y_n_gfs = gridfuncs->y_n_gfs;
  [[maybe_unused]] const REAL *restrict auxevol_gfs = gridfuncs->auxevol_gfs;
  [[maybe_unused]] const REAL *restrict diagnostic_output_gfs = gridfuncs->diagnostic_output_gfs;

  // 1D output
  char filename[256];
  sprintf(filename, "out2d-yz-n-%08d.txt", nn);
  FILE *outfile = (nn == 0) ? fopen(filename, "w") : fopen(filename, "a");
  if (!outfile) {
    fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
    exit(1);
  }

  // Output data in yz-plane in SinhSymTP coordinates.
  const int numpts_i0 = Nxx0, numpts_i1 = Nxx1, numpts_i2 = 2;
  int i0_pts[numpts_i0], i1_pts[numpts_i1], i2_pts[numpts_i2];
#pragma omp parallel for
  for (int i0 = NGHOSTS; i0 < Nxx0 + NGHOSTS; i0++)
    i0_pts[i0 - NGHOSTS] = i0;
#pragma omp parallel for
  for (int i1 = NGHOSTS; i1 < Nxx1 + NGHOSTS; i1++)
    i1_pts[i1 - NGHOSTS] = i1;
  i2_pts[0] = (int)(NGHOSTS + (1.0 / 4.0) * Nxx2 - 1.0 / 2.0);
  i2_pts[1] = (int)(NGHOSTS + (3.0 / 4.0) * Nxx2 - 1.0 / 2.0);
  // Main loop:
  LOOP_NOOMP(i0_pt, 0, numpts_i0, i1_pt, 0, numpts_i1, i2_pt, 0, numpts_i2) {
    const int i0 = i0_pts[i0_pt], i1 = i1_pts[i1_pt], i2 = i2_pts[i2_pt];
    const int idx3 = IDX3(i0, i1, i2);
    REAL xCart[3];
    xx_to_Cart(commondata, params, xx, i0, i1, i2, xCart);
    {
      const REAL numUU = y_n_gfs[IDX4pt(UUGF, idx3)];
      const REAL log10ResidualH = log10(fabs(diagnostic_output_gfs[IDX4pt(RESIDUAL_HGF, idx3)] + 1e-16));
      fprintf(outfile, "%.15e %.15e %.15e %.15e\n", xCart[1], xCart[2], numUU, log10ResidualH);
    }
  }

  fclose(outfile);
}
