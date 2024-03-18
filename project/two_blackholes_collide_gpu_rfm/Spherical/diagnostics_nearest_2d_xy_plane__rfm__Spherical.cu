#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Output diagnostic quantities at gridpoints closest to xy plane.
 */
void diagnostics_nearest_2d_xy_plane__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL * xx[3], MoL_gridfunctions_struct *restrict gridfuncs) {
#include "../set_CodeParameters.h"

  // Unpack gridfuncs struct:
  __attribute_maybe_unused__ const REAL *restrict y_n_gfs = gridfuncs->y_n_gfs;
  __attribute_maybe_unused__ const REAL *restrict auxevol_gfs = gridfuncs->auxevol_gfs;
  __attribute_maybe_unused__ const REAL *restrict diagnostic_output_gfs = gridfuncs->diagnostic_output_gfs;

  // 1D output
  char filename[256];
  sprintf(filename, "out2d-xy-conv_factor%.2f-t%08.4f.txt", convergence_factor, time);
  FILE *outfile = (nn == 0) ? fopen(filename, "w") : fopen(filename, "a");
  if (!outfile) {
    fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
    exit(1);
  }

  // Output data in xy-plane in Spherical coordinates.
  const int numpts_i0 = Nxx0, numpts_i1 = 1, numpts_i2 = Nxx2;
  int i0_pts[numpts_i0], i1_pts[numpts_i1], i2_pts[numpts_i2];
  
  const auto xx_to_cart = [&params] (auto const xx0, auto const xx1, auto const xx2, REAL * xCart) {
    const REAL tmp0 = xx0 * sin(xx1);
    xCart[0] = params->Cart_originx + tmp0 * cos(xx2);
    xCart[1] = params->Cart_originy + tmp0 * sin(xx2);
    xCart[2] = params->Cart_originz + xx0 * cos(xx1);
    return xCart;
  };
#pragma omp parallel for
  for (int i0 = NGHOSTS; i0 < Nxx0 + NGHOSTS; i0++)
    i0_pts[i0 - NGHOSTS] = i0;
  i1_pts[0] = (int)((1.0 / 2.0) * Nxx_plus_2NGHOSTS1);
#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < Nxx2 + NGHOSTS; i2++)
    i2_pts[i2 - NGHOSTS] = i2;
  // Main loop:
  LOOP_NOOMP(i0_pt, 0, numpts_i0, i1_pt, 0, numpts_i1, i2_pt, 0, numpts_i2) {
    const int i0 = i0_pts[i0_pt], i1 = i1_pts[i1_pt], i2 = i2_pts[i2_pt];
    const int idx3 = IDX3(i0, i1, i2);
    REAL xCart[3];
    {
      const REAL xx0 = xx[0][i0];
      const REAL xx1 = xx[1][i1];
      const REAL xx2 = xx[2][i2];
      xx_to_cart(xx0, xx1, xx2, xCart);
    }
    {
      const REAL HL = diagnostic_output_gfs[IDX4pt(HGF, idx3)];
      const REAL log10HL = log10(fabs(HL + 1e-16));
      
      const REAL M2L = diagnostic_output_gfs[IDX4pt(MSQUAREDGF, idx3)];
      const REAL log10sqrtM2L = log10(sqrt(M2L) + 1e-16);
      
      const REAL cfL = y_n_gfs[IDX4pt(DIAG_CFGF, idx3)];
      const REAL alphaL = y_n_gfs[IDX4pt(DIAG_ALPHAGF, idx3)];
      const REAL trKL = y_n_gfs[IDX4pt(DIAG_TRKGF, idx3)];
      fprintf(outfile, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", xCart[0], xCart[1], log10HL, log10sqrtM2L, cfL, alphaL, trKL);
    }
  }

  fclose(outfile);
}
