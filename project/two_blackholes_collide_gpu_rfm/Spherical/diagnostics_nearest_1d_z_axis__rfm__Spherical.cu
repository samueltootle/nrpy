#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"

typedef struct {
  REAL xCart_axis;
  REAL log10HL;
  REAL log10sqrtM2L;
  REAL cfL;
  REAL alphaL;
  REAL trKL;
} data_point_1d_struct;

// qsort() comparison function for 1D output.
static int compare(const void *a, const void *b) {
  REAL l = ((data_point_1d_struct *)a)->xCart_axis;
  REAL r = ((data_point_1d_struct *)b)->xCart_axis;
  return (l > r) - (l < r);
}

/*
 * Output diagnostic quantities at gridpoints closest to z axis.
 */
void diagnostics_nearest_1d_z_axis__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, REAL * xx[3],
                                                   MoL_gridfunctions_struct *restrict gridfuncs) {
#include "../set_CodeParameters.h"

  // Unpack gridfuncs struct:
  __attribute_maybe_unused__ const REAL *restrict y_n_gfs = gridfuncs->y_n_gfs;
  __attribute_maybe_unused__ const REAL *restrict auxevol_gfs = gridfuncs->auxevol_gfs;
  __attribute_maybe_unused__ const REAL *restrict diagnostic_output_gfs = gridfuncs->diagnostic_output_gfs;
  __attribute_maybe_unused__ const REAL *restrict k_odd_gfs = gridfuncs->k_odd_gfs;

  // 1D output
  char filename[256];
  sprintf(filename, "out1d-z-conv_factor%.2f-t%08.2f.txt", convergence_factor, time);
  FILE *outfile = (nn == 0) ? fopen(filename, "w") : fopen(filename, "a");
  if (!outfile) {
    fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
    exit(1);
  }

  // Output along z-axis in Spherical coordinates.
  const int numpts_i0 = Nxx_plus_2NGHOSTS0, numpts_i1 = 2, numpts_i2 = 1;
  int i0_pts[numpts_i0], i1_pts[numpts_i1], i2_pts[numpts_i2];

  data_point_1d_struct data_points[numpts_i0 * numpts_i1 * numpts_i2];
  int data_index = 0;

  const auto get_diagnostics = [](auto index, const REAL *restrict g_data) {
    REAL h_data;
    cudaMemcpy(&h_data, &g_data[index], sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory error");
    return h_data;
  };
  const auto xx_to_cart = [&params] (auto const xx0, auto const xx1, auto const xx2, REAL * xCart) {
    const REAL tmp0 = xx0 * sin(xx1);
    xCart[0] = params->Cart_originx + tmp0 * cos(xx2);
    xCart[1] = params->Cart_originy + tmp0 * sin(xx2);
    xCart[2] = params->Cart_originz + xx0 * cos(xx1);
    return xCart;
  };
#pragma omp parallel for
  // for (int i0 = NGHOSTS; i0 < Nxx0 + NGHOSTS; i0++)
  for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
    i0_pts[i0] = i0;
  i1_pts[0] = (int)(NGHOSTS);
  i1_pts[1] = (int)(-NGHOSTS + Nxx_plus_2NGHOSTS1 - 1);
  i2_pts[0] = (int)(NGHOSTS);
  // Main loop:
  LOOP_NOOMP(i0_pt, 0, numpts_i0, i1_pt, 0, numpts_i1, i2_pt, 0, numpts_i2) {
    const int i0 = i0_pts[i0_pt], i1 = i1_pts[i1_pt], i2 = i2_pts[i2_pt];
    const int idx3 = IDX3(i0, i1, i2);
    REAL xCart[3];
    {
      const REAL xx0 = get_diagnostics(i0, xx[0]);
      const REAL xx1 = get_diagnostics(i1, xx[1]);
      const REAL xx2 = get_diagnostics(i2, xx[2]);
      xx_to_cart(xx0, xx1, xx2, xCart);
    }

    {
      data_point_1d_struct dp1d;
      dp1d.xCart_axis = xCart[2];
      const REAL HL = get_diagnostics(IDX4pt(HGF, idx3), diagnostic_output_gfs);
      dp1d.log10HL = log10(fabs(HL + 1e-16));
      const REAL M2L = get_diagnostics(IDX4pt(MSQUAREDGF, idx3), diagnostic_output_gfs);
      dp1d.log10sqrtM2L = log10(sqrt(M2L) + 1e-16);
      dp1d.cfL = get_diagnostics(IDX4pt(CFGF, idx3), y_n_gfs);
      dp1d.alphaL = get_diagnostics(IDX4pt(ALPHAGF, idx3), y_n_gfs);
      dp1d.trKL = get_diagnostics(IDX4pt(TRKGF, idx3), y_n_gfs);
      data_points[data_index] = dp1d;
      data_index++;
    }
  }

  qsort(data_points, data_index, sizeof(data_point_1d_struct), compare);

  for (int i = 0; i < data_index; i++) {
    fprintf(outfile, "%.15e %.15e %.15e %.15e %.15e %.15e\n", data_points[i].xCart_axis, data_points[i].log10HL, data_points[i].log10sqrtM2L,
            data_points[i].cfL, data_points[i].alphaL, data_points[i].trKL);
  }

  fclose(outfile);
}
