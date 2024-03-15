#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
/*
 * Output diagnostic quantities at grid's *physical* center.
 * For example:
 * In Cartesian this will be at i0_mid,i1_mid,i2_mid.
 * In Spherical, this will be at i0_min,i1_mid,i2_mid (i1 and i2 don't matter).
 * In Cylindrical, this will be at i0_min,i1_mid,i2_mid (i1 == phi doesn't matter).
 * In SinhSymTP, this will be at i0_min,i1_mid,i2_mid (i2 == phi doesn't matter).
 */
void diagnostics_nearest_grid_center__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     MoL_gridfunctions_struct *restrict gridfuncs) {
#include "../set_CodeParameters.h"

  // Unpack gridfuncs struct:
  __attribute_maybe_unused__ const REAL *restrict y_n_gfs = gridfuncs->y_n_gfs;
  __attribute_maybe_unused__ const REAL *restrict auxevol_gfs = gridfuncs->auxevol_gfs;
  __attribute_maybe_unused__ const REAL *restrict diagnostic_output_gfs = gridfuncs->diagnostic_output_gfs;

  // Output to file diagnostic quantities at grid's *physical* center.
  char filename[256];
  sprintf(filename, "out0d-conv_factor%.2f.txt", convergence_factor);
  FILE *outfile = (nn == 0) ? fopen(filename, "w") : fopen(filename, "a");
  if (!outfile) {
    fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
    exit(1);
  }

  const int i0_center = NGHOSTS;
  const int i1_center = Nxx_plus_2NGHOSTS1 / 2;
  const int i2_center = Nxx_plus_2NGHOSTS2 / 2;

  const auto get_diagnostics = [](auto index, const REAL *restrict g_data) {
    // REAL h_data;
    // cudaMemcpy(&h_data, &g_data[index], sizeof(REAL), cudaMemcpyDeviceToHost);
    // cudaCheckErrors(cudaMemcpy, "memory error");
    REAL h_data = g_data[index];
    return h_data;
  };
  if (i0_center != -1 && i1_center != -1 && i2_center != -1) {
    const int idx3 = IDX3(i0_center, i1_center, i2_center);

    const REAL HL = get_diagnostics(IDX4pt(HGF, idx3), diagnostic_output_gfs);
    const REAL log10HL = log10(fabs(HL + 1e-16));
    
    const REAL M2L = get_diagnostics(IDX4pt(MSQUAREDGF, idx3), diagnostic_output_gfs);
    const REAL log10sqrtM2L = log10(sqrt(M2L) + 1e-16);
    
    const REAL cfL = get_diagnostics(IDX4pt(DIAG_CFGF, idx3), y_n_gfs);
    const REAL alphaL = get_diagnostics(IDX4pt(DIAG_ALPHAGF, idx3), y_n_gfs);
    const REAL trKL = get_diagnostics(IDX4pt(DIAG_TRKGF, idx3), y_n_gfs);

    fprintf(outfile, "%e %.15e %.15e %.15e %.15e %.15e\n", time, log10HL, log10sqrtM2L, cfL, alphaL, trKL);
  }
  fclose(outfile);
}
