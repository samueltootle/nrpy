#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Diagnostics.
 */
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host) {

  const REAL currtime = commondata->time, currdt = commondata->dt, outevery = commondata->diagnostics_output_every;
  // Explanation of the if() below:
  // Step 1: round(currtime / outevery) rounds to the nearest integer multiple of currtime.
  // Step 2: Multiplying by outevery yields the exact time we should output again, t_out.
  // Step 3: If fabs(t_out - currtime) < 0.5 * currdt, then currtime is as close to t_out as possible!
  if (fabs(round(currtime / outevery) * outevery - currtime) < 0.5 * currdt) {
    for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
      // Unpack griddata struct:
      const REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
      REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
      REAL *restrict diagnostic_output_gfs = griddata[grid].gridfuncs.diagnostic_output_gfs;
      
      // REAL *restrict xx[3];
      // {
      //   for (int ww = 0; ww < 3; ww++) {
      //     xx[ww] = griddata[grid].xx[ww];
      //   }
      // }
      const params_struct *restrict params = &griddata[grid].params;
      set_param_constants(params);
      #include "set_CodeParameters.h"

      // Depends on the number of output GFs
      REAL * y_n_gfs__host[NUM_DIAG_YN];
      REAL * diagnostics_gfs__host[NUM_AUX_GFS];
      const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
      
      for(int i = 0; i < NUM_DIAG_YN; ++i) {
        cudaMallocHost(&y_n_gfs__host[i], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot);
      }
      for(int i = 0; i < NUM_AUX_GFS; ++i) {
        // y_n_gfs__host[i] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS_tot);
        cudaMallocHost(&diagnostics_gfs__host[i], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot);
      }

      // Constraint output
      {
        Ricci_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs);
        constraints_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs, diagnostic_output_gfs);
      }
      cpyDevicetoHost__gf(commondata, params, y_n_gfs__host[0], y_n_gfs, CFGF);
      cpyDevicetoHost__gf(commondata, params, y_n_gfs__host[1], y_n_gfs, ALPHAGF);
      cpyDevicetoHost__gf(commondata, params, y_n_gfs__host[2], y_n_gfs, TRKGF);

      cpyDevicetoHost__gf(commondata, params, diagnostics_gfs__host[HGF], diagnostic_output_gfs, HGF);
      cpyDevicetoHost__gf(commondata, params, diagnostics_gfs__host[MSQUAREDGF], diagnostic_output_gfs, MSQUAREDGF);

      // 0D output
      diagnostics_nearest_grid_center(commondata, params, &griddata[grid].gridfuncs);

      // 1D output
      diagnostics_nearest_1d_y_axis(commondata, params, griddata[grid].xx, &griddata[grid].gridfuncs);
      diagnostics_nearest_1d_z_axis(commondata, params, griddata[grid].xx, &griddata[grid].gridfuncs);

      // 2D output
      diagnostics_nearest_2d_xy_plane(commondata, params, griddata[grid].xx, &griddata[grid].gridfuncs);
      diagnostics_nearest_2d_yz_plane(commondata, params, griddata[grid].xx, &griddata[grid].gridfuncs);
      for(int i = 0; i < NUM_DIAG_YN; ++i) {
        cudaFreeHost(y_n_gfs__host[i]); 
      }
      for(int i = 0; i < NUM_AUX_GFS; ++i) {
        cudaFreeHost(diagnostics_gfs__host[i]); 
      }
    }
  }
  progress_indicator(commondata, griddata);
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
}
