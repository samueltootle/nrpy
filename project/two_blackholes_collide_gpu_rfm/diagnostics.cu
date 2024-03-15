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

      // In principle diag_y_n_gfs is only a subset of EVOLGFS and the index of the quantities
      // saved during diagnostics are not necessarily indexed 0, 1, 2, etc (see EVOL_GFS in BHaH_defines.h).  
      // Therefore, host_diag_y_n_gfs currently uses different array indicies (see DIAG_YN in BHaH_defines.h) 
      // to reduce the memory footprint of griddata_host.
      REAL *restrict host_diag_y_n_gfs = griddata_host[grid].gridfuncs.y_n_gfs;
      REAL *restrict host_diagnostic_output_gfs = griddata_host[grid].gridfuncs.diagnostic_output_gfs;
      
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
      // REAL * y_n_gfs__host[NUM_DIAG_YN];
      // REAL * diagnostics_gfs__host[NUM_AUX_GFS];
      const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
      
      // for(int i = 0; i < NUM_DIAG_YN; ++i) {
      //   cudaMallocHost(&y_n_gfs__host[i], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot);
      // }
      // for(int i = 0; i < NUM_AUX_GFS; ++i) {
      //   cudaMallocHost(&diagnostics_gfs__host[i], sizeof(REAL) * Nxx_plus_2NGHOSTS_tot);
      // }

      // Constraint output
      {
        Ricci_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs);
        constraints_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs, diagnostic_output_gfs);
      }
      // printf("Copy ALPHAGF\n");
      cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_ALPHAGF, ALPHAGF);
      // printf("Copy CFGF\n");
      cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_CFGF, CFGF);
      // printf("Copy TRKGF\n");
      cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_TRKGF, TRKGF);

      cpyDevicetoHost__gf(commondata, params, host_diagnostic_output_gfs, diagnostic_output_gfs, HGF, HGF);
      cpyDevicetoHost__gf(commondata, params, host_diagnostic_output_gfs, diagnostic_output_gfs, MSQUAREDGF, MSQUAREDGF);
      cudaDeviceSynchronize();

      // 0D output
      diagnostics_nearest_grid_center(commondata, params, &griddata_host[grid].gridfuncs);

      // 1D output
      diagnostics_nearest_1d_y_axis(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
      diagnostics_nearest_1d_z_axis(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);

      // 2D output
      diagnostics_nearest_2d_xy_plane(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
      diagnostics_nearest_2d_yz_plane(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
      // for(int i = 0; i < NUM_DIAG_YN; ++i) {
      //   cudaFreeHost(y_n_gfs__host[i]); 
      // }
      // for(int i = 0; i < NUM_AUX_GFS; ++i) {
      //   cudaFreeHost(diagnostics_gfs__host[i]); 
      // }
    }
  }
  progress_indicator(commondata, griddata);
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
}
