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

      const params_struct *restrict params = &griddata[grid].params;
      set_param_constants(params);
      #include "set_CodeParameters.h"

      // Constraint output
      {
        Ricci_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs);
        cudaEvent_t start;
        cudaEventCreateWithFlags(&start, cudaEventDisableTiming);

        constraints_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs, diagnostic_output_gfs);
        size_t streamid = params->grid_idx % nstreams;
        cudaEventRecord(start, streams[streamid]);
        cudaEventSynchronize(start);
        cudaEventDestroy(start);
      }
      constexpr int num_copy_kernels = 5;
      cudaEvent_t start[num_copy_kernels];
      for(int i = 0; i < num_copy_kernels; ++i) {
        cudaEventCreateWithFlags(&start[i], cudaEventDisableTiming);
      }
      // Could this be a for loop somehow?
      size_t streamid = cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_ALPHAGF, ALPHAGF);
      cudaEventRecord(start[0], streams[streamid]);

      streamid = cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_CFGF, CFGF);
      cudaEventRecord(start[1], streams[streamid]);

      streamid = cpyDevicetoHost__gf(commondata, params, host_diag_y_n_gfs, y_n_gfs, DIAG_TRKGF, TRKGF);
      cudaEventRecord(start[2], streams[streamid]);

      streamid = cpyDevicetoHost__gf(commondata, params, host_diagnostic_output_gfs, diagnostic_output_gfs, HGF, HGF);
      cudaEventRecord(start[3], streams[streamid]);

      streamid = cpyDevicetoHost__gf(commondata, params, host_diagnostic_output_gfs, diagnostic_output_gfs, MSQUAREDGF, MSQUAREDGF);
      cudaEventRecord(start[4], streams[streamid]);

      for(int i = 0; i < num_copy_kernels; ++i) {
        cudaEventSynchronize(start[i]);
        cudaEventDestroy(start[i]);
      }
      cudaDeviceSynchronize();

      // 0D output
      diagnostics_nearest_grid_center(commondata, params, &griddata_host[grid].gridfuncs);

      // 1D output
      diagnostics_nearest_1d_y_axis(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
      diagnostics_nearest_1d_z_axis(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);

      // 2D output
      diagnostics_nearest_2d_xy_plane(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
      diagnostics_nearest_2d_yz_plane(commondata, params, griddata_host[grid].xx, &griddata_host[grid].gridfuncs);
    }
  }
  progress_indicator(commondata, griddata);
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
}
