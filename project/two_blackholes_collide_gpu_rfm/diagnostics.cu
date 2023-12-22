#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Diagnostics.
 */
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

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
      //   for (int ww = 0; ww < 3; ww++)
      //     xx[ww] = griddata[grid].xx[ww];
      // }
      const params_struct *restrict params = &griddata[grid].params;
      set_param_constants(params);
#include "set_CodeParameters.h"

      // Constraint output
      {
        Ricci_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs);
        constraints_eval(commondata, params, &griddata[grid].rfmstruct, y_n_gfs, auxevol_gfs, diagnostic_output_gfs);
      }

      // 0D output
      diagnostics_nearest_grid_center(commondata, params, &griddata[grid].gridfuncs);

      // // 1D output
      // diagnostics_nearest_1d_y_axis(commondata, params, xx, &griddata[grid].gridfuncs);
      // diagnostics_nearest_1d_z_axis(commondata, params, xx, &griddata[grid].gridfuncs);

      // // 2D output
      // diagnostics_nearest_2d_xy_plane(commondata, params, xx, &griddata[grid].gridfuncs);
      // diagnostics_nearest_2d_yz_plane(commondata, params, xx, &griddata[grid].gridfuncs);
    }
  }
  // progress_indicator(commondata, griddata);
  // if (commondata->time + commondata->dt > commondata->t_final)
  //   printf("\n");
}
