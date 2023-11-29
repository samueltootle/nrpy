#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
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
      REAL *restrict xx[3];
      for (int ww = 0; ww < 3; ww++)
        xx[ww] = griddata[grid].xx[ww];
      const params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"

      // 0D output
      {
        char filename[256];
        sprintf(filename, "out0d-conv_factor%.2f.txt", convergence_factor);
        FILE *outfile;
        if (nn == 0)
          outfile = fopen(filename, "w");
        else
          outfile = fopen(filename, "a");
        if (outfile == NULL) {
          fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
          exit(1);
        }

        const int i0_center = Nxx_plus_2NGHOSTS0 / 2;
        const int i1_center = Nxx_plus_2NGHOSTS1 / 2;
        const int i2_center = Nxx_plus_2NGHOSTS2 / 2;
        const int center_of_grid_idx = IDX3(i0_center, i1_center, i2_center);

        const REAL num_soln_at_center_UUGF = y_n_gfs[IDX4pt(UUGF, center_of_grid_idx)];
        const REAL num_soln_at_center_VVGF = y_n_gfs[IDX4pt(VVGF, center_of_grid_idx)];
        REAL exact_soln_at_center_UUGF, exact_soln_at_center_VVGF;
        exact_solution_single_Cartesian_point(commondata, 
                                              params, 
                                              xx[0][i0_center], 
                                              xx[1][i1_center], 
                                              xx[2][i2_center], 
                                              &exact_soln_at_center_UUGF,
                                              &exact_soln_at_center_VVGF);

        fprintf(outfile, "%e %e %e %e %e\n", time, fabs(fabs(num_soln_at_center_UUGF - exact_soln_at_center_UUGF) / exact_soln_at_center_UUGF),
                fabs(fabs(num_soln_at_center_VVGF - exact_soln_at_center_VVGF) / (1e-16 + exact_soln_at_center_VVGF)), num_soln_at_center_UUGF,
                exact_soln_at_center_UUGF);

        fclose(outfile);
      }
    }
  }
  progress_indicator(commondata, griddata);
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
}
