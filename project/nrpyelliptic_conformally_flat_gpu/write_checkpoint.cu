#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Write a checkpoint file
 */
void write_checkpoint(const commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_GPU) {

  char filename[256];
  snprintf(filename, 256, "checkpoint-conv_factor%.2f.dat", commondata->convergence_factor);
  const REAL currtime = commondata->time, currdt = commondata->dt, outevery = commondata->checkpoint_every;
  // Explanation of the if() below:
  // Step 1: round(currtime / outevery) rounds to the nearest integer multiple of currtime.
  // Step 2: Multiplying by outevery yields the exact time we should output again, t_out.
  // Step 3: If fabs(t_out - currtime) < 0.5 * currdt, then currtime is as close to t_out as possible!
  if (fabs(round(currtime / outevery) * outevery - currtime) < 0.5 * currdt) {
    FILE *cp_file = fopen(filename, "w+");
    fwrite(commondata, sizeof(commondata_struct), 1, cp_file);
    fprintf(stderr, "WRITING CHECKPOINT: cd struct size = %ld time=%e\n", sizeof(commondata_struct), commondata->time);
    for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
      // Set gridfunctions aliases for HOST data
      REAL *restrict diagnostic_output_gfs = griddata[grid].gridfuncs.y_n_gfs;

      // Set gridfunctions aliases for GPU data
      REAL *restrict y_n_gfs = griddata_GPU[grid].gridfuncs.y_n_gfs;

      // Make sure host griddata has correct params
      memcpy(&griddata[grid].params, &griddata_GPU[grid].params, sizeof(params_struct));
      for (int gf = 0; gf < NUM_EVOL_GFS; ++gf) {
        cpyDevicetoHost__gf(commondata, &griddata[grid].params, diagnostic_output_gfs, y_n_gfs, gf, gf);
      }

      fwrite(&griddata[grid].params, sizeof(params_struct), 1, cp_file);
      const int ntot =
          (griddata[grid].params.Nxx_plus_2NGHOSTS0 * griddata[grid].params.Nxx_plus_2NGHOSTS1 * griddata[grid].params.Nxx_plus_2NGHOSTS2);

      // Does this need to be on GPU?  Where is MASK?
      int count = 0;
      const int maskval = 1; // to be replaced with griddata[grid].mask[i].
#pragma omp parallel for reduction(+ : count)
      for (int i = 0; i < ntot; i++) {
        if (maskval >= +0)
          count++;
      }
      fwrite(&count, sizeof(int), 1, cp_file);

      int *out_data_indices = (int *)malloc(sizeof(int) * count);
      REAL *compact_out_data = (REAL *)malloc(sizeof(REAL) * NUM_EVOL_GFS * count);
      int which_el = 0;
      // Should be a local sync?
      cudaDeviceSynchronize();
      for (int i = 0; i < ntot; i++) {
        if (maskval >= +0) {
          out_data_indices[which_el] = i;
          for (int gf = 0; gf < NUM_EVOL_GFS; gf++)
            compact_out_data[which_el * NUM_EVOL_GFS + gf] = diagnostic_output_gfs[ntot * gf + i];
          which_el++;
        }
      }
      // printf("HEY which_el = %d | count = %d\n", which_el, count);
      fwrite(out_data_indices, sizeof(int), count, cp_file);
      fwrite(compact_out_data, sizeof(REAL), count * NUM_EVOL_GFS, cp_file);
      free(out_data_indices);
      free(compact_out_data);
    }
    fclose(cp_file);
    fprintf(stderr, "FINISHED WRITING CHECKPOINT\n");
  }
}
