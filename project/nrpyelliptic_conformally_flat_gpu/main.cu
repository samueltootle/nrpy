#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_global_defines.h"
/*
 * -={ main() function }=-
 * Step 1.a: Set each commondata CodeParameter to default.
 * Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
 * Step 1.c: Allocate MAXNUMGRIDS griddata structs, each containing data specific to an individual grid.
 * Step 1.d: Set each CodeParameter in griddata.params to default.
 * Step 1.e: Set up numerical grids: NUMGRIDS, xx[3], masks, Nxx, dxx, invdxx, bcstruct, rfm_precompute, timestep, etc.
 * Step 2: Initial data are set on y_n_gfs gridfunctions. Allocate storage for them first.
 * Step 3: Finalize initialization: set up {self.initial_data_desc}initial data, etc.
 * Step 4: Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping.
 * Step 4.a: Functions called after memory for non-y_n and auxevol gridfunctions is allocated.Step 5: MAIN SIMULATION LOOP
 * - Step 5.a: Output diagnostics.
 * - Step 5.b: Prepare to step forward in time.
 * - Step 5.c: Step forward in time using Method of Lines with RK4 algorithm, applying outgoing radiation boundary conditions.
 * - Step 5.d: Finish up step in time.
 * Step 6: Free all allocated memory.
 */
int main(int argc, const char *argv[]) {

#include "BHaH_gpu_global_init.h"
  commondata_struct commondata;            // commondata contains parameters common to all grids.
  griddata_struct *restrict griddata;      // griddata contains data specific to an individual grid.
  griddata_struct *restrict griddata_host; // stores only the host data needed for diagnostics

  // Step 1.a: Set each commondata CodeParameter to default.
  commondata_struct_set_to_default(&commondata);

  // Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
  cmdline_input_and_parfile_parser(&commondata, argc, argv);

  // Step 1.c: Allocate NUMGRIDS griddata arrays, each containing data specific to an individual grid.
  griddata = (griddata_struct *)malloc(sizeof(griddata_struct) * commondata.NUMGRIDS);
  griddata_host = (griddata_struct *)malloc(sizeof(griddata_struct) * commondata.NUMGRIDS);

  // Step 1.d: Set up numerical grids: xx[3], masks, Nxx, dxx, invdxx, bcstruct, rfm_precompute, timestep, etc.
  {
    // if calling_for_first_time, then initialize commondata time=nn=t_0=nn_0 = 0
    const bool calling_for_first_time = true;
    numerical_grids_and_timestep(&commondata, griddata, griddata_host, calling_for_first_time);
  }

  for (int grid = 0; grid < commondata.NUMGRIDS; grid++) {
    // Step 2: Initial data are set on y_n_gfs gridfunctions. Allocate storage for them first.
    MoL_malloc_y_n_gfs(&commondata, &griddata[grid].params, &griddata[grid].gridfuncs);
    CUDA__malloc_host_gfs(&commondata, &griddata[grid].params, &griddata_host[grid].gridfuncs);
  }

  // Step 3: Finalize initialization: set up initial data, etc.
  if (!read_checkpoint(&commondata, griddata_host, griddata))
    initial_data(&commondata, griddata);

  // Step 4: Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping
  for (int grid = 0; grid < commondata.NUMGRIDS; grid++) {
    MoL_malloc_non_y_n_gfs(&commondata, &griddata[grid].params, &griddata[grid].gridfuncs);
  }
  // Step 4.a: Functions called after memory for non-y_n and auxevol gridfunctions is allocated.
  initialize_constant_auxevol(&commondata, griddata);

  // Step 5: MAIN SIMULATION LOOP
  // while(commondata.time < commondata.t_final) { // Main loop to progress forward in time.
  for (int i = 0; i < 1; ++i) {
    // Step 5.a: Main loop, part 1: Output diagnostics
    diagnostics(&commondata, griddata, griddata_host);

    // Step 5.b: Main loop, part 2 (pre_MoL_step_forward_in_time): Prepare to step forward in time
    write_checkpoint(&commondata, griddata_host, griddata);

    // Step 5.c: Main loop, part 3: Step forward in time using Method of Lines with RK4 algorithm,
    //           applying outgoing radiation boundary conditions.
    MoL_step_forward_in_time(&commondata, griddata);

    // Step 5.d: Main loop, part 4 (post_MoL_step_forward_in_time): Finish up step in time
    check_stop_conditions(&commondata, griddata);
    if (commondata.stop_relaxation) {
      // Force a checkpoint when stop condition is reached.
      commondata.checkpoint_every = 1e-4 * commondata.dt;
      write_checkpoint(&commondata, griddata_host, griddata);
      break;
    }

  } // End main loop to progress forward in time.
  // Make sure all workers are done
  cudaDeviceSynchronize();
  for (int i = 0; i < nstreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  // Step 6: Free all allocated memory
  {
    const bool enable_free_non_y_n_gfs = true;
    griddata_free(&commondata, griddata, griddata_host, enable_free_non_y_n_gfs);
  }
  return 0;
}
