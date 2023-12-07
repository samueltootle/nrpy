#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include "init_gpu_globals.h"
#define DEBUG_RHS
/*
 * -={ main() function }=-
 * Step 1.a: Set each commondata CodeParameter to default.
 * Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
 * Step 1.c: Allocate NUMGRIDS griddata arrays, each containing data specific to an individual grid.
 * Step 1.d: Set each CodeParameter in griddata.params to default.
 * Step 2: Initial data are set on y_n_gfs gridfunctions. Allocate storage for them first.
 * Step 3: Finalize initialization: set up SphericalGaussian initial data, etc.
 * Step 4: Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping.
 *
 * Step 5: MAIN SIMULATION LOOP
 * - Step 5.a: Output diagnostics.
 * - Step 5.b: Prepare to step forward in time.
 * - Step 5.c: Step forward in time using Method of Lines with RK4 algorithm, applying Quadratic extrapolation, manually defined boundary conditions.
 * - Step 5.d: Finish up step in time.
 * Step 6: Free all allocated memory.
 */
int main(int argc, const char *argv[]) {
  commondata_struct commondata;       // commondata contains parameters common to all grids.
  griddata_struct * griddata; // griddata contains data specific to an individual grid.
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  // Step 1.a: Set each commondata CodeParameter to default.
  commondata_struct_set_to_default(&commondata);

  // Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
  cmdline_input_and_parfile_parser(&commondata, argc, argv);

  // Step 1.c: Allocate NUMGRIDS griddata arrays, each containing data specific to an individual grid.
  griddata = (griddata_struct *)malloc(sizeof(griddata_struct) * commondata.NUMGRIDS);
  cudaCheckErrors(cudaMalloc, "malloc Failed")

  // Step 1.d: Set each CodeParameter in griddata.params to default.
  params_struct_set_to_default(&commondata, griddata);
  set_fd_constants();
  set_commondata_constants(&commondata);

  // Step 1.e: Set up numerical grids: xx[3], masks, Nxx, dxx, invdxx, bcstruct, rfm_precompute, timestep, etc.
  {
    // if calling_for_first_time, then initialize commondata time=nn=t_0=nn_0 = 0
    const bool calling_for_first_time = true;
    numerical_grids_and_timestep(&commondata, griddata, calling_for_first_time);
  }

  for (int grid = 0; grid < commondata.NUMGRIDS; grid++) {
    // Step 2: Initial data are set on y_n_gfs gridfunctions. Allocate storage for them first.
    MoL_malloc_y_n_gfs(&commondata, griddata[grid].params, &griddata[grid].gridfuncs);
  }

  // Step 3: Finalize initialization: set up initial data, etc.
  initial_data(&commondata, griddata);

  // Step 4: Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping
  for (int grid = 0; grid < commondata.NUMGRIDS; grid++) {
    MoL_malloc_non_y_n_gfs(&commondata, griddata[grid].params, &griddata[grid].gridfuncs);
  }

  // Step 5: MAIN SIMULATION LOOP
  #ifndef DEBUG_RHS
  while (commondata.time < commondata.t_final) { // Main loop to progress forward in time.
    // Step 5.a: Main loop, part 1: Output diagnostics
    diagnostics(&commondata, griddata);

    // Step 5.b: Main loop, part 2 (pre_MoL_step_forward_in_time): Prepare to step forward in time
    // (nothing here; specify by setting pre_MoL_step_forward_in_time string in register_CFunction_main_c().)

    // Step 5.c: Main loop, part 3: Step forward in time using Method of Lines with RK4 algorithm,
    //           applying Quadratic extrapolation, manually defined boundary conditions.
    MoL_step_forward_in_time(&commondata, griddata);

    // Step 5.d: Main loop, part 4 (post_MoL_step_forward_in_time): Finish up step in time
    // (nothing here; specify by setting post_MoL_step_forward_in_time string in register_CFunction_main_c().)
  } // End main loop to progress forward in time.
  #else
  int n = 0;
  while (n < 5) {
    MoL_step_forward_in_time(&commondata, griddata);
    printf("\n\n");
    n++;
  } // End main loop to progress forward in time.
  #endif

  // Step 5: Free all allocated memory
  for (int grid = 0; grid < commondata.NUMGRIDS; grid++) {
    MoL_free_memory_y_n_gfs(&griddata[grid].gridfuncs);
    MoL_free_memory_non_y_n_gfs(&griddata[grid].gridfuncs);
    for (int i = 0; i < 3; i++) {
      cudaFree(griddata[grid].xx[i]);
      cudaFree(griddata[grid].params);
    }
  }
  cudaFree(griddata);
  return 0;
}
