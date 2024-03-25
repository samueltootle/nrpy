#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include "TESTS/TEST_prototypes.h"
/*
 * Set up a cell-centered grids of size grid_physical_size.
 */
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host, bool calling_for_first_time) {
  // Step 1.a: Set CoordSystem_hash
  CoordSystem_hash_setup(commondata, griddata);

  // Step 1.b: Set Nxx & Nxx_plus_2NGHOSTS, as well as dxx, invdxx, & xx based on grid_physical_size
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    numerical_grid_params_Nxx_dxx_xx(commondata, &griddata[grid].params, griddata[grid].xx);
    #ifdef GPU_TESTS
    TEST_param_struct(grid, &griddata[grid].params, "setup");
    TEST_coord_direction(grid, griddata[grid].xx[0], "xx0", griddata[grid].params.Nxx_plus_2NGHOSTS0);
    TEST_coord_direction(grid, griddata[grid].xx[1], "xx1", griddata[grid].params.Nxx_plus_2NGHOSTS1);
    TEST_coord_direction(grid, griddata[grid].xx[2], "xx2", griddata[grid].params.Nxx_plus_2NGHOSTS2);
    // TEST_coord_direction(grid, griddata[grid].rfmstruct.f1_of_xx1, "rfm_f1_of_xx1", griddata[grid].params.Nxx_plus_2NGHOSTS1);
    // dump_coord_direction(grid, griddata[grid].rfmstruct.f1_of_xx1__D1, "rfm_f1_of_xx1__D1", griddata[grid].params.Nxx_plus_2NGHOSTS1);
    // dump_coord_direction(grid, griddata[grid].rfmstruct.f1_of_xx1__DD11, "rfm_f1_of_xx1__DD11", griddata[grid].params.Nxx_plus_2NGHOSTS1);
    #endif
  }

  // Step 1.c: Allocate memory for and define reference-metric precomputation lookup tables

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    rfm_precompute_malloc(commondata, &griddata[grid].params, &griddata[grid].rfmstruct);
    set_param_constants(&griddata[grid].params);
    rfm_precompute_defines(commondata, &griddata[grid].params, &griddata[grid].rfmstruct, griddata[grid].xx);
  }
  cpyDevicetoHost__grid(commondata, griddata_host, griddata);
  cudaDeviceSynchronize();
  // Step 1.d: Set up curvilinear boundary condition struct (bcstruct)

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    set_param_constants(&griddata[grid].params);
    bcstruct_set_up(commondata, &griddata[grid].params, griddata_host[grid].xx, &griddata[grid].bcstruct);
  }
  cudaDeviceSynchronize();
  // Step 1.e: Set timestep based on minimum spacing between neighboring gridpoints.
  commondata->dt = 1e30;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    set_param_constants(&griddata[grid].params);
    cfl_limited_timestep(commondata, &griddata[grid].params, griddata[grid].xx, &griddata[grid].bcstruct);
  }

  // Step 1.f: Initialize timestepping parameters to zero if this is the first time this function is called.
  if (calling_for_first_time) {
    commondata->nn = 0;
    commondata->nn_0 = 0;
    commondata->t_0 = 0.0;
    commondata->time = 0.0;
    #ifdef GPU_TESTS
    TEST_commondata(commondata, "first_time");
    #endif
  }
}
