#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Set up a cell-centered grids of size grid_physical_size.
 */
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host,
                                  bool calling_for_first_time) {

  // Step 1.a: Set each CodeParameter in griddata.params to default, for MAXNUMGRIDS grids.
  params_struct_set_to_default(commondata, griddata);
  if (strncmp(commondata->gridding_choice, "independent grid(s)", 200) == 0) {
    // Independent grids
    bool grid_is_resized = false;
    int Nx[3] = {-1, -1, -1};

    // Step 1.b: For each grid, set Nxx & Nxx_plus_2NGHOSTS, as well as dxx, invdxx, & xx based on grid_physical_size
    int grid = 0;
    griddata[grid].params.CoordSystem_hash = SINHSYMTP;
    griddata[grid].params.grid_idx = grid;
    numerical_grid_params_Nxx_dxx_xx(commondata, &griddata[grid].params, griddata[grid].xx, Nx, grid_is_resized);
    grid++;
  }

  // Step 1.c: Allocate memory for and define reference-metric precomputation lookup tables

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    rfm_precompute_malloc(commondata, &griddata[grid].params, &griddata[grid].rfmstruct);
    cpyHosttoDevice_params__constant(&griddata[grid].params);
    rfm_precompute_defines(commondata, &griddata[grid].params, &griddata[grid].rfmstruct, griddata[grid].xx);
  }
  cpyDevicetoHost__grid(commondata, griddata_host, griddata);
  cudaDeviceSynchronize();

  // Step 1.d: Set up curvilinear boundary condition struct (bcstruct)

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    cpyHosttoDevice_params__constant(&griddata[grid].params);
    bcstruct_set_up(commondata, &griddata[grid].params, griddata_host[grid].xx, &griddata[grid].bcstruct);
  }

  // Step 1.e: Set timestep based on minimum spacing between neighboring gridpoints.
  commondata->dt = 1e30;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    cpyHosttoDevice_params__constant(&griddata[grid].params);
    cfl_limited_timestep(commondata, &griddata[grid].params, griddata[grid].xx, &griddata[grid].bcstruct);
  }

  // Step 1.f: Initialize timestepping parameters to zero if this is the first time this function is called.
  if (calling_for_first_time) {
    commondata->nn = 0;
    commondata->nn_0 = 0;
    commondata->t_0 = 0.0;
    commondata->time = 0.0;
  }
}
