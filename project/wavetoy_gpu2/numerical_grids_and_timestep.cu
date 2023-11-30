#include "BHaH_defines.h"
/*
 * Set up cell-centered Cartesian grids.
 */
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, bool calling_for_first_time) {

  commondata->dt = 1e30;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    params_struct *restrict params = &griddata[grid].params;
    const REAL convergence_factor = commondata->convergence_factor;
    const REAL xxmin0 = params->xxmin0;
    const REAL xxmin1 = params->xxmin1;
    const REAL xxmin2 = params->xxmin2;
    const REAL xxmax0 = params->xxmax0;
    const REAL xxmax1 = params->xxmax1;
    const REAL xxmax2 = params->xxmax2;

    params->Nxx0 *= convergence_factor;
    params->Nxx1 *= convergence_factor;
    params->Nxx2 *= convergence_factor;

    params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
    params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
    params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;

    params->dxx0 = (xxmax0 - xxmin0) / ((REAL)params->Nxx0);
    params->dxx1 = (xxmax1 - xxmin1) / ((REAL)params->Nxx1);
    params->dxx2 = (xxmax2 - xxmin2) / ((REAL)params->Nxx2);

    params->invdxx0 = ((REAL)params->Nxx0) / (xxmax0 - xxmin0);
    params->invdxx1 = ((REAL)params->Nxx1) / (xxmax1 - xxmin1);
    params->invdxx2 = ((REAL)params->Nxx2) / (xxmax2 - xxmin2);

    // Initialize timestepping parameters to zero if this is the first time this function is called.
    if (calling_for_first_time) {
      commondata->nn = 0;
      commondata->nn_0 = 0;
      commondata->t_0 = 0.0;
      commondata->time = 0.0;
    }
    commondata->dt = MIN(commondata->dt, commondata->CFL_FACTOR * MIN(params->dxx0, MIN(params->dxx1, params->dxx2))); // CFL condition

    // Set up cell-centered Cartesian coordinate grid, centered at the origin.
    griddata[grid].xx[0] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
    griddata[grid].xx[1] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS1);
    griddata[grid].xx[2] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS2);
    for (int j = 0; j < params->Nxx_plus_2NGHOSTS0; j++)
      griddata[grid].xx[0][j] = xxmin0 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx0;
    for (int j = 0; j < params->Nxx_plus_2NGHOSTS1; j++)
      griddata[grid].xx[1][j] = xxmin1 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx1;
    for (int j = 0; j < params->Nxx_plus_2NGHOSTS2; j++)
      griddata[grid].xx[2][j] = xxmin2 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx2;
  }
}