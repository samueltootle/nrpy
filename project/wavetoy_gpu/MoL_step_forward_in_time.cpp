#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
/*
 * Method of Lines (MoL) for "RK4" method: Step forward one full timestep.
 *
 */


void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ RK4 }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    cudaDeviceSynchronize();

    REAL rk_weight = 1./6.;
    REAL dt_step_factor = 1./2.;
    
    rk_substep(commondata, params, &griddata[grid].gridfuncs, \
                rk_weight, dt_step_factor);
    
    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    cudaDeviceSynchronize();

    REAL rk_weight = 1./3.;
    REAL dt_step_factor = 1./2.;
    
    rk_substep(commondata, params, &griddata[grid].gridfuncs, \
                rk_weight, dt_step_factor);
    
    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    cudaDeviceSynchronize();

    REAL rk_weight = 1. / 3.;
    REAL dt_step_factor = 1.;
    
    rk_substep(commondata, params, &griddata[grid].gridfuncs, \
                rk_weight, dt_step_factor);
    
    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    cudaDeviceSynchronize();

    REAL rk_weight = 1.;
    REAL dt_step_factor = 1. / 6.;
    
    rk_substep(commondata, params, &griddata[grid].gridfuncs, \
                rk_weight, dt_step_factor);
    
    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
