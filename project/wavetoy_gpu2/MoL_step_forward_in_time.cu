#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Method of Lines (MoL) for "RK4" method: Step forward one full timestep.
 *
 */
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ RK4 }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;
  const REAL dt = commondata->dt;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    set_param_constants(griddata[grid].params);

    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    __attribute__((unused)) REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    __attribute__((unused)) REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    __attribute__((unused)) REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    __attribute__((unused)) REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    __attribute__((unused)) REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    __attribute__((unused)) params_struct *restrict params = griddata[grid].params;
    __attribute__((unused)) REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++) {
      xx[ww] = griddata[grid].xx[ww];
    }

    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    rk_substep1(params,
            y_n_gfs,
            y_nplus1_running_total_gfs,
            k_odd_gfs,
            k_even_gfs,
            auxevol_gfs,dt);

    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.50000000000000000e+00 * commondata->dt;
    set_param_constants(griddata[grid].params);

    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    __attribute__((unused)) REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    __attribute__((unused)) REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    __attribute__((unused)) REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    __attribute__((unused)) REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    __attribute__((unused)) REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    __attribute__((unused)) params_struct *restrict params = griddata[grid].params;
    __attribute__((unused)) REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++) {
      xx[ww] = griddata[grid].xx[ww];
    }

    rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
    rk_substep2(params,
              y_n_gfs,
              y_nplus1_running_total_gfs,
              k_odd_gfs,
              k_even_gfs,
              auxevol_gfs,dt);

    apply_bcs(commondata, params, k_even_gfs);
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.50000000000000000e+00 * commondata->dt;
    set_param_constants(griddata[grid].params);

    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    __attribute__((unused)) REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    __attribute__((unused)) REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    __attribute__((unused)) REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    __attribute__((unused)) REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    __attribute__((unused)) REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    __attribute__((unused)) params_struct *restrict params = griddata[grid].params;
    __attribute__((unused)) REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++) {
      xx[ww] = griddata[grid].xx[ww];
    }
    rhs_eval(commondata, params, k_even_gfs, k_odd_gfs);
    apply_bcs(commondata, params, k_odd_gfs);
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
  set_param_constants(griddata[grid].params);

    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    __attribute__((unused)) REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    __attribute__((unused)) REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    __attribute__((unused)) REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    __attribute__((unused)) REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    __attribute__((unused)) REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    __attribute__((unused)) params_struct *restrict params = griddata[grid].params;
    __attribute__((unused)) REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++) {
      xx[ww] = griddata[grid].xx[ww];
    }
    rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
    apply_bcs(commondata, params, y_n_gfs);
  }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
