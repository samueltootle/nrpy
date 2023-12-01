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
  const REAL dt = commondata->dt;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    set_param_constants(griddata[grid].params);
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
  //   REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
  //   // Temporary timelevel & AUXEVOL gridfunctions:
  //   REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
  //   REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
  //   REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
  //   REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  //   params_struct *restrict params = &griddata[grid].params;
  //   REAL *restrict xx[3];
  //   for (int ww = 0; ww < 3; ww++)
  //     xx[ww] = griddata[grid].xx[ww];
  //   const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
  //   const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
  //   const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
  //   rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
  //   LOOP_ALL_GFS_GPS(i) {
  //     const REAL k_odd_gfsL = k_odd_gfs[i];
  //     const REAL y_n_gfsL = y_n_gfs[i];
  //     y_nplus1_running_total_gfs[i] = (1.0 / 6.0) * commondata->dt * k_odd_gfsL;
  //     k_odd_gfs[i] = (1.0 / 2.0) * commondata->dt * k_odd_gfsL + y_n_gfsL;
  //   }
  //   apply_bcs(commondata, params, k_odd_gfs);
  }
  // // -={ END k1 substep }=-

  // // -={ START k2 substep }=-
  // for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
  //   commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
  //   // Set gridfunction aliases from gridfuncs struct
  //   // y_n gridfunctions
  //   REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
  //   // Temporary timelevel & AUXEVOL gridfunctions:
  //   REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
  //   REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
  //   REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
  //   REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  //   params_struct *restrict params = &griddata[grid].params;
  //   REAL *restrict xx[3];
  //   for (int ww = 0; ww < 3; ww++)
  //     xx[ww] = griddata[grid].xx[ww];
  //   const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
  //   const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
  //   const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
  //   rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
  //   LOOP_ALL_GFS_GPS(i) {
  //     const REAL k_even_gfsL = k_even_gfs[i];
  //     const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
  //     const REAL y_n_gfsL = y_n_gfs[i];
  //     y_nplus1_running_total_gfs[i] = (1.0 / 3.0) * commondata->dt * k_even_gfsL + y_nplus1_running_total_gfsL;
  //     k_even_gfs[i] = (1.0 / 2.0) * commondata->dt * k_even_gfsL + y_n_gfsL;
  //   }
  //   apply_bcs(commondata, params, k_even_gfs);
  // }
  // // -={ END k2 substep }=-

  // // -={ START k3 substep }=-
  // for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
  //   commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
  //   // Set gridfunction aliases from gridfuncs struct
  //   // y_n gridfunctions
  //   REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
  //   // Temporary timelevel & AUXEVOL gridfunctions:
  //   REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
  //   REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
  //   REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
  //   REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  //   params_struct *restrict params = &griddata[grid].params;
  //   REAL *restrict xx[3];
  //   for (int ww = 0; ww < 3; ww++)
  //     xx[ww] = griddata[grid].xx[ww];
  //   const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
  //   const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
  //   const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
  //   rhs_eval(commondata, params, k_even_gfs, k_odd_gfs);
  //   LOOP_ALL_GFS_GPS(i) {
  //     const REAL k_odd_gfsL = k_odd_gfs[i];
  //     const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
  //     const REAL y_n_gfsL = y_n_gfs[i];
  //     y_nplus1_running_total_gfs[i] = (1.0 / 3.0) * commondata->dt * k_odd_gfsL + y_nplus1_running_total_gfsL;
  //     k_odd_gfs[i] = commondata->dt * k_odd_gfsL + y_n_gfsL;
  //   }
  //   apply_bcs(commondata, params, k_odd_gfs);
  // }
  // // -={ END k3 substep }=-

  // // -={ START k4 substep }=-
  // for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
  //   commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
  //   // Set gridfunction aliases from gridfuncs struct
  //   // y_n gridfunctions
  //   REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
  //   // Temporary timelevel & AUXEVOL gridfunctions:
  //   REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
  //   REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
  //   REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
  //   REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  //   params_struct *restrict params = &griddata[grid].params;
  //   REAL *restrict xx[3];
  //   for (int ww = 0; ww < 3; ww++)
  //     xx[ww] = griddata[grid].xx[ww];
  //   const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
  //   const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
  //   const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
  //   rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
  //   LOOP_ALL_GFS_GPS(i) {
  //     const REAL k_even_gfsL = k_even_gfs[i];
  //     const REAL y_n_gfsL = y_n_gfs[i];
  //     const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
  //     y_n_gfs[i] = (1.0 / 6.0) * commondata->dt * k_even_gfsL + y_n_gfsL + y_nplus1_running_total_gfsL;
  //   }
  //   apply_bcs(commondata, params, y_n_gfs);
  // }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
