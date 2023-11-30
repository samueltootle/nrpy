#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
/*
 * Method of Lines (MoL) for "RK4" method: Step forward one full timestep.
 *
 */
#define TEST(arg) printf("\n %s - %1.15f - %1.15f - %1.15f - %1.15f\n", \
      #arg, \
      y_nplus1_running_total_gfs[43], \
      y_n_gfs[43], \
      k_odd_gfs[43], \
      k_even_gfs[43]); \

void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ RK4 }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;
  const REAL dt = commondata->dt;

  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    // cudaMemPrefetchAsync(griddata[grid].xx[0], sizeof(REAL) * params->Nxx_plus_2NGHOSTS0, 0);
    // cudaCheckErrors(griddata[grid].xx[0], "prefetch failed");

    params_struct *restrict params = &griddata[grid].params;
    set_param_constants(params);
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    rhs_eval(commondata, params, y_n_gfs, k_odd_gfs);
    cudaDeviceSynchronize();
    
    rk_substep1(params,
               y_n_gfs,
               y_nplus1_running_total_gfs,
               k_odd_gfs,
               k_even_gfs,
               auxevol_gfs,dt);
    cudaDeviceSynchronize();
    apply_bcs(commondata, params, k_odd_gfs);
    cudaDeviceSynchronize();
    TEST(RK_12)
    cudaDeviceSynchronize();
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    set_param_constants(params);
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;

    rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
    cudaDeviceSynchronize();
    TEST(RK_20)
    cudaDeviceSynchronize();
    rk_substep2(params,
               y_n_gfs,
               y_nplus1_running_total_gfs,
               k_odd_gfs,
               k_even_gfs,
               auxevol_gfs,dt);
    cudaDeviceSynchronize();
    TEST(RK_21)
    cudaDeviceSynchronize();
    apply_bcs(commondata, params, k_even_gfs);
    cudaDeviceSynchronize();
    TEST(RK_22)
    cudaDeviceSynchronize();
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    set_param_constants(params);
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;

    rhs_eval(commondata, params, k_even_gfs, k_odd_gfs);
    cudaDeviceSynchronize();
    
    rk_substep3(params,
               y_n_gfs,
               y_nplus1_running_total_gfs,
               k_odd_gfs,
               k_even_gfs,
               auxevol_gfs,dt);
    cudaDeviceSynchronize();
    apply_bcs(commondata, params, k_odd_gfs);
    cudaDeviceSynchronize();
    TEST(RK_32)
    cudaDeviceSynchronize();
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;

    params_struct *restrict params = &griddata[grid].params;
    set_param_constants(params);
    
    // Not used...codegen baggage?
    // REAL *restrict xx[3];
    // for (int ww = 0; ww < 3; ww++)
    //   xx[ww] = griddata[grid].xx[ww];
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;

    rhs_eval(commondata, params, k_odd_gfs, k_even_gfs);
    cudaDeviceSynchronize();
    
    rk_substep4(params,
               y_n_gfs,
               y_nplus1_running_total_gfs,
               k_odd_gfs,
               k_even_gfs,
               auxevol_gfs,dt);
    cudaDeviceSynchronize();
    apply_bcs(commondata, params, y_n_gfs);
    cudaDeviceSynchronize();    
    TEST(RK_42)
    cudaDeviceSynchronize();    
  }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
