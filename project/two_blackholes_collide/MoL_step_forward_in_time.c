#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#ifdef GPU_TESTS
#include "trusted_data_dump/trusted_data_dump_prototypes.h"
#endif
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
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    params_struct *restrict params = &griddata[grid].params;
    const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;

    Ricci_eval(commondata, params, rfmstruct, y_n_gfs, auxevol_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, auxevol_gfs, "auxgfs", "post-ricci-rk1", NUM_AUXEVOL_GFS);
    #endif
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, k_odd_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "rhs-rk1", NUM_EVOL_GFS);
    #endif
    // return;
    
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0)
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata[grid].xx, gridfunctions_wavespeed, gridfunctions_f_infinity, y_n_gfs,
                                         k_odd_gfs);
    // for (int(i) = 0; (i) < Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * 24; (i)++) {
    LOOP_ALL_GFS_GPS(i) {
      const REAL k_odd_gfsL = k_odd_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      y_nplus1_running_total_gfs[i] = (1.0 / 6.0) * commondata->dt * k_odd_gfsL;
      k_odd_gfs[i] = (1.0 / 2.0) * commondata->dt * k_odd_gfsL + y_n_gfsL;
      // printf("%d - %d\n", i, IDX4(VETU2GF, 34, 18 , 18));
      // if(i == IDX4(VETU2GF, 34, 18 , 18))
      //   printf("RK1: %1.15e - %1.15e - %1.15e\n", k_odd_gfsL, y_nplus1_running_total_gfs[i], k_odd_gfs[i]);
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk1", NUM_EVOL_GFS);
    dump_gf_array(grid, params, y_nplus1_running_total_gfs, "ynp1", "post-rk1", NUM_EVOL_GFS);
    #endif
    // for(int i = 0; i < NUM_EVOL_GFS; ++i)
    //   print_var(k_odd_gfs, IDX4(i, 34, 18 , 18));
    // printf("**************************_rk1\n");
    
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0) {
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_odd_gfs);
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk1-bcs", NUM_EVOL_GFS);
    #endif
    enforce_detgammabar_equals_detgammahat(commondata, params, rfmstruct, k_odd_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk1-enforce", NUM_EVOL_GFS);
    #endif
  }
  // return;
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    params_struct *restrict params = &griddata[grid].params;
    const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;

    Ricci_eval(commondata, params, rfmstruct, k_odd_gfs, auxevol_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, auxevol_gfs, "auxgfs", "post-ricci-rk2", NUM_AUXEVOL_GFS);
    #endif
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_odd_gfs, k_even_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_even_gfs, "keven", "rhs-rk2", NUM_EVOL_GFS);
    #endif
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0)
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata[grid].xx, gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_odd_gfs, k_even_gfs);
    LOOP_ALL_GFS_GPS(i) {
      const REAL k_even_gfsL = k_even_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      y_nplus1_running_total_gfs[i] = (1.0 / 3.0) * commondata->dt * k_even_gfsL + y_nplus1_running_total_gfsL;
      k_even_gfs[i] = (1.0 / 2.0) * commondata->dt * k_even_gfsL + y_n_gfsL;
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_even_gfs, "keven", "post-rk2", NUM_EVOL_GFS);
    dump_gf_array(grid, params, y_nplus1_running_total_gfs, "ynp1", "post-rk2", NUM_EVOL_GFS);
    #endif
    // for(int i = 0; i < NUM_EVOL_GFS; ++i)
      // print_var(k_even_gfs, IDX4(i, 34, 18 , 18));
    // printf("**************************_rk2\n");
    
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0) {
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_even_gfs);
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_even_gfs, "keven", "post-rk2-bcs", NUM_EVOL_GFS);
    #endif
    enforce_detgammabar_equals_detgammahat(commondata, params, rfmstruct, k_even_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_even_gfs, "keven", "post-rk2-enforce", NUM_EVOL_GFS);
    #endif
  }
  // return;
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    params_struct *restrict params = &griddata[grid].params;
    const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;

    Ricci_eval(commondata, params, rfmstruct, k_even_gfs, auxevol_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, auxevol_gfs, "auxgfs", "post-ricci-rk2", NUM_AUXEVOL_GFS);
    #endif
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_even_gfs, k_odd_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "rhs-rk3", NUM_EVOL_GFS);
    #endif
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0)
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata[grid].xx, gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_even_gfs, k_odd_gfs);
    LOOP_ALL_GFS_GPS(i) {
      const REAL k_odd_gfsL = k_odd_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      y_nplus1_running_total_gfs[i] = (1.0 / 3.0) * commondata->dt * k_odd_gfsL + y_nplus1_running_total_gfsL;
      k_odd_gfs[i] = commondata->dt * k_odd_gfsL + y_n_gfsL;
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk3", NUM_EVOL_GFS);
    dump_gf_array(grid, params, y_nplus1_running_total_gfs, "ynp1", "post-rk3", NUM_EVOL_GFS);
    #endif
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_odd_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk3-bcs", NUM_EVOL_GFS);
    #endif
    enforce_detgammabar_equals_detgammahat(commondata, params, rfmstruct, k_odd_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_odd_gfs, "kodd", "post-rk3-enforce", NUM_EVOL_GFS);
    #endif
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    params_struct *restrict params = &griddata[grid].params;
    const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;

    Ricci_eval(commondata, params, rfmstruct, k_odd_gfs, auxevol_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, auxevol_gfs, "auxgfs", "post-ricci-rk2", NUM_AUXEVOL_GFS);
    #endif
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_odd_gfs, k_even_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, k_even_gfs, "keven", "rhs-rk4", NUM_EVOL_GFS);
    #endif
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0)
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata[grid].xx, gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_odd_gfs, k_even_gfs);
    LOOP_ALL_GFS_GPS(i) {
      const REAL k_even_gfsL = k_even_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      y_n_gfs[i] = (1.0 / 6.0) * commondata->dt * k_even_gfsL + y_n_gfsL + y_nplus1_running_total_gfsL;
    }
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, y_n_gfs, "yn", "post-rk4", NUM_EVOL_GFS);
    #endif
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, y_n_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, y_n_gfs, "yn", "post-rk4-bcs", NUM_EVOL_GFS);
    #endif
    enforce_detgammabar_equals_detgammahat(commondata, params, rfmstruct, y_n_gfs);
    #ifdef GPU_TESTS
    dump_gf_array(grid, params, y_n_gfs, "yn", "post-rk4-enforce", NUM_EVOL_GFS);
    #endif
  }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}