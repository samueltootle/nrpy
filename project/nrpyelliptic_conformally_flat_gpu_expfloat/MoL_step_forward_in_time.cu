#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "expansion_math.h"
/*
 * GPU Kernel: rk_substep_1_gpu.
 * GPU Kernel to compute RK substep 1.
 */
__global__ static void rk_substep_1_gpu(REAL *restrict k_odd_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs, const double dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  float dt_exp, dt_exp_rem;
  split<double>(dt, dt_exp, dt_exp_rem);

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k_odd_gfsL = k_odd_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    constexpr REAL RK_Rational_1_6 = 1.0 / 6.0;
    constexpr REAL RK_Rational_1_2 = 1.0 / 2.0;
    
    float exp_tmp_dt, exp_rem_tmp_dt, exp_tmp_dtR, exp_rem_tmp_dtR;
    split<double>(k_odd_gfsL, exp_tmp_dt, exp_rem_tmp_dt);
    float const exp_tmp_c = exp_tmp_dt;
    float const exp_rem_tmp_c = exp_rem_tmp_dt;
    exp_tmp_dtR = exp_tmp_c;
    exp_rem_tmp_dtR = exp_rem_tmp_c;
    
    // y_nplus1_running_total_gfs[i] = RK_Rational_1_6 * dt * k_odd_gfsL;
    scale_expansion(&exp_tmp_dt, &exp_rem_tmp_dt , RK_Rational_1_6 * dt_exp);
    scale_expansion(&exp_tmp_dtR, &exp_rem_tmp_dtR , RK_Rational_1_6 * dt_exp_rem);
    y_nplus1_running_total_gfs[i] = static_cast<double>(exp_tmp_dt) + static_cast<double>(exp_rem_tmp_dt)
                                  + static_cast<double>(exp_tmp_dtR) + static_cast<double>(exp_rem_tmp_dtR);
    // if(i == 1916329){
    //   printf("%1.15e - %1.15e\n", static_cast<double>(exp_tmp_dt) + static_cast<double>(exp_rem_tmp_dt), static_cast<double>(exp_tmp_dtR) + static_cast<double>(exp_rem_tmp_dtR));
    // }
    double old1 = RK_Rational_1_6 * dt * k_odd_gfsL;

    // Reset to original split values
    exp_tmp_dt = exp_tmp_c;
    exp_rem_tmp_dt = exp_rem_tmp_c;
    exp_tmp_dtR = exp_tmp_c;
    exp_rem_tmp_dtR = exp_rem_tmp_c;

    daxpy(&exp_tmp_dt, &exp_rem_tmp_dt, RK_Rational_1_2 * dt_exp, y_n_gfsL);
    daxpy(&exp_tmp_dtR, &exp_rem_tmp_dtR, RK_Rational_1_2 * dt_exp_rem, y_n_gfsL);

    k_odd_gfs[i] = static_cast<double>(exp_tmp_dt) + static_cast<double>(exp_rem_tmp_dt) 
                 + static_cast<double>(exp_tmp_dtR) + static_cast<double>(exp_rem_tmp_dtR);
    double old2 = RK_Rational_1_2 * dt * k_odd_gfsL + y_n_gfsL;

    // if(i == 1916329){
    //   // float tmp1, tmp2;
    //   // split<double>(old, tmp1, tmp2);
    //   // double tmp3 = static_cast<double>(tmp1) + static_cast<double>(tmp2);
    //   // printf("dt: %1.15e - %1.15e, %1.15e - %1.15e\n", dt, dt_exp, dt_exp_rem, static_cast<double>(dt_exp) + static_cast<double>(dt_exp_rem));
    //   printf("%1.15e, %1.15e - %1.15e, %1.15e\n", y_nplus1_running_total_gfs[i], old1, k_odd_gfs[i], old2);
    // }
  }
}

/*
 * Runge-Kutta function for substep 1.
 */
static void rk_substep_1(params_struct *restrict params, REAL *restrict k_odd_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs,
                         const double dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_gfs, y_n_gfs, y_nplus1_running_total_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_1_gpu failure");
}

/*
 * GPU Kernel: rk_substep_2_gpu.
 * GPU Kernel to compute RK substep 2.
 */
__global__ static void rk_substep_2_gpu(REAL *restrict k_even_gfs, REAL *restrict y_nplus1_running_total_gfs, REAL *restrict y_n_gfs, const double dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const double k_even_gfsL = k_even_gfs[i];
    const double y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
    const double y_n_gfsL = y_n_gfs[i];
    constexpr REAL RK_Rational_1_3 = 1.0 / 3.0;
    constexpr REAL RK_Rational_1_2 = 1.0 / 2.0;
    y_nplus1_running_total_gfs[i] = RK_Rational_1_3 * dt * k_even_gfsL + y_nplus1_running_total_gfsL;
    k_even_gfs[i] = RK_Rational_1_2 * dt * k_even_gfsL + y_n_gfsL;
  }
}

/*
 * Runge-Kutta function for substep 2.
 */
static void rk_substep_2(params_struct *restrict params, REAL *restrict k_even_gfs, REAL *restrict y_nplus1_running_total_gfs, REAL *restrict y_n_gfs,
                         const double dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_2_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_even_gfs, y_nplus1_running_total_gfs, y_n_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_2_gpu failure");
}

/*
 * GPU Kernel: rk_substep_3_gpu.
 * GPU Kernel to compute RK substep 3.
 */
__global__ static void rk_substep_3_gpu(REAL *restrict k_odd_gfs, REAL *restrict y_nplus1_running_total_gfs, REAL *restrict y_n_gfs, const double dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const double k_odd_gfsL = k_odd_gfs[i];
    const double y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
    const double y_n_gfsL = y_n_gfs[i];
    constexpr REAL RK_Rational_1_3 = 1.0 / 3.0;
    y_nplus1_running_total_gfs[i] = RK_Rational_1_3 * dt * k_odd_gfsL + y_nplus1_running_total_gfsL;
    k_odd_gfs[i] = dt * k_odd_gfsL + y_n_gfsL;
  }
}

/*
 * Runge-Kutta function for substep 3.
 */
static void rk_substep_3(params_struct *restrict params, REAL *restrict k_odd_gfs, REAL *restrict y_nplus1_running_total_gfs, REAL *restrict y_n_gfs,
                         const double dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_3_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_gfs, y_nplus1_running_total_gfs, y_n_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_3_gpu failure");
}

/*
 * GPU Kernel: rk_substep_4_gpu.
 * GPU Kernel to compute RK substep 4.
 */
__global__ static void rk_substep_4_gpu(REAL *restrict k_even_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs, const double dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const double k_even_gfsL = k_even_gfs[i];
    const double y_n_gfsL = y_n_gfs[i];
    const double y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
    constexpr REAL RK_Rational_1_6 = 1.0 / 6.0;
    y_n_gfs[i] = RK_Rational_1_6 * dt * k_even_gfsL + y_n_gfsL + y_nplus1_running_total_gfsL;
  }
}

/*
 * Runge-Kutta function for substep 4.
 */
static void rk_substep_4(params_struct *restrict params, REAL *restrict k_even_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs,
                         const double dt) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  const size_t threads_in_x_dir = 32;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  rk_substep_4_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_even_gfs, y_n_gfs, y_nplus1_running_total_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_4_gpu failure");
}

/*
 * Method of Lines (MoL) for "RK4" method: Step forward one full timestep.
 *
 */
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ RK4 }=- Method of Lines timestepping.

  // First set the initial time:
  const double time_start = commondata->time;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    [[maybe_unused]] REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    [[maybe_unused]] REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    [[maybe_unused]] REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    [[maybe_unused]] REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    [[maybe_unused]] REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    [[maybe_unused]] params_struct *restrict params = &griddata[grid].params;
    [[maybe_unused]] const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    [[maybe_unused]] const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, k_odd_gfs);
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0) {
      REAL wavespeed_at_outer_boundary;
      cudaMemcpy(&wavespeed_at_outer_boundary,
                 &auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, Nxx_plus_2NGHOSTS0 - NGHOSTS - 1, NGHOSTS, Nxx_plus_2NGHOSTS2 / 2)], sizeof(REAL),
                 cudaMemcpyDeviceToHost);
      const REAL custom_gridfunctions_wavespeed[2] = {wavespeed_at_outer_boundary, wavespeed_at_outer_boundary};
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata->xx, custom_gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         y_n_gfs, k_odd_gfs);
    }
    rk_substep_1(params, k_odd_gfs, y_n_gfs, y_nplus1_running_total_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_odd_gfs);
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    [[maybe_unused]] REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    [[maybe_unused]] REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    [[maybe_unused]] REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    [[maybe_unused]] REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    [[maybe_unused]] REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    [[maybe_unused]] params_struct *restrict params = &griddata[grid].params;
    [[maybe_unused]] const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    [[maybe_unused]] const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_odd_gfs, k_even_gfs);
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0) {
      REAL wavespeed_at_outer_boundary;
      cudaMemcpy(&wavespeed_at_outer_boundary,
                 &auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, Nxx_plus_2NGHOSTS0 - NGHOSTS - 1, NGHOSTS, Nxx_plus_2NGHOSTS2 / 2)], sizeof(REAL),
                 cudaMemcpyDeviceToHost);
      const REAL custom_gridfunctions_wavespeed[2] = {wavespeed_at_outer_boundary, wavespeed_at_outer_boundary};
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata->xx, custom_gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_odd_gfs, k_even_gfs);
    }
    rk_substep_2(params, k_even_gfs, y_nplus1_running_total_gfs, y_n_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_even_gfs);
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    [[maybe_unused]] REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    [[maybe_unused]] REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    [[maybe_unused]] REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    [[maybe_unused]] REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    [[maybe_unused]] REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    [[maybe_unused]] params_struct *restrict params = &griddata[grid].params;
    [[maybe_unused]] const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    [[maybe_unused]] const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_even_gfs, k_odd_gfs);
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0) {
      REAL wavespeed_at_outer_boundary;
      cudaMemcpy(&wavespeed_at_outer_boundary,
                 &auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, Nxx_plus_2NGHOSTS0 - NGHOSTS - 1, NGHOSTS, Nxx_plus_2NGHOSTS2 / 2)], sizeof(REAL),
                 cudaMemcpyDeviceToHost);
      const REAL custom_gridfunctions_wavespeed[2] = {wavespeed_at_outer_boundary, wavespeed_at_outer_boundary};
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata->xx, custom_gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_even_gfs, k_odd_gfs);
    }
    rk_substep_3(params, k_odd_gfs, y_nplus1_running_total_gfs, y_n_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, k_odd_gfs);
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    [[maybe_unused]] REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    [[maybe_unused]] REAL *restrict y_nplus1_running_total_gfs = griddata[grid].gridfuncs.y_nplus1_running_total_gfs;
    [[maybe_unused]] REAL *restrict k_odd_gfs = griddata[grid].gridfuncs.k_odd_gfs;
    [[maybe_unused]] REAL *restrict k_even_gfs = griddata[grid].gridfuncs.k_even_gfs;
    [[maybe_unused]] REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    [[maybe_unused]] params_struct *restrict params = &griddata[grid].params;
    [[maybe_unused]] const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;
    [[maybe_unused]] const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    [[maybe_unused]] const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, k_odd_gfs, k_even_gfs);
    if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0) {
      REAL wavespeed_at_outer_boundary;
      cudaMemcpy(&wavespeed_at_outer_boundary,
                 &auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, Nxx_plus_2NGHOSTS0 - NGHOSTS - 1, NGHOSTS, Nxx_plus_2NGHOSTS2 / 2)], sizeof(REAL),
                 cudaMemcpyDeviceToHost);
      const REAL custom_gridfunctions_wavespeed[2] = {wavespeed_at_outer_boundary, wavespeed_at_outer_boundary};
      apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata->xx, custom_gridfunctions_wavespeed, gridfunctions_f_infinity,
                                         k_odd_gfs, k_even_gfs);
    }
    rk_substep_4(params, k_even_gfs, y_n_gfs, y_nplus1_running_total_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, y_n_gfs);
  }
  // -={ END k4 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}
