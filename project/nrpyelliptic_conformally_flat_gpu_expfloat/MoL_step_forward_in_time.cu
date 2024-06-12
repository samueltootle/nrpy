#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "expansion_math.h"
/*
 * GPU Kernel: rk_substep_1_gpu.
 * GPU Kernel to compute RK substep 1.
 */
#define vecsize 1
__global__ static void rk_substep_1_double_gpu(REAL *restrict k_odd_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  // const int stride0 = blockDim.x * gridDim.x;
  size_t vec_size = vecsize;
  size_t start_idx = tid0 * vec_size;
  size_t idx = start_idx;

  for (int i = 0; i < vec_size && idx < Ntot; i ++) {
    idx = start_idx + i;
    const REAL k_odd_gfsL = k_odd_gfs[idx];
    const REAL y_n_gfsL = y_n_gfs[idx];
    constexpr REAL RK_Rational_1_6 = 1.0 / 6.0;
    constexpr REAL RK_Rational_1_2 = 1.0 / 2.0;
    
    y_nplus1_running_total_gfs[idx] = RK_Rational_1_6 * dt * k_odd_gfsL;
    k_odd_gfs[idx] = RK_Rational_1_2 * dt * k_odd_gfsL + y_n_gfsL;
  }
}

__global__ static void rk_substep_1_gpu(float *restrict k_odd_gfs, float *restrict y_n_gfs, float *restrict y_nplus1_running_total_gfs, const expansion_math::float2<float> dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  
  constexpr REAL RK_Rational_1_6_ = 1.0 / 6.0;
  static constexpr expansion_math::float2<float> RK_Rational_1_6 = expansion_math::split<float>(RK_Rational_1_6_);

  constexpr REAL RK_Rational_1_2_ = 1.0 / 2.0;
  static constexpr expansion_math::float2<float> RK_Rational_1_2 = expansion_math::split<float>(RK_Rational_1_2_);

  size_t vec_size = 2U * vecsize;
  size_t start_idx = tid0 * vec_size;
  size_t j = start_idx;

  for (int i = 0; i < vec_size && j < 2 * Ntot - 1; i += 2U) {
    j = i + start_idx;
    const expansion_math::float2<float> rhs_exp_c(k_odd_gfs[j], k_odd_gfs[j+1]);
    
    // Original calculataion: RK_Rational_1_6 * dt * k_odd_gfsL;
    // this becomes:
    // = RK_Rational_1_6<expanded> * dt<expanded> * k_odd_gfsL<expanded>
    expansion_math::float2<float> tmp_res  = expansion_math::scale_expansion(
      expansion_math::scale_expansion(rhs_exp_c, RK_Rational_1_6), dt
    );
    y_nplus1_running_total_gfs[j] = tmp_res.value;
    y_nplus1_running_total_gfs[j+1] = tmp_res.remainder;

    const expansion_math::float2<float> y_exp_c(y_n_gfs[j], y_n_gfs[j+1]);

    // Repeat the methodology as above
    // Original calculataion: RK_Rational_1_2 * dt * k_odd_gfsL + y_n_gfsL;
    // this becomes:
    // = RK_Rational_1_2<expanded> * dt<expanded> * k_odd_gfsL<expanded> + y_n_gfsL<expanded>
    tmp_res = expansion_math::grow_expansion(
      y_exp_c, expansion_math::scale_expansion(
        RK_Rational_1_2, expansion_math::scale_expansion(
          dt, rhs_exp_c
        )
      )
    );

    k_odd_gfs[j] = tmp_res.value;
    k_odd_gfs[j+1] = tmp_res.remainder;
  }
}

__global__ static void compare(float *restrict k_odd_exp, float *restrict y_nplus1_exp, REAL *restrict k_odd_gfs, REAL *restrict y_nplus1_running_total_gfs) { 
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;

  size_t vec_size = vecsize;
  size_t i = tid0 * vec_size;
  
  if(i < Ntot) {
    size_t j = 2U * i;
    REAL ref = y_nplus1_running_total_gfs[i];
    REAL cmp = expansion_math::recast_sum<double>(expansion_math::float2<float>(y_nplus1_exp[j], y_nplus1_exp[j+1]));
    if(std::fabs(ref) > 0) {
      REAL rel = std::fabs(1.0 - cmp / ref);
      if(rel > 1e-13) {
        printf("Failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
      }
    }

    ref = k_odd_gfs[i];
    cmp = expansion_math::recast_sum<double>(expansion_math::float2<float>(k_odd_exp[j], k_odd_exp[j+1]));
    if(std::fabs(ref) > 0) {
      REAL rel = std::fabs(1.0 - cmp / ref);
      if(rel > 1e-13) {
        printf("Failure at %d; %1.15e, %1.15e, %1.15e\n", i, ref, cmp, rel);
      }
    }
  }
}

__global__ static void cpy_back(float *restrict gf_in, REAL *restrict gf_out) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const uint tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const uint stride0 = blockDim.x * gridDim.x;

  for (uint i = tid0; i < Ntot; i += stride0) {
    uint j = 2U * i;
    gf_out[i] = expansion_math::recast_sum<double>(
      expansion_math::float2<float>(gf_in[j], gf_in[j+1])
    );
  }
}

__global__ static void decompose_gf(REAL *restrict gf_in, float *restrict gf_out) {
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const uint tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const uint stride0 = blockDim.x * gridDim.x;

  for (uint i = tid0; i < Ntot; i += stride0) {
    uint j = 2U * i;
    const expansion_math::float2<float> gf_exp_c = expansion_math::split<float>(gf_in[i]);
    gf_out[j] = gf_exp_c.value;
    gf_out[j+1] = gf_exp_c.remainder;
  }
}

/*
 * Runge-Kutta function for substep 1.
 */
static void rk_substep_1(params_struct *restrict params, REAL *restrict k_odd_gfs, REAL *restrict y_n_gfs, REAL *restrict y_nplus1_running_total_gfs,
                         const double dt_) {
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  expansion_math::float2<float> dt = expansion_math::split<float>(dt_);
  float* k_odd_exp;
  float* y_n_exp;
  float* y_nplus1_exp;
  
  cudaMalloc(&k_odd_exp, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&y_n_exp, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");
  cudaMalloc(&y_nplus1_exp, sizeof(float) * NUM_EVOL_GFS * Ntot * 2U);
  cudaCheckErrors(malloc, "Malloc failed");

  const size_t threads_in_x_dir = 96;
  const size_t threads_in_y_dir = 1;
  const size_t threads_in_z_dir = 1;
  dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 blocks_per_grid((Ntot + threads_in_x_dir - 1) / threads_in_x_dir / vecsize, 1, 1);
  size_t sm = 0;
  size_t streamid = params->grid_idx % nstreams;
  
  // Decompose REAL into floats and store in the new arrays
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_gfs, k_odd_exp);
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(y_n_gfs, y_n_exp);
  decompose_gf<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(y_nplus1_running_total_gfs, y_nplus1_exp);
  
  // Run exansion kernel
  rk_substep_1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_exp, y_n_exp, y_nplus1_exp, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_1_gpu failure");
  // Run old double kernel
  rk_substep_1_double_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_gfs, y_n_gfs, y_nplus1_running_total_gfs, dt_);
  cudaCheckErrors(cudaKernel, "rk_substep_1_double_gpu failure");
  
  // Ensure solutions are equivalent
  compare<<<blocks_per_grid,threads_per_block,sm,streams[streamid]>>>(k_odd_exp, y_nplus1_exp, k_odd_gfs, y_nplus1_running_total_gfs);
  cudaCheckErrors(cudaKernel, "compare failure");

  // Overwrite data with exansion computed data
  cpy_back<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(k_odd_exp, k_odd_gfs);
  cudaCheckErrors(cudaKernel, "cpyback failure");
  cpy_back<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(y_n_exp, y_n_gfs);
  cudaCheckErrors(cudaKernel, "cpyback failure");
  cpy_back<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(y_nplus1_exp, y_nplus1_running_total_gfs);
  cudaCheckErrors(cudaKernel, "cpyback failure");
  // Free expansion arrays
  cudaFree(k_odd_exp);
  cudaFree(y_n_exp);
  cudaFree(y_nplus1_exp);
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


// __global__ static void rk_substep_1_gpu(float *restrict k_odd_gfs, float *restrict y_n_gfs, float *restrict y_nplus1_running_total_gfs, const expansion_math::float2<float> dt) {
//   const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
//   const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
//   const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
//   const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

//   // Kernel thread/stride setup
//   const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
//   // const int stride0 = blockDim.x * gridDim.x;
  
//   constexpr REAL RK_Rational_1_6_ = 1.0 / 6.0;
//   constexpr expansion_math::float2<float> RK_Rational_1_6 = expansion_math::split<float>(RK_Rational_1_6_);

//   constexpr REAL RK_Rational_1_2_ = 1.0 / 2.0;
//   constexpr expansion_math::float2<float> RK_Rational_1_2 = expansion_math::split<float>(RK_Rational_1_2_);

//   // Number of double precision calculations!
//   constexpr size_t vec_size = vecsize;
//   // Number of single precision elements
//   constexpr size_t ary_size = vecsize * 2U; 

//   const size_t start_idx = tid0 * ary_size;
//   size_t j = start_idx;

//   float data[2][ary_size];
//   for(int i = 0; i < ary_size && j < 2U * Ntot - 1; ++i) {
//     j = i + start_idx;
//     data[0][i] = k_odd_gfs[j];
//     data[1][i] = y_n_gfs[j];
//   }
//   j = start_idx;
//   for (int i = 0; i < ary_size && j < 2U * Ntot - 1; i += 2U) {
//     j = i + start_idx;
//     const expansion_math::float2<float> rhs_exp_c(data[0][i], data[0][i+1]);
//     const expansion_math::float2<float> y_exp_c(data[1][i], data[1][i+1]);
    
//     // Original calculataion: RK_Rational_1_6 * dt * k_odd_gfsL;
//     // this becomes:
//     // = RK_Rational_1_6<expanded> * dt<expanded> * k_odd_gfsL<expanded>
//     expansion_math::float2<float> tmp_res = expansion_math::scale_expansion(
//       dt, expansion_math::scale_expansion(rhs_exp_c, RK_Rational_1_6)
//     );
//     data[1][i] = tmp_res.value;
//     data[1][i+1] = tmp_res.remainder;

//     // Repeat the methodology as above
//     // Original calculataion: RK_Rational_1_2 * dt * k_odd_gfsL + y_n_gfsL;
//     // this becomes:
//     // = RK_Rational_1_2<expanded> * dt<expanded> * k_odd_gfsL<expanded> + y_n_gfsL<expanded>
//     tmp_res = expansion_math::grow_expansion(
//       y_exp_c, expansion_math::scale_expansion(
//         RK_Rational_1_2, expansion_math::scale_expansion(
//           dt, rhs_exp_c
//         )
//       )
//     );

//     data[0][i] = tmp_res.value;
//     data[0][i+1] = tmp_res.remainder;
//     if(j == 1899333){
//       printf("%1.15e, %1.15e - %1.15e, %1.15e\n",data[1][i], data[1][i+1], data[0][i], data[0][i+1]);
//     }
//   }

//   for(int i = 0; i < ary_size && j < 2U * Ntot - 1; ++i) {
//     j = start_idx + i;
//     k_odd_gfs[j] = data[0][i];
//     y_nplus1_running_total_gfs[j] = data[1][i];
//   }
// }