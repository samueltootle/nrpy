#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * GPU Kernel: rk_substep_1_gpu.
 * GPU Kernel to compute RK substep 1.
 */
__global__ static void rk_substep_1_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs,
                                        const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_1_10 = 1.0 / 10.0;
    next_y_input_gfs[i] = RK_Rational_1_10 * dt * k1_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 1.
 */
static void rk_substep_1(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs,
                         const REAL dt) {
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
  rk_substep_1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, y_n_gfs, next_y_input_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_1_gpu failure");
}

/**
 * GPU Kernel: rk_substep_2_gpu.
 * GPU Kernel to compute RK substep 2.
 */
__global__ static void rk_substep_2_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict y_n_gfs,
                                        REAL *restrict next_y_input_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_2_81 = 2.0 / 81.0;
    static constexpr REAL RK_Rational_20_81 = 20.0 / 81.0;
    next_y_input_gfs[i] = RK_Rational_20_81 * dt * k2_gfsL - RK_Rational_2_81 * dt * k1_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 2.
 */
static void rk_substep_2(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict y_n_gfs,
                         REAL *restrict next_y_input_gfs, const REAL dt) {
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
  rk_substep_2_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k2_gfs, y_n_gfs, next_y_input_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_2_gpu failure");
}

/**
 * GPU Kernel: rk_substep_3_gpu.
 * GPU Kernel to compute RK substep 3.
 */
__global__ static void rk_substep_3_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                                        REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_270_343 = 270.0 / 343.0;
    static constexpr REAL RK_Rational_615_1372 = 615.0 / 1372.0;
    static constexpr REAL RK_Rational_1053_1372 = 1053.0 / 1372.0;
    next_y_input_gfs[i] = RK_Rational_1053_1372 * dt * k3_gfsL - RK_Rational_270_343 * dt * k2_gfsL + RK_Rational_615_1372 * dt * k1_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 3.
 */
static void rk_substep_3(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs, REAL *restrict y_n_gfs,
                         REAL *restrict next_y_input_gfs, const REAL dt) {
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
  rk_substep_3_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k2_gfs, k3_gfs, y_n_gfs, next_y_input_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_3_gpu failure");
}

/**
 * GPU Kernel: rk_substep_4_gpu.
 * GPU Kernel to compute RK substep 4.
 */
__global__ static void rk_substep_4_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                                        REAL *restrict k4_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL k4_gfsL = k4_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_54_55 = 54.0 / 55.0;
    static constexpr REAL RK_Rational_3243_5500 = 3243.0 / 5500.0;
    static constexpr REAL RK_Rational_4998_17875 = 4998.0 / 17875.0;
    static constexpr REAL RK_Rational_50949_71500 = 50949.0 / 71500.0;
    next_y_input_gfs[i] = RK_Rational_3243_5500 * dt * k1_gfsL + RK_Rational_4998_17875 * dt * k4_gfsL + RK_Rational_50949_71500 * dt * k3_gfsL -
                          RK_Rational_54_55 * dt * k2_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 4.
 */
static void rk_substep_4(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs, REAL *restrict k4_gfs,
                         REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs, const REAL dt) {
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
  rk_substep_4_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k2_gfs, k3_gfs, k4_gfs, y_n_gfs, next_y_input_gfs,
                                                                                  dt);
  cudaCheckErrors(cudaKernel, "rk_substep_4_gpu failure");
}

/**
 * GPU Kernel: rk_substep_5_gpu.
 * GPU Kernel to compute RK substep 5.
 */
__global__ static void rk_substep_5_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                                        REAL *restrict k4_gfs, REAL *restrict k5_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs,
                                        const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL k4_gfsL = k4_gfs[i];
    const REAL k5_gfsL = k5_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_26492_37125 = 26492.0 / 37125.0;
    static constexpr REAL RK_Rational_24206_37125 = 24206.0 / 37125.0;
    static constexpr REAL RK_Rational_72_55 = 72.0 / 55.0;
    static constexpr REAL RK_Rational_338_459 = 338.0 / 459.0;
    static constexpr REAL RK_Rational_2808_23375 = 2808.0 / 23375.0;
    next_y_input_gfs[i] = -RK_Rational_24206_37125 * dt * k4_gfsL - RK_Rational_26492_37125 * dt * k1_gfsL + RK_Rational_2808_23375 * dt * k3_gfsL +
                          RK_Rational_338_459 * dt * k5_gfsL + RK_Rational_72_55 * dt * k2_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 5.
 */
static void rk_substep_5(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs, REAL *restrict k4_gfs,
                         REAL *restrict k5_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs, const REAL dt) {
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
  rk_substep_5_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k2_gfs, k3_gfs, k4_gfs, k5_gfs, y_n_gfs,
                                                                                  next_y_input_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_5_gpu failure");
}

/**
 * GPU Kernel: rk_substep_6_gpu.
 * GPU Kernel to compute RK substep 6.
 */
__global__ static void rk_substep_6_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                                        REAL *restrict k4_gfs, REAL *restrict k5_gfs, REAL *restrict k6_gfs, REAL *restrict y_n_gfs,
                                        REAL *restrict next_y_input_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL k4_gfsL = k4_gfs[i];
    const REAL k5_gfsL = k5_gfs[i];
    const REAL k6_gfsL = k6_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_24117_31603 = 24117.0 / 31603.0;
    static constexpr REAL RK_Rational_5225_1836 = 5225.0 / 1836.0;
    static constexpr REAL RK_Rational_35_11 = 35.0 / 11.0;
    static constexpr REAL RK_Rational_3925_4056 = 3925.0 / 4056.0;
    static constexpr REAL RK_Rational_5561_2376 = 5561.0 / 2376.0;
    static constexpr REAL RK_Rational_899983_200772 = 899983.0 / 200772.0;
    next_y_input_gfs[i] = -RK_Rational_24117_31603 * dt * k3_gfsL - RK_Rational_35_11 * dt * k2_gfsL + RK_Rational_3925_4056 * dt * k6_gfsL -
                          RK_Rational_5225_1836 * dt * k5_gfsL + RK_Rational_5561_2376 * dt * k1_gfsL + RK_Rational_899983_200772 * dt * k4_gfsL +
                          y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 6.
 */
static void rk_substep_6(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs, REAL *restrict k4_gfs,
                         REAL *restrict k5_gfs, REAL *restrict k6_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs, const REAL dt) {
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
  rk_substep_6_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k2_gfs, k3_gfs, k4_gfs, k5_gfs, k6_gfs, y_n_gfs,
                                                                                  next_y_input_gfs, dt);
  cudaCheckErrors(cudaKernel, "rk_substep_6_gpu failure");
}

/**
 * GPU Kernel: rk_substep_7_gpu.
 * GPU Kernel to compute RK substep 7.
 */
__global__ static void rk_substep_7_gpu(const size_t streamid, REAL *restrict k1_gfs, REAL *restrict k3_gfs, REAL *restrict k4_gfs,
                                        REAL *restrict k5_gfs, REAL *restrict k6_gfs, REAL *restrict k7_gfs, REAL *restrict y_n_gfs, const REAL dt) {
  const int Nxx_plus_2NGHOSTS0 = d_params[streamid].Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params[streamid].Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params[streamid].Nxx_plus_2NGHOSTS2;
  const int Ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i = tid0; i < Ntot; i += stride0) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL k4_gfsL = k4_gfs[i];
    const REAL k5_gfsL = k5_gfs[i];
    const REAL k6_gfsL = k6_gfs[i];
    const REAL k7_gfsL = k7_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    static constexpr REAL RK_Rational_3_50 = 3.0 / 50.0;
    static constexpr REAL RK_Rational_395_3672 = 395.0 / 3672.0;
    static constexpr REAL RK_Rational_785_2704 = 785.0 / 2704.0;
    static constexpr REAL RK_Rational_821_10800 = 821.0 / 10800.0;
    static constexpr REAL RK_Rational_19683_71825 = 19683.0 / 71825.0;
    static constexpr REAL RK_Rational_175273_912600 = 175273.0 / 912600.0;
    y_n_gfs[i] = RK_Rational_175273_912600 * dt * k4_gfsL + RK_Rational_19683_71825 * dt * k3_gfsL + RK_Rational_395_3672 * dt * k5_gfsL +
                 RK_Rational_3_50 * dt * k7_gfsL + RK_Rational_785_2704 * dt * k6_gfsL + RK_Rational_821_10800 * dt * k1_gfsL + y_n_gfsL;
  }
}

/**
 * Runge-Kutta function for substep 7.
 */
static void rk_substep_7(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k3_gfs, REAL *restrict k4_gfs, REAL *restrict k5_gfs,
                         REAL *restrict k6_gfs, REAL *restrict k7_gfs, REAL *restrict y_n_gfs, const REAL dt) {
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
  rk_substep_7_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(streamid, k1_gfs, k3_gfs, k4_gfs, k5_gfs, k6_gfs, k7_gfs, y_n_gfs,
                                                                                  dt);
  cudaCheckErrors(cudaKernel, "rk_substep_7_gpu failure");
}

/**
 * Method of Lines (MoL) for "DP5alt" method: Step forward one full timestep.
 *
 */
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ DP5alt }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, k1_gfs);
    rk_substep_1(params, k1_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000006e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k2_gfs);
    rk_substep_2(params, k1_gfs, k2_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 2.22222222222222210e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k3_gfs);
    rk_substep_3(params, k1_gfs, k2_gfs, k3_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 4.28571428571428548e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k4_gfs);
    rk_substep_4(params, k1_gfs, k2_gfs, k3_gfs, k4_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k4 substep }=-

  // -={ START k5 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.99999999999999978e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k5_gfs);
    rk_substep_5(params, k1_gfs, k2_gfs, k3_gfs, k4_gfs, k5_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k5 substep }=-

  // -={ START k6 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 8.00000000000000044e-01 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k6_gfs);
    rk_substep_6(params, k1_gfs, k2_gfs, k3_gfs, k4_gfs, k5_gfs, k6_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k6 substep }=-

  // -={ START k7 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    // Temporary timelevel & AUXEVOL gridfunctions:
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict k4_gfs = griddata[grid].gridfuncs.k4_gfs;
    MAYBE_UNUSED REAL *restrict k5_gfs = griddata[grid].gridfuncs.k5_gfs;
    MAYBE_UNUSED REAL *restrict k6_gfs = griddata[grid].gridfuncs.k6_gfs;
    MAYBE_UNUSED REAL *restrict k7_gfs = griddata[grid].gridfuncs.k7_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k7_gfs);
    rk_substep_7(params, k1_gfs, k3_gfs, k4_gfs, k5_gfs, k6_gfs, k7_gfs, y_n_gfs, commondata->dt);
    if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
      apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, y_n_gfs);
  }
  // -={ END k7 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  //   so here we set time based on the iteration number.
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Finally, increment the timestep n:
  commondata->nn++;
}