#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include <unistd.h>

__global__
void rk_substep1_gpu(const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                int const N) {
    int const Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
    int const Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
    int const Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr REAL rk_weight = 1./6.;
    constexpr REAL dt_step_factor = 1./2.;

    for(int i=index;i<N;i+=stride) {
        const REAL k_odd_gfsL = k_odd_gfs[i];
        const REAL y_n_gfsL = y_n_gfs[i];
        y_nplus1_running_total_gfs[i] = rk_weight * dt * k_odd_gfsL;
        k_odd_gfs[i] = dt_step_factor *dt * k_odd_gfsL + y_n_gfsL;
    }
}

__host__
void rk_substep1(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) {
    // Compute optimal grid/block configuration for GPU
    int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
    const int N = Nxx_plus_2NGHOSTS0 \
                * Nxx_plus_2NGHOSTS1 \
                * Nxx_plus_2NGHOSTS2 \
                * NUM_EVOL_GFS;
    int block_threads = MIN(GPU_THREADX_MAX, N/32);
    int grid_blocks = (N + block_threads - 1) / block_threads;
    size_t streamid = params->grid_idx % nstreams;

    rk_substep1_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(y_n_gfs,
                                                   y_nplus1_running_total_gfs,
                                                   k_odd_gfs,
                                                   k_even_gfs,
                                                   auxevol_gfs,
                                                   dt, N);
    cudaCheckErrors(rhs_substep1_gpu, "kernel failed")
}

__global__
void rk_substep2_gpu(params_struct *restrict params,
                const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                const REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr REAL rk_weight = 1./3.;
    constexpr REAL dt_step_factor = 1./2.;

    for(int i=index;i<N;i+=stride) {
        const REAL k_even_gfsL = k_even_gfs[i];
        const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
        const REAL y_n_gfsL = y_n_gfs[i];
        y_nplus1_running_total_gfs[i] = rk_weight * dt * k_even_gfsL + y_nplus1_running_total_gfsL;
        k_even_gfs[i] = dt_step_factor * dt * k_even_gfsL + y_n_gfsL;
    }
}

__host__
void rk_substep2(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,REAL const dt) {
    // Compute optimal grid/block configuration for GPU
    int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    const int N = Nxx_plus_2NGHOSTS0 \
                * Nxx_plus_2NGHOSTS1 \
                * Nxx_plus_2NGHOSTS2 \
                * NUM_EVOL_GFS;
    int block_threads = MIN(GPU_THREADX_MAX, N/32);
    int grid_blocks = (N + block_threads - 1) / block_threads;
    size_t streamid = params->grid_idx % nstreams;

    rk_substep2_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(params, 
                                                   y_n_gfs,
                                                   y_nplus1_running_total_gfs,
                                                   k_odd_gfs,
                                                   k_even_gfs,
                                                   auxevol_gfs,
                                                   dt, (size_t) N);
    cudaCheckErrors(rhs_substep2_gpu, "kernel failed")
}

__global__
void rk_substep3_gpu(params_struct *restrict params,
                const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr REAL rk_weight = 1./3.;
    constexpr REAL dt_step_factor = 1.;

    for(int i=index;i<N;i+=stride) {
        const REAL k_odd_gfsL = k_odd_gfs[i];
        const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
        const REAL y_n_gfsL = y_n_gfs[i];
        y_nplus1_running_total_gfs[i] =     \
            rk_weight * dt * k_odd_gfsL     \
            + y_nplus1_running_total_gfsL;
        k_odd_gfs[i] = dt_step_factor * dt * k_odd_gfsL + y_n_gfsL;
    }
}

__host__
void rk_substep3(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) {
    // Compute optimal grid/block configuration for GPU
    int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    const int N = Nxx_plus_2NGHOSTS0 \
                * Nxx_plus_2NGHOSTS1 \
                * Nxx_plus_2NGHOSTS2 \
                * NUM_EVOL_GFS;
    int block_threads = MIN(GPU_THREADX_MAX, N/32);
    int grid_blocks = (N + block_threads - 1) / block_threads;
    size_t streamid = params->grid_idx % nstreams;

    rk_substep3_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(params, 
                                                   y_n_gfs,
                                                   y_nplus1_running_total_gfs,
                                                   k_odd_gfs,
                                                   k_even_gfs,
                                                   auxevol_gfs,
                                                   dt, (size_t) N);
    cudaCheckErrors(rhs_substep3_gpu, "kernel failed")
}

__global__
void rk_substep4_gpu(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                const REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr REAL dt_step_factor = 1./6.;

    for(int i=index;i<N;i+=stride) {
        const REAL k_even_gfsL = k_even_gfs[i];
        const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
        const REAL y_n_gfsL = y_n_gfs[i];
        y_n_gfs[i] = dt_step_factor * dt * k_even_gfsL + y_n_gfsL + y_nplus1_running_total_gfsL;
    }
}

__host__
void rk_substep4(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) {
    // Compute optimal grid/block configuration for GPU
    int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    const int N = Nxx_plus_2NGHOSTS0 \
                * Nxx_plus_2NGHOSTS1 \
                * Nxx_plus_2NGHOSTS2 \
                * NUM_EVOL_GFS;
    int block_threads = MIN(GPU_THREADX_MAX, N / 32);
    int grid_blocks = (N + block_threads - 1) / block_threads;
    size_t streamid = params->grid_idx % nstreams;

    rk_substep4_gpu<<<grid_blocks, block_threads, 0, streams[streamid]>>>(params, 
                                                   y_n_gfs,
                                                   y_nplus1_running_total_gfs,
                                                   k_odd_gfs,
                                                   k_even_gfs,
                                                   auxevol_gfs,
                                                   dt, (size_t) N);
    cudaCheckErrors(rhs_substep4_gpu, "kernel failed")
}