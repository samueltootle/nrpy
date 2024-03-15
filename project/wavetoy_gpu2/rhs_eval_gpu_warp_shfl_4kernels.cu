#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include <stdexcept>
#include <cuda_runtime.h>
#define DEBUG_INDEX 1158762
#if RHS_IMP == 3
__global__ void compute_uu_dDDxx_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs) { 

  const REAL & invdxx0 = d_params.invdxx0;

  const int & Nxx0 = d_params.Nxx0;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = threadIdx.x;
  int tid1  = blockIdx.x * blockDim.y + threadIdx.y;
  int tid2  = blockIdx.y;
  REAL uu_i0m2, uu_i0m1, uu, uu_i0p1, uu_i0p2;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  uint warpx = tid / warpSize;
  uint lanex = tid % warpSize;
  
  // Global memory index - need to shift by ghost zones
  int i = tid0 - warpx * 2 * NGHOSTS;
  int j = tid1 + NGHOSTS;
  int k = tid2 + NGHOSTS;
  int globalIdx = IDX4(UUGF, i, j, k);
  
  uint mask = 0xFFFFFFFFU;
  uu = in_gfs[globalIdx];
  uu_i0m2 = __shfl_up_sync(mask, uu, 2);
  uu_i0m1 = __shfl_up_sync(mask, uu, 1);
  uu_i0p1 = __shfl_down_sync(mask, uu, 1);
  uu_i0p2 = __shfl_down_sync(mask, uu, 2);

  // Warp threads living in the ghost zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && i < Nxx0 + NGHOSTS && lanex > 1 && lanex < 30);

  if(active) {
    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;

    int globalIdx_out = IDX4(UD00, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx0) * (invdxx0)) * (
        FDPart1_Rational_1_12 * (-uu_i0m2 - uu_i0p2) 
      + FDPart1_Rational_4_3  * ( uu_i0m1 + uu_i0p1) 
      + FDPart1tmp0
    );
  
  
  #ifdef DEBUG_RHS
  if(globalIdx == DEBUG_INDEX) {
    printf("uD00: %1.15f - %1.15f - %1.15f - %1.15f - %1.15f - %1.15f\n", 
      aux_gfs[globalIdx_out], uu_i0m2, uu_i0m1, uu, uu_i0p1, uu_i0p2);
  }
  #endif
  }
}

__global__ void compute_uu_dDDyy_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs) { 

  const REAL & invdxx1 = d_params.invdxx1;

  // const int & Nxx0 = d_params.Nxx0;
  const int & Nxx1 = d_params.Nxx1;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = threadIdx.x;
  int tid1  = blockIdx.x * blockDim.y + threadIdx.y;
  int tid2  = blockIdx.y;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  uint warpx = tid / warpSize;
  uint lanex = tid %warpSize;
  
  REAL uu_j0m2, uu_j0m1, uu, uu_j0p1, uu_j0p2;

  // Global array indicies
  int i = tid1 + NGHOSTS;
  int j = tid0 - warpx * 2 * NGHOSTS;;
  int k = tid2 + NGHOSTS;
  int globalIdx = IDX4(UUGF, i, j, k);

  uint mask = 0xFFFFFFFFU;
  uu = in_gfs[globalIdx];
  uu_j0m2 = __shfl_up_sync(mask, uu, 2);
  uu_j0m1 = __shfl_up_sync(mask, uu, 1);
  uu_j0p1 = __shfl_down_sync(mask, uu, 1);
  uu_j0p2 = __shfl_down_sync(mask, uu, 2);

  // Warp threads living in the ghost/halo zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && j < Nxx1 + NGHOSTS && lanex > 1 && lanex < 30);
  
  #ifdef DEBUG_IDX
  if((lanex == 0 || lanex == 30) && (warpx == 3 || warpx == 4) && blockIdx.y == 2)
    printf("(%d, %d, %d) \t- (%u, %u) \t- (%u, %u), (%u, %u) \t- %1.15e - %1.15e - %1.15e - %1.15e - %1.15e\n", 
      i, j, k, warpx, lanex, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, uu_j0m2, uu_j0m1, uu, uu_j0p2, uu_j0p2);
  #endif
  if(active) {
    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;

    int globalIdx_out = IDX4(UD11, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx1) * (invdxx1)) * (
        FDPart1_Rational_1_12 * (-uu_j0m2 - uu_j0p2) 
      + FDPart1_Rational_4_3  * ( uu_j0m1 + uu_j0p1) 
      + FDPart1tmp0
    );
  
  
  #ifdef DEBUG_RHS
  if(globalIdx == DEBUG_INDEX) {
      printf("uD11: %1.15f - %1.15f - %1.15f - %1.15f - %1.15f - %1.15f\n", 
        aux_gfs[globalIdx_out], uu_j0m2, uu_j0m1, uu, uu_j0p1, uu_j0p2);
    }
  #endif
  }
}

__global__ void compute_uu_dDDzz_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs)
{ 
  const REAL & invdxx2 = d_params.invdxx2;

  // const int & Nxx0 = d_params.Nxx0;
  const int & Nxx2 = d_params.Nxx2;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = threadIdx.x;
  int tid1  = blockIdx.x * blockDim.y + threadIdx.y;
  int tid2  = blockIdx.y;
  
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  uint warpx = tid / warpSize;
  uint lanex = tid0 % warpSize;
  
  REAL uu_k0m2, uu_k0m1, uu, uu_k0p1, uu_k0p2;
  uint mask = 0xFFFFFFFFU;

  // Global array indicies
  int i = tid1 + NGHOSTS;
  int j = tid2 + NGHOSTS;
  int k = tid0 - warpx * 2 * NGHOSTS;
  
  int globalIdx = IDX4(UUGF, i, j, k);

  uu = in_gfs[globalIdx];
  uu_k0m2 = __shfl_up_sync(mask, uu, 2);
  uu_k0m1 = __shfl_up_sync(mask, uu, 1);
  uu_k0p1 = __shfl_down_sync(mask, uu, 1);
  uu_k0p2 = __shfl_down_sync(mask, uu, 2);

  // Warp threads living in the ghost/halo zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && k < Nxx2 + NGHOSTS && lanex > 1 && lanex < 30);

  if(active) {
    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;    
    
    int globalIdx_out = IDX4(UD22, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx2) * (invdxx2)) * (
        FDPart1_Rational_1_12 * (-uu_k0m2 - uu_k0p2) 
      + FDPart1_Rational_4_3  * ( uu_k0m1 + uu_k0p1) 
      + FDPart1tmp0
    );
  
  #ifdef DEBUG_RHS
    if(globalIdx == DEBUG_INDEX) {
      printf("uD22: %1.15f - %1.15f - %1.15f - %1.15f - %1.15f - %1.15f\n", 
        aux_gfs[globalIdx_out], uu_k0m2, uu_k0m1, uu, uu_k0p1, uu_k0p2);
    }
  #endif
  }
}

__host__ 
void compute_uu_dDDxx(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS0) {
  // Number of threads in the +/- halos. 32/warp, so 2 warps since it's 1D
  size_t halo_threads = 64;
  
  // The between halo warps, each warp will have a halo of Nghosts that are inactive
  // therefore we need to determine how many threads are needed to break up the interior
  size_t remaining_cells = Nxx_plus_2NGHOSTS0 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  // Note: 4.0 * NGHOSTS = cumulative halo per warp
  size_t interior_threads = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 4.0 * NGHOSTS)) * 32u;
  
  size_t threads_in_x_dir = halo_threads + interior_threads;
  size_t threads_in_y_dir = 1; //1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx1 / threads_in_y_dir, Nxx2, 1);

  compute_uu_dDDxx_gpu<<<grid_blocks, block_threads, 0, stream1>>>(params, in_gfs, aux_gfs);
  cudaCheckErrors(compute_uu_dDDxx_gpu, "kernel failed")
}

__host__ 
void compute_uu_dDDyy(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS1) {
  // Number of threads in the +/- halos. 32/warp, so 2 warps since it's 1D
  size_t halo_warps = 2;
  size_t halo_threads = 2 * 32;
  
  // The between halo warps, each warp will have a halo of Nghosts that are inactive
  // therefore we need to determine how many threads are needed to break up the interior
  size_t remaining_cells = Nxx_plus_2NGHOSTS1 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  size_t interior_warps   = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 4.0 * NGHOSTS));
  size_t interior_threads = interior_warps * 32u;
  
  // threads in logical thread direction (not coordinate direction)
  size_t threads_in_x_dir = halo_threads + interior_threads;
  size_t threads_in_y_dir = 1; //1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx0 / threads_in_y_dir, Nxx2, 1);
  // printf("Block: %u - %u\n", block_threads.x, block_threads.y);
  // printf("Grid: %u - %u\n", grid_blocks.x, grid_blocks.y);

  compute_uu_dDDyy_gpu<<<grid_blocks, block_threads, 0, stream2>>>(params, in_gfs, aux_gfs);
  cudaCheckErrors(compute_uu_dDDyy_gpu, "kernel failed")
}

__host__ 
void compute_uu_dDDzz(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS2) {
  // Number of threads in the +/- halos. 32/warp, so 2 warps since it's 1D
  size_t halo_warps = 2;
  size_t halo_threads = 2 * 32;
  
  // The between halo warps, each warp will have a halo of Nghosts that are inactive
  // therefore we need to determine how many threads are needed to break up the interior
  size_t remaining_cells = Nxx_plus_2NGHOSTS2 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  size_t interior_warps   = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 4.0 * NGHOSTS));
  size_t interior_threads = interior_warps * 32u;
  
  // threads in logical thread direction (not coordinate direction)
  size_t threads_in_x_dir = halo_threads + interior_threads;
  size_t threads_in_y_dir = 1; //1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx0 / threads_in_y_dir, Nxx2, 1);

  compute_uu_dDDzz_gpu<<<grid_blocks, block_threads, 0, stream3>>>(params, in_gfs, aux_gfs);
  cudaCheckErrors(compute_uu_dDDzz_gpu, "kernel failed")
  // printf("\n");
}

__global__ 
void compute_rhs_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 const REAL *restrict in_gfs_derivatives,
                                 REAL *restrict out_gfs) { 

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = threadIdx.x;
  int tid1  = blockIdx.x*blockDim.y + threadIdx.y;
  int tid2  = blockIdx.y;

  // Global memory index - need to shift by ghost zones
  int i = tid0 + NGHOSTS;
  int j = tid1 + NGHOSTS;
  int k = tid2 + NGHOSTS;

  const REAL vv = in_gfs[IDX4(VVGF, i, j, k)];
  const REAL uu_dDD00 = in_gfs_derivatives[IDX4(UD00, i, j, k)];
  const REAL uu_dDD11 = in_gfs_derivatives[IDX4(UD11, i, j, k)];
  const REAL uu_dDD22 = in_gfs_derivatives[IDX4(UD22, i, j, k)];

  const REAL FDPart3tmp0 = ((wavespeed) * (wavespeed));
  out_gfs[IDX4(UUGF, i, j, k)] = vv;
  out_gfs[IDX4(VVGF, i, j, k)] = FDPart3tmp0 * uu_dDD00 + FDPart3tmp0 * uu_dDD11 + FDPart3tmp0 * uu_dDD22;
  #ifdef DEBUG_RHS
  if(IDX4(UUGF, i, j, k) == DEBUG_INDEX) {
    printf("rhs:  %1.15f - %1.15f\n\n",
    out_gfs[IDX4(UUGF, i, j, k)], out_gfs[IDX4(VVGF, i, j, k)]);
  }
  // printf("\ntid1 : %u\n", tid1);
  #endif
}

__host__ 
void compute_rhs(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          const REAL *restrict aux_gfs,
                          REAL *restrict out_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2) {

  // Maximize threads in x - fastest - direction
  size_t threads_in_x_dir = MIN(1024, Nxx0);

  size_t threads_in_y_dir = 1024 / threads_in_x_dir;

  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  
  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(Nxx1 / threads_in_y_dir, Nxx2, 1);
  
  compute_rhs_gpu<<<grid_blocks, block_threads>>>(params, in_gfs, aux_gfs, out_gfs);
  cudaCheckErrors(compute_rhs_gpu, "kernel failed")
}
#endif