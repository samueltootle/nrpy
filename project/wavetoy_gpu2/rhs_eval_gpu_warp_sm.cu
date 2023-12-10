#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include <stdexcept>
#define DEBUG_INDEX 35114
#if RHS_IMP == 3
__global__ void compute_uu_dDDxx_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs) { 

  // extern __shared__ REAL s_f[]; // 2-wide halo for 4th order FD

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
  int i = tid0 - warpx * NGHOSTS;
  int j = tid1 + NGHOSTS;
  int k = tid2 + NGHOSTS;
  int globalIdx = IDX4(UUGF, i, j, k);
  
  uint mask = __activemask();
  uu = in_gfs[globalIdx];
  uu_i0m2 = __shfl_up_sync(mask, uu, 2);
  uu_i0m1 = __shfl_up_sync(mask, uu, 1);
  uu_i0p1 = __shfl_down_sync(mask, uu, 1);
  uu_i0p2 = __shfl_down_sync(mask, uu, 2);

  //   if((lanex == 0 || lanex == 31) && blockIdx.y == 0 && j == 2)
  // // if(blockIdx.y == 0 && blockIdx.x == 0 && k == 0)
  //   printf("(%d, %d, %d) \t- %d \t- %d \t- %d \t- %d \t- %d\n", 
  //     i, j, k, warpx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

  // Warp threads living in the ghost zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && i < Nxx0 + NGHOSTS);
  if(active) {
    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;

    int globalIdx_out = IDX4(UD00, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx0) * (invdxx0)) * (
        FDPart1_Rational_1_12 * (-uu_i0m2 - uu_i0p2) 
      + FDPart1_Rational_4_3 * (uu_i0m1 + uu_i0p1) 
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
  
  REAL uu_j0m2, uu_j0m1, uu, uu_j0p1, uu_j0p2;

  // Global array indicies
  int i = tid1 + NGHOSTS;
  int j = tid0 - warpx * NGHOSTS;;
  int k = tid2 + NGHOSTS;
  int globalIdx = IDX4(UUGF, i, j, k);

  uint mask = __activemask();
  uu = in_gfs[globalIdx];
  uu_j0m2 = __shfl_up_sync(mask, uu, 2);
  uu_j0m1 = __shfl_up_sync(mask, uu, 1);
  uu_j0p1 = __shfl_down_sync(mask, uu, 1);
  uu_j0p2 = __shfl_down_sync(mask, uu, 2);

  // Warp threads living in the ghost zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && j < Nxx1 + NGHOSTS);
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
  uint mask = __activemask();

  // Global array indicies
  int i = tid1 + NGHOSTS;
  int j = tid2 + NGHOSTS;
  int k = tid0 - warpx * NGHOSTS;
  // int tid = threadIdx.x + threadIdx.y * blockDim.x;;
  
  int globalIdx = IDX4(UUGF, i, j, k);
  // if(lanex == 0 && blockIdx.y == 0)
  // // if(blockIdx.y == 0 && blockIdx.x == 0 && k == 0)
  //   printf("(%d, %d, %d) \t- %d \t- %d \t- %d \t- %d \t- %d\n", 
  //     i, j, k, warpx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
  

  
  
  uu = in_gfs[globalIdx];
  uu_k0m2 = __shfl_up_sync(mask, uu, 2);
  uu_k0m1 = __shfl_up_sync(mask, uu, 1);
  uu_k0p1 = __shfl_down_sync(mask, uu, 1);
  uu_k0p2 = __shfl_down_sync(mask, uu, 2);

  // Warp threads living in the ghost zones will be inactive
  // Not sure how bad this is yet...
  // we do this to avoid shared memory
  bool active = (tid0 >= NGHOSTS && k < Nxx2 + NGHOSTS);
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
  
  // The between halo warps, each warp will need to overlap by NGHOST since
  // we avoid using shared memory
  size_t remaining_cells = Nxx_plus_2NGHOSTS0 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  size_t interior_threads = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 2.0 * NGHOSTS)) * 32u;
  
  size_t threads_in_x_dir = halo_threads + interior_threads;
  size_t threads_in_y_dir = 1; //1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx1 / threads_in_y_dir, Nxx2, 1);

  compute_uu_dDDxx_gpu<<<grid_blocks, block_threads>>>(params, in_gfs, aux_gfs);
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
  
  // The between halo warps, each warp will need to overlap by NGHOST since
  // we avoid using shared memory
  size_t remaining_cells = Nxx_plus_2NGHOSTS1 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  size_t interior_warps   = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 2.0 * NGHOSTS));
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

  compute_uu_dDDyy_gpu<<<grid_blocks, block_threads>>>(params, in_gfs, aux_gfs);
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
  
  // The between halo warps, each warp will need to overlap by NGHOST since
  // we avoid using shared memory
  size_t remaining_cells = Nxx_plus_2NGHOSTS2 - halo_threads;
  
  // Over estimate of number of threads needed to process data in the interior
  size_t interior_warps   = (size_t) std::ceil((REAL)remaining_cells / (32.0 - 2.0 * NGHOSTS));
  size_t interior_threads = interior_warps * 32u;
  
  // threads in logical thread direction (not coordinate direction)
  size_t threads_in_x_dir = halo_threads + interior_threads;
  size_t threads_in_y_dir = 1; //1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx0 / threads_in_y_dir, Nxx2, 1);

  compute_uu_dDDzz_gpu<<<grid_blocks, block_threads>>>(params, in_gfs, aux_gfs);
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
  // To ensure coalescence, we want retain reads in the x-direction
  // i.e. the contiguous memory space, based on the standard
  // instruction limits.  Each warp will attempt memory reads up
  // to 128 bytes in a single instruction, in powers of 2,
  // (e.g. 8,16,32,64,128), but this is dependent on the 
  // compute capability of the GPU.  Here we dedicate
  // one thread per data element we read in the x-direction.
  size_t threads_in_x_dir = MIN(1024, Nxx0);

  // Max threads in the y-direction.  Even if we can read
  // the entire tile into shared memory, that doesn't mean
  // we have enough threads per SM to process the entire tile.
  // Therefore we can only have a maximum number of threads in the
  // y direction and each thread will have to compute multiple points.
  size_t threads_in_y_dir = 1024 / threads_in_x_dir;

  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  
  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(Nxx1 / threads_in_y_dir, Nxx2, 1);

  // printf("SM_size : %lu , max: %lu\n\n\n", SM_size, maxMemPerBlock);
  
  compute_rhs_gpu<<<grid_blocks, block_threads>>>(params, in_gfs, aux_gfs, out_gfs);
  cudaCheckErrors(compute_rhs_gpu, "kernel failed")
}
#endif