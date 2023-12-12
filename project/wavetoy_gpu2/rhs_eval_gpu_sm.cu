#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include <stdexcept>
#define DEBUG_INDEX 35114
#if RHS_IMP == 2
__global__ void compute_uu_dDDxx_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs)
{ 

  extern __shared__ REAL s_f[]; // 2-wide halo for 4th order FD

  const REAL & invdxx0 = d_params.invdxx0;

  const int & Nxx0 = d_params.Nxx0;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = threadIdx.x;
  int tid1  = blockIdx.x * blockDim.y + threadIdx.y;
  int tid2  = blockIdx.y;
  int si = tid0 + NGHOSTS; // local i for shared memory access + halo offset
  int sj = threadIdx.y; // local j for shared memory access
  
  // s_f stores pencils in linear memory so we need a
  // shared memory index
  int sm_idx = sj * Nxx_plus_2NGHOSTS0 + si;

  // Global memory index - need to shift by ghost zones
  int i = tid0 + NGHOSTS;
  int j = tid1 + NGHOSTS;
  int k = tid2 + NGHOSTS;
  int globalIdx = IDX4(UUGF, i, j, k);

  s_f[sm_idx] = in_gfs[globalIdx];

  __syncthreads();

  // fill in SM ghost zones
  if (tid0 < NGHOSTS) {
    int temp_idx = IDX4(UUGF, i - NGHOSTS, j, k);
    s_f[sm_idx - NGHOSTS] = in_gfs[temp_idx];
    
    temp_idx = IDX4(UUGF, i + Nxx0, j, k);
    s_f[sm_idx+Nxx0] = in_gfs[temp_idx];
  }

  __syncthreads();
  const REAL uu_i0m2 = s_f[sm_idx - 2];
  const REAL uu_i0m1 = s_f[sm_idx - 1];
  const REAL uu      = s_f[sm_idx    ];
  const REAL uu_i0p1 = s_f[sm_idx + 1];
  const REAL uu_i0p2 = s_f[sm_idx + 2];

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

__global__ void compute_uu_dDDyy_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs)
{ 

  extern __shared__ REAL s_f[];

  const REAL & invdxx1 = d_params.invdxx1;

  // const int & Nxx0 = d_params.Nxx0;
  const int & Nxx1 = d_params.Nxx1;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = blockIdx.x*blockDim.x + threadIdx.x;
  int tid1  = threadIdx.y;
  int tid2  = blockIdx.y;
  int si = threadIdx.x; // local i for shared memory access

  // Global array indicies
  int i = tid0 + NGHOSTS;
  int k = tid2 + NGHOSTS;

  for (int j = tid1 + NGHOSTS; j < Nxx1 + NGHOSTS; j += blockDim.y) {
    int sj = j;
    
    // s_f stores pencils in linear memory so we need a
    // shared memory index such that the contiguous elements
    // are now the "y" data.
    int sm_idx = si * Nxx_plus_2NGHOSTS1 + sj;

    int globalIdx = IDX4(UUGF, i, j, k);

    s_f[sm_idx] = in_gfs[globalIdx];
    // if(sm_idx == 2 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   printf("sf: (%u, %u) %u, %1.15f - gf: %1.15f\n\n", 
    //     blockIdx.x, blockIdx.y, sm_idx, s_f[sm_idx], in_gfs[globalIdx]);
    // }
  }

  int sj = tid1 + NGHOSTS;
  int sm_idx = si * Nxx_plus_2NGHOSTS1 + sj;
  int j = sj;
  __syncthreads();
  
  // fill in SM ghost zones
  if (tid1 < NGHOSTS) {
    uint temp_idx1 = IDX4(UUGF, i, j - NGHOSTS, k);
    s_f[sm_idx - NGHOSTS]  = in_gfs[temp_idx1];
    
    uint temp_idx2 = IDX4(UUGF, i, j + Nxx1, k);
    s_f[sm_idx + Nxx1] = in_gfs[temp_idx2];
  }

  __syncthreads();

  for (int j = tid1 + NGHOSTS; j < Nxx1 + NGHOSTS; j += blockDim.y) {
    int sj = j;
    int sm_idx = si * Nxx_plus_2NGHOSTS1 + sj;

    const REAL uu_j0m2 = s_f[sm_idx - 2];
    const REAL uu_j0m1 = s_f[sm_idx - 1];
    const REAL uu      = s_f[sm_idx    ];
    const REAL uu_j0p1 = s_f[sm_idx + 1];
    const REAL uu_j0p2 = s_f[sm_idx + 2];

    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;

    int globalIdx_out = IDX4(UD11, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx1) * (invdxx1)) * (
        FDPart1_Rational_1_12 * (-uu_j0m2 - uu_j0p2) 
      + FDPart1_Rational_4_3  * ( uu_j0m1 + uu_j0p1) 
      + FDPart1tmp0
    );
    int globalIdx = IDX4(UUGF, i, j, k);
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

  extern __shared__ REAL s_f[]; // 2-wide halo for 4th order FD

  const REAL & invdxx2 = d_params.invdxx2;

  // const int & Nxx0 = d_params.Nxx0;
  const int & Nxx2 = d_params.Nxx2;

  const int & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  // Local tile indices - not global thread indicies
  int tid0  = blockIdx.x*blockDim.x + threadIdx.x;
  int tid1  = threadIdx.y;
  int tid2  = blockIdx.y;
  int si = threadIdx.x; // local i for shared memory access

  // Global array indicies
  int i = tid0 + NGHOSTS;
  int j = tid2 + NGHOSTS;

  for (int k = tid1 + NGHOSTS; k < Nxx2 + NGHOSTS; k += blockDim.y) {
    int sk = k;
    
    // s_f stores pencils in linear memory so we need a
    // shared memory index such that the contiguous elements
    // are now the "y" data.
    int sm_idx = si * Nxx_plus_2NGHOSTS2 + sk;

    int globalIdx = IDX4(UUGF, i, j, k);

    s_f[sm_idx] = in_gfs[globalIdx];
  }

  int sk = tid1 + NGHOSTS;
  int sm_idx = si * Nxx_plus_2NGHOSTS2 + sk;
  int k = sk;
  __syncthreads();
  
  // fill in SM ghost zones
  if (tid1 < NGHOSTS) {
    uint temp_idx1 = IDX4(UUGF, i, j, k - NGHOSTS);
    s_f[sm_idx-NGHOSTS]  = in_gfs[temp_idx1];
    uint temp_idx2 = IDX4(UUGF, i, j, k + Nxx2);
    // printf("%d - %d : %d - %d \n", temp_idx1, sm_idx-4, temp_idx2, sm_idx + Nxx1 + NGHOSTS);
    s_f[sm_idx + Nxx2] = in_gfs[temp_idx2];
  }

  __syncthreads();

  for (int k = tid1 + NGHOSTS; k < Nxx2 + NGHOSTS; k += blockDim.y) {
    int sk = k;
    int sm_idx = si * Nxx_plus_2NGHOSTS2 + sk;

    const REAL uu_k0m2 = s_f[sm_idx - 2];
    const REAL uu_k0m1 = s_f[sm_idx - 1];
    const REAL uu      = s_f[sm_idx    ];
    const REAL uu_k0p1 = s_f[sm_idx + 1];
    const REAL uu_k0p2 = s_f[sm_idx + 2];

    const REAL FDPart1tmp0 = -FDPart1_Rational_5_2 * uu;

    int globalIdx_out = IDX4(UD22, i, j, k);
    aux_gfs[globalIdx_out] = ((invdxx2) * (invdxx2)) * (
        FDPart1_Rational_1_12 * (-uu_k0m2 - uu_k0p2) 
      + FDPart1_Rational_4_3  * ( uu_k0m1 + uu_k0p1) 
      + FDPart1tmp0
    );
    int globalIdx = IDX4(UUGF, i, j, k);
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
  size_t threads_in_x_dir = Nxx0;
  size_t threads_in_y_dir = 1024 / threads_in_x_dir;
  size_t threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
  dim3 grid_blocks(Nxx1 / threads_in_y_dir, Nxx2, 1);

  // Determine dynamic shared memory size in bytes
  const size_t SM_size = threads_in_y_dir * Nxx_plus_2NGHOSTS0 * sizeof(REAL);
  
  // Fetch maximum shared memory size per block
  const int device = 0; //assumes single GPU
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  const size_t maxMemPerBlock = deviceProp.sharedMemPerBlock;
  
  if(SM_size > maxMemPerBlock) {
    throw std::runtime_error("Grid is too large for current GPU shared memory restrictions");
  }

  compute_uu_dDDxx_gpu<<<grid_blocks, block_threads, SM_size, stream1>>>(params, in_gfs, aux_gfs);
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
  const int device = 0; //assumes single GPU
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  // To ensure coalescence, we want retain reads in the x-direction
  // i.e. the contiguous memory space, based on the standard
  // instruction limits.  Each warp will attempt memory reads up
  // to 128 bytes in a single instruction, in powers of 2,
  // (e.g. 8,16,32,64,128), but this is dependent on the 
  // compute capability of the GPU.  Here we dedicate
  // one thread per data element we read in the x-direction.
  size_t threads_in_x_dir = 128 / sizeof(REAL);

  // Max threads in the y-direction.  Even if we can read
  // the entire tile into shared memory, that doesn't mean
  // we have enough threads per SM to process the entire tile.
  // Therefore we can only have a maximum number of threads in the
  // y direction and each thread will have to compute multiple points.
  size_t threads_in_y_dir = deviceProp.maxThreadsPerBlock / threads_in_x_dir;

  // The tile size should attempt to avoid halo data,
  // i.e. zones of data that are read by two or more blocks
  // into shared memory.
  // For the simple case of cartesian coordinates, the
  // derivatives are 1D, so this shouldn't be a problem
  // so long as the 1D Grid size * threads_in_x_dir
  // will fit into shared memory.
  // Determine dynamic shared memory size in bytes:
  const size_t SM_size = threads_in_x_dir * Nxx_plus_2NGHOSTS1 * sizeof(REAL);
  
  // For now we just throw an exception based on maximum shared memory size per block
  // but it should be possible to decide a better course of action - e.g. using halos
  // instead
  const size_t maxMemPerBlock = deviceProp.sharedMemPerBlock;
  if(SM_size > maxMemPerBlock) {
    throw std::runtime_error("compute_uu_dDDyy: Grid is too large for current GPU shared memory restrictions");
  }

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, 1);
  
  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(Nxx0 / threads_in_x_dir, Nxx2, 1);

  // printf("SM: %u - (%u, %u, %u) - (%u, %u, %u)\n", SM_size,
  //   block_threads.x, block_threads.y, block_threads.z, 
  //   grid_blocks.x, grid_blocks.y, grid_blocks.z);
  
  compute_uu_dDDyy_gpu<<<grid_blocks, block_threads, SM_size, stream2>>>(params, in_gfs, aux_gfs);
  cudaCheckErrors(compute_uu_dDDyy_gpu, "kernel failed")
  // cudaDeviceSynchronize();
  // printf("SM lengh: %u\n",threads_in_x_dir * Nxx_plus_2NGHOSTS1);
}

__host__ 
void compute_uu_dDDzz(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS2) {
  const int device = 0; //assumes single GPU
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  
  // To ensure coalescence, we want retain reads in the x-direction
  // i.e. the contiguous memory space, based on the standard
  // instruction limits.  Each warp will attempt memory reads up
  // to 128 bytes in a single instruction, in powers of 2,
  // (e.g. 8,16,32,64,128), but this is dependent on the 
  // compute capability of the GPU.  Here we dedicate
  // one thread per data element we read in the x-direction.
  size_t threads_in_x_dir = 128 / sizeof(REAL);

  // Max threads in the y-direction.  Even if we can read
  // the entire tile into shared memory, that doesn't mean
  // we have enough threads per SM to process the entire tile.
  // Therefore we can only have a maximum number of threads in the
  // y direction and each thread will have to compute multiple points.
  size_t threads_in_z_dir = deviceProp.maxThreadsPerBlock / threads_in_x_dir;

  // The tile size should attempt to avoid halo data,
  // i.e. zones of data that are read by two or more blocks
  // into shared memory.
  // For the simple case of cartesian coordinates, the
  // derivatives are 1D, so this shouldn't be a problem
  // so long as the 1D Grid size * threads_in_x_dir
  // will fit into shared memory.
  // Determine dynamic shared memory size in bytes:
  const size_t SM_size = threads_in_x_dir * Nxx_plus_2NGHOSTS2 * sizeof(REAL);
  
  // For now we just throw an exception based on maximum shared memory size per block
  // but it should be possible to decide a better course of action - e.g. using halos
  // instead
  const size_t maxMemPerBlock = deviceProp.sharedMemPerBlock;
  if(SM_size > maxMemPerBlock) {
    throw std::runtime_error("compute_uu_dDDzz: Grid is too large for current GPU shared memory restrictions");
  }

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_z_dir, 1);
  
  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(Nxx0 / threads_in_x_dir, Nxx1, 1);

  // printf("SM_size : %lu , max: %lu\n\n\n", SM_size, maxMemPerBlock);
  
  compute_uu_dDDzz_gpu<<<grid_blocks, block_threads, SM_size, stream3>>>(params, in_gfs, aux_gfs);
  cudaCheckErrors(compute_uu_dDDzz_gpu, "kernel failed")
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