#include "intrinsics/simd_intrinsics.h"
#include "intrinsics/cuda_intrinsics.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#define STANDALONE

#ifndef REAL
#define REAL double
#endif

#ifndef restrict
#ifdef __cplusplus
#define restrict __restrict__
#endif
#endif

#define MIN(A, B)                                                                                                                                    \
  ({                                                                                                                                                 \
    __typeof__(A) _a = (A);                                                                                                                          \
    __typeof__(B) _b = (B);                                                                                                                          \
    _a < _b ? _a : _b;                                                                                                                               \
  })
#define MAX(A, B)                                                                                                                                    \
  ({                                                                                                                                                 \
    __typeof__(A) _a = (A);                                                                                                                          \
    __typeof__(B) _b = (B);                                                                                                                          \
    _a > _b ? _a : _b;                                                                                                                               \
  })

#ifndef BHAH_TYPEOF
#if __cplusplus >= 2000707L
#define BHAH_TYPEOF(a) decltype(a)
#elif defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define BHAH_TYPEOF(a) __typeof__(a)
#else
#define BHAH_TYPEOF(a)
#endif // END check for GCC, Clang, or C++
#endif // END BHAH_TYPEOF

#ifdef DEBUG
#define cudaCheckErrors(v, msg)                                                                                                                      \
  do {                                                                                                                                               \
    cudaError_t __err = cudaGetLastError();                                                                                                          \
    if (__err != cudaSuccess) {                                                                                                                      \
      fprintf(stderr, "Fatal error: %s %s (%s at %s:%d)\n", #v, msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                 \
      fprintf(stderr, "*** FAILED - ABORTING\n");                                                                                                    \
      exit(1);                                                                                                                                       \
    }                                                                                                                                                \
  } while (0);
#else
#define cudaCheckErrors(v, msg)
#endif

#define BHAH_MALLOC(a, sz)                                                                                                                           \
  do {                                                                                                                                               \
    a = (BHAH_TYPEOF(a))malloc(sz);                                                                                                                  \
  } while (0);
#define BHAH_MALLOC__PtrMember(a, b, sz)                                                                                                             \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      BHAH_MALLOC(a->b, sz);                                                                                                                         \
    }                                                                                                                                                \
  } while (0);

#define BHAH_FREE(a)                                                                                                                                 \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      free((void *)(a));                                                                                                                             \
      (a) = NULL;                                                                                                                                    \
    }                                                                                                                                                \
  } while (0);
#define BHAH_FREE__PtrMember(a, b)                                                                                                                   \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      BHAH_FREE(a->b);                                                                                                                               \
    }                                                                                                                                                \
  } while (0);

#define BHAH_MALLOC_DEVICE(a, sz)                                                                                                                    \
  do {                                                                                                                                               \
    cudaMalloc(&a, sz);                                                                                                                              \
    cudaCheckErrors(cudaMalloc, "Malloc: " #a " failed");                                                                                            \
  } while (0);
#define BHAH_FREE_DEVICE(a)                                                                                                                          \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      cudaFree((void *)(a));                                                                                                                         \
      cudaCheckErrors(cudaFree, "Free: " #a " failed");                                                                                              \
      (a) = nullptr;                                                                                                                                 \
    }                                                                                                                                                \
  } while (0);

#define BHAH_FREE_DEVICE__PtrMember(a, b)                                                                                                            \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      decltype(a->b) tmp_ptr_##b = nullptr;                                                                                                          \
      cudaMemcpy(&tmp_ptr_##b, &a->b, sizeof(void *), cudaMemcpyDeviceToHost);                                                                       \
      if (tmp_ptr_##b) {                                                                                                                             \
        BHAH_FREE_DEVICE(tmp_ptr_##b);                                                                                                               \
        cudaMemcpy(&a->b, &tmp_ptr_##b, sizeof(void *), cudaMemcpyHostToDevice);                                                                     \
      }                                                                                                                                              \
    }                                                                                                                                                \
  } while (0);
#define BHAH_MALLOC_DEVICE__PtrMember(a, b, sz)                                                                                                      \
  do {                                                                                                                                               \
    if (a) {                                                                                                                                         \
      decltype(a->b) tmp_ptr_##b = nullptr;                                                                                                          \
      BHAH_MALLOC_DEVICE(tmp_ptr_##b, sz);                                                                                                           \
      cudaMemcpy(&a->b, &tmp_ptr_##b, sizeof(void *), cudaMemcpyHostToDevice);                                                                       \
    }                                                                                                                                                \
  } while (0);
#define BHAH_DEVICE_SYNC() cudaDeviceSynchronize()

#ifndef MAYBE_UNUSED
#if __cplusplus >= 201703L
#define MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif // END check for GCC, Clang, or NVCC
#endif // END MAYBE_UNUSED

enum { INTERP_SUCCESS, INTERP3D_GENERAL_NULL_PTRS, INTERP3D_GENERAL_INTERP_ORDER_GT_NXX123, INTERP3D_GENERAL_HORIZON_OUT_OF_BOUNDS } error_codes;


// __global__
// void interpolation_3d_general__uniform_src_grid_gpu() {

// }

__global__
void compute_inverse_denominators(const int INTERP_ORDER, REAL *restrict inv_denom) {
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;                                                                                            \
  const int stride0 = blockDim.x * gridDim.x;
  for (int i = tid0; i < INTERP_ORDER; i+=stride0) {
    REAL denom = 1.0;
    for (int j = 0; j < i; j++)
      denom *= (REAL)(i - j);
    for (int j = i + 1; j < INTERP_ORDER; j++)
      denom *= (REAL)(i - j);
    inv_denom[i] = 1.0 / denom; // Divisions are expensive, so we do them only once.
  } // END LOOP: Precompute inverse denominators.
}

__global__
void interpolation_3d_general__uniform_src_grid_host(const int INTERP_ORDER, const REAL src_invdxx012_INTERP_ORDERm1,
  const int NinterpGHOSTS,const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
  const int num_dst_pts, const REAL dst_x0x1x2[][3],
  const int NUM_INTERP_GFS, REAL** src_x0x1x2,
  const REAL src_invdxx0, const REAL src_invdxx1, const REAL src_invdxx2,
  const REAL *restrict inv_denom, REAL** src_gf_ptrs, REAL ** dst_data, const int s_bytes_per_array) {

  extern __shared__ char shared_memory_buffer[];

  REAL* s_diffs_x0 = reinterpret_cast<REAL*>(shared_memory_buffer);
  REAL* s_diffs_x1 = reinterpret_cast<REAL*>(shared_memory_buffer + s_bytes_per_array); // Offset by 1 array size
  REAL* s_diffs_x2 = reinterpret_cast<REAL*>(shared_memory_buffer + 2 * s_bytes_per_array); // Offset by 2 array sizes

  REAL* s_coeff_x0 = reinterpret_cast<REAL*>(shared_memory_buffer + 3 * s_bytes_per_array);
  REAL* s_coeff_x1 = reinterpret_cast<REAL*>(shared_memory_buffer + 4 * s_bytes_per_array);
  REAL* s_coeff_x2 = reinterpret_cast<REAL*>(shared_memory_buffer + 5 * s_bytes_per_array);

  int shared_ary_idx = threadIdx.x * INTERP_ORDER;

  // Perform interpolation for each destination point (x0, x1, x2).
  const REAL xxmin_incl_ghosts0 = src_x0x1x2[0][0];
  const REAL xxmin_incl_ghosts1 = src_x0x1x2[1][0];
  const REAL xxmin_incl_ghosts2 = src_x0x1x2[2][0];
  int error_flag = INTERP_SUCCESS;

  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;                                                                                            \
  const int stride0 = blockDim.x * gridDim.x;

  for (int dst_pt = tid0; dst_pt < num_dst_pts; dst_pt+=stride0) {
    // Extract destination point coordinates.
    const REAL x0_dst = dst_x0x1x2[dst_pt][0];
    const REAL x1_dst = dst_x0x1x2[dst_pt][1];
    const REAL x2_dst = dst_x0x1x2[dst_pt][2];

    // Compute the central index of the interpolation stencil in each dimension.
    int idx_center_x0 = (int)((x0_dst - xxmin_incl_ghosts0) * src_invdxx0 + 0.5);
    int idx_center_x1 = (int)((x1_dst - xxmin_incl_ghosts1) * src_invdxx1 + 0.5);
    int idx_center_x2 = (int)((x2_dst - xxmin_incl_ghosts2) * src_invdxx2 + 0.5);

    // Check if the interpolation stencil goes out of bounds, and adjust indices to prevent memory corruption.
      if ((idx_center_x0 - NinterpGHOSTS < 0) || (idx_center_x0 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS0) || (idx_center_x1 - NinterpGHOSTS < 0) ||
          (idx_center_x1 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS1) || (idx_center_x2 - NinterpGHOSTS < 0) ||
          (idx_center_x2 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS2)) {
        {
          // error_flag = INTERP3D_GENERAL_HORIZON_OUT_OF_BOUNDS;
          // If out of bounds, set indices to NinterpGHOSTS to avoid accessing invalid memory.
          idx_center_x0 = idx_center_x1 = idx_center_x2 = NinterpGHOSTS;
          continue;
        }
      }

    // Compute base indices for interpolation stencil.
    const int base_idx_x0 = idx_center_x0 - NinterpGHOSTS;
    const int base_idx_x1 = idx_center_x1 - NinterpGHOSTS;
    const int base_idx_x2 = idx_center_x2 - NinterpGHOSTS;

    for (int s_j = shared_ary_idx, j = 0; j < INTERP_ORDER; j++, s_j++) {
      s_diffs_x0[s_j] = x0_dst - src_x0x1x2[0][base_idx_x0 + j];
      s_diffs_x1[s_j] = x1_dst - src_x0x1x2[1][base_idx_x1 + j];
      s_diffs_x2[s_j] = x2_dst - src_x0x1x2[2][base_idx_x2 + j];
    } // END LOOP: Compute differences for Lagrange interpolation.

    // Compute the numerator of the Lagrange basis polynomials.
    for (int s_i = shared_ary_idx, i = 0; i < INTERP_ORDER; i++, s_i++) {
      REAL numer_i_x0 = 1.0, numer_i_x1 = 1.0, numer_i_x2 = 1.0;
      for (int j = shared_ary_idx; j < s_i; j++) {
        numer_i_x0 *= s_diffs_x0[j];
        numer_i_x1 *= s_diffs_x1[j];
        numer_i_x2 *= s_diffs_x2[j];
      }
      for (int j = s_i + 1; j < shared_ary_idx + INTERP_ORDER; j++) {
        numer_i_x0 *= s_diffs_x0[j];
        numer_i_x1 *= s_diffs_x1[j];
        numer_i_x2 *= s_diffs_x2[j];
      }
      s_coeff_x0[s_i] = numer_i_x0 * inv_denom[i];
      s_coeff_x1[s_i] = numer_i_x1 * inv_denom[i];
      s_coeff_x2[s_i] = numer_i_x2 * inv_denom[i];
    } // END LOOP: Compute Lagrange basis polynomials.

#define SRC_IDX3(i0, i1, i2) ((i0) + src_Nxx_plus_2NGHOSTS0 * ((i1) + src_Nxx_plus_2NGHOSTS1 * (i2)))

    // For each grid function, compute the interpolated value.
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      REAL sum = 0.0;

      for (int ix2 = 0; ix2 < INTERP_ORDER; ix2++) {
        const int idx2 = base_idx_x2 + ix2;
        const REAL coeff_x2_i = s_coeff_x2[ix2 + shared_ary_idx];
        for (int ix1 = 0; ix1 < INTERP_ORDER; ix1++) {
          const int idx1 = base_idx_x1 + ix1;
          const REAL coeff_x1_i = s_coeff_x1[ix1 + shared_ary_idx];

          int ix0 = 0;
          REAL_CUDA_ARRAY vec_sum = SetZeroCUDA; // Initialize vector sum to zero

          // Precompute the base offset for the current ix2 and ix1
          // This avoids recalculating the constant part inside the vector loop
          const int base_offset = base_idx_x0 + src_Nxx_plus_2NGHOSTS0 * (idx1 + src_Nxx_plus_2NGHOSTS1 * idx2);

          // Vectorized loop using CUDA with FMA, if available
          for (; ix0 < INTERP_ORDER; ix0 += 1) { // Process simd_width doubles at a time
            // Calculate the flat index for the current set of ix0
            // Ensure that ix0 is added correctly to the base_offset
            const int current_idx0 = base_offset + ix0;

            // Load simd_width elements from src_gf_ptrs and coeff_3d
            REAL_CUDA_ARRAY vec_src = src_gf_ptrs[gf][current_idx0];
            REAL_CUDA_ARRAY vec_coeff = s_coeff_x0[ix0 + shared_ary_idx] * coeff_x1_i * coeff_x2_i;
            // Use FMA to multiply src and coeff and add to vec_sum
            vec_sum = FusedMulAddCUDA(vec_src, vec_coeff, vec_sum);
          }

          sum += HorizAddCUDA(vec_sum);
        }
      }

      // Store the interpolated value for this grid function and destination point.
      dst_data[gf][dst_pt] = sum * src_invdxx012_INTERP_ORDERm1;
    } // END LOOP: Over grid functions.
  } // END PARALLEL FOR: Interpolate all destination points.
}

/**
 * Performs 3D Lagrange interpolation from a set of uniform grid points on the source grid to arbitrary destination points.
 *
 * This function interpolates scalar grid functions from a source grid to a set of destination points in the x0, x1, and x2 directions,
 * using Lagrange interpolation of order INTERP_ORDER.
 *
 * @param N_interp_GHOSTS - Number of ghost zones from the center of source point; interpolation order = 2 * N_interp_GHOSTS + 1.
 * @param src_dxx0 - Grid spacing in the x0 direction on the source grid.
 * @param src_dxx1 - Grid spacing in the x1 direction on the source grid.
 * @param src_dxx2 - Grid spacing in the x2 direction on the source grid.
 * @param src_Nxx_plus_2NGHOSTS0 - Dimension of the source grid in x0, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS1 - Dimension of the source grid in x1, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS2 - Dimension of the source grid in x2, including ghost zones.
 * @param NUM_INTERP_GFS - Number of grid functions to interpolate.
 * @param src_x0x1x2 - Arrays of coordinate values for x0, x1, and x2 on the source grid.
 * @param src_gf_ptrs - Array of pointers to source grid functions data.
 * @param num_dst_pts - Number of destination points.
 * @param dst_x0x1x2 - Destination points' coordinates (x0, x1, x2).
 * @param dst_data - Output interpolated data for each grid function at the destination points, of size [NUM_INTERP_GFS][num_dst_pts].
 *
 * @return - Error code indicating success or type of error encountered.
 *
 * @note - The function interpolates each grid function separately and stores the results independently.
 * The source and destination grids are assumed to be uniform in x0, x1, and x2 directions.
 * The function assumes that the destination grid points are within the range of the source grid.
 *
 */
int interpolation_3d_general__uniform_src_grid(const int N_interp_GHOSTS, const REAL src_dxx0, const REAL src_dxx1, const REAL src_dxx2,
                                               const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
                                               const int NUM_INTERP_GFS, REAL ** src_x0x1x2, REAL** src_gf_ptrs,
                                               const int num_dst_pts, const REAL dst_x0x1x2[][3], REAL ** dst_data) {

  // Unpack parameters.
  const int NinterpGHOSTS = N_interp_GHOSTS;
  const int INTERP_ORDER = (2 * NinterpGHOSTS + 1); // Interpolation order (number of points in stencil in each dimension).

  const REAL src_invdxx0 = 1.0 / src_dxx0;
  const REAL src_invdxx1 = 1.0 / src_dxx1;
  const REAL src_invdxx2 = 1.0 / src_dxx2;

  // Compute normalization factor once to avoid repeated expensive pow() operations.
  const REAL src_invdxx012_INTERP_ORDERm1 = pow(src_dxx0 * src_dxx1 * src_dxx2, -(INTERP_ORDER - 1));

  // Check for null pointers in source coordinates and output data.
  // if (src_x0x1x2[0] == NULL || src_x0x1x2[1] == NULL || src_x0x1x2[2] == NULL)
  //   return INTERP3D_GENERAL_NULL_PTRS;
  // for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
  //   if (dst_data[gf] == NULL)
  //     return INTERP3D_GENERAL_NULL_PTRS;
  // }

  // Ensure interpolation order does not exceed grid dimensions.
  if (INTERP_ORDER > src_Nxx_plus_2NGHOSTS0 || INTERP_ORDER > src_Nxx_plus_2NGHOSTS1 || INTERP_ORDER > src_Nxx_plus_2NGHOSTS2)
    return INTERP3D_GENERAL_INTERP_ORDER_GT_NXX123;

  // Precompute inverse denominators for Lagrange interpolation coefficients to optimize performance.
  REAL* inv_denom;
  BHAH_MALLOC_DEVICE(inv_denom, sizeof(REAL) * INTERP_ORDER);
  {
    int threadCount = 32;
    int blockCount = MAX((INTERP_ORDER + threadCount - 1) / threadCount, 1);
    compute_inverse_denominators<<<blockCount, threadCount>>>(INTERP_ORDER, inv_denom);
    cudaCheckErrors(compute_inverse_denominators, "compute_inverse_denominators kernel failed");
  }

  // fix error_flag for cuda
  int error_flag = INTERP_SUCCESS;
  {
    const int threadCount = 32; //MAX(1, 32 % num_dst_pts) * 32;
    const int number_of_arrays = 6;
    const int s_bytes_per_array = (threadCount) *  INTERP_ORDER * sizeof(REAL);
    const int shared_mem_size = s_bytes_per_array * number_of_arrays;
    int blockCount = MAX((num_dst_pts + threadCount - 1) / threadCount, 1);
    interpolation_3d_general__uniform_src_grid_host<<<blockCount, threadCount, shared_mem_size>>>(INTERP_ORDER, src_invdxx012_INTERP_ORDERm1, NinterpGHOSTS,
      src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2,
      num_dst_pts, dst_x0x1x2, NUM_INTERP_GFS, src_x0x1x2, src_invdxx0, src_invdxx1, src_invdxx2,
      inv_denom, src_gf_ptrs, dst_data, s_bytes_per_array);
    cudaCheckErrors(interpolation_3d_general__uniform_src_grid_host, "Interpolation kernel failed");

  }

  return error_flag;
} // END FUNCTION interpolation_3d_general__uniform_src_grid

// #pragma GCC reset_options // Reset compiler optimizations after the function

#ifdef STANDALONE

// Define the number of grid functions to interpolate as a macro for compile-time constant.
#define NUM_INTERP_GFS 4 // Number of grid functions to interpolate.

// Define multiple analytic functions.
__host__ __device__
static inline REAL analytic_function1(REAL x0, REAL x1, REAL x2) { return sin(x0) * cos(x1) * exp(-x2 * x2); }
__host__ __device__
static inline REAL analytic_function2(REAL x0, REAL x1, REAL x2) { return cos(x0) * sin(x1) * exp(-x2); }
__host__ __device__
static inline REAL analytic_function3(REAL x0, REAL x1, REAL x2) { return sin(x0 + x1 + x2); }
__host__ __device__
static inline REAL analytic_function4(REAL x0, REAL x1, REAL x2) { return cos(x1) * sin(x0) + x2 * x2 * x2; }

__global__
void initialize_coordinates_gpu(const int src_Nxx_plus_2NGHOSTS, const int N_interp_GHOSTS, REAL src_dxx, REAL *restrict src_xx, const REAL add_fac = 0.0) {
  // Initialize coordinates in each dimension.
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS; i++) {
    src_xx[i] = add_fac + (i - N_interp_GHOSTS) * (src_dxx);
  }
}

/**
 * Initializes the coordinates for the source grid.
 *
 * @param N_interp_GHOSTS - Number of ghost zones.
 * @param N_x0, N_x1, N_x2 - Number of grid points in x0, x1, x2 directions.
 * @param src_x0x1x2 - Arrays to store coordinate values.
 * @param src_dxx0, src_dxx1, src_dxx2 - Pointers to grid spacings.
 * @param src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2 - Dimensions including ghost zones.
 */
void initialize_coordinates(const int N_interp_GHOSTS, const int N_x0, const int N_x1, const int N_x2, REAL *src_x0x1x2[3], REAL *src_dxx0,
                            REAL *src_dxx1, REAL *src_dxx2, const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1,
                            const int src_Nxx_plus_2NGHOSTS2) {
  // Initialize grid spacings.
  *src_dxx0 = (2.0 * M_PI) / N_x0;
  *src_dxx1 = (2.0 * M_PI) / N_x1;
  *src_dxx2 = (2.0) / N_x2;

  // Allocate memory for coordinate arrays.
  BHAH_MALLOC_DEVICE(src_x0x1x2[0], sizeof(REAL) * src_Nxx_plus_2NGHOSTS0);
  BHAH_MALLOC_DEVICE(src_x0x1x2[1], sizeof(REAL) * src_Nxx_plus_2NGHOSTS1);
  BHAH_MALLOC_DEVICE(src_x0x1x2[2], sizeof(REAL) * src_Nxx_plus_2NGHOSTS2);

  {
    int threadCount = 32;
    int blockCount = MAX((src_Nxx_plus_2NGHOSTS0 + threadCount - 1) / threadCount, 1);
    initialize_coordinates_gpu<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS0, N_interp_GHOSTS, *src_dxx0, src_x0x1x2[0]);
  }
  {
    int threadCount = 32;
    int blockCount = MAX((src_Nxx_plus_2NGHOSTS1 + threadCount - 1) / threadCount, 1);
    initialize_coordinates_gpu<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS1, N_interp_GHOSTS, *src_dxx1, src_x0x1x2[1]);
  }
  {
    int threadCount = 32;
    int blockCount = MAX((src_Nxx_plus_2NGHOSTS2 + threadCount - 1) / threadCount, 1);
    initialize_coordinates_gpu<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS2, N_interp_GHOSTS, *src_dxx2, src_x0x1x2[2], -1.0);
  }
} // END FUNCTION: initialize_coordinates.

/**
 * Initializes the source grid function with an analytic function.
 *
 * @param src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2 - Grid dimensions including ghost zones.
 * @param src_x0x1x2 - Arrays of coordinate values.
 * @param src_gf - Source grid function to initialize.
 * @param func - Analytic function to compute grid values.
 */
enum FUNCTIONS { FUNC1, FUNC2, FUNC3, FUNC4 };
__global__
void initialize_src_gf(const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2, REAL *src_x0x1x2[3],
                       REAL *src_gf, FUNCTIONS func_type) {
  MAYBE_UNUSED const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
  MAYBE_UNUSED const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
  MAYBE_UNUSED const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;

  MAYBE_UNUSED const int stride0 = blockDim.x * gridDim.x;
  MAYBE_UNUSED const int stride1 = blockDim.y * gridDim.y;
  MAYBE_UNUSED const int stride2 = blockDim.z * gridDim.z;
  // Initialize source grid function using the provided analytic function.
  for (int i2 = tid2; i2 < src_Nxx_plus_2NGHOSTS2; i2+=stride2) {
    for (int i1 = tid1; i1 < src_Nxx_plus_2NGHOSTS1; i1+=stride1) {
      for (int i0 = tid0; i0 < src_Nxx_plus_2NGHOSTS0; i0+=stride0) {
        const int idx = i0 + src_Nxx_plus_2NGHOSTS0 * (i1 + src_Nxx_plus_2NGHOSTS1 * i2);
        switch(func_type) {
          case FUNC1:
            src_gf[idx] = analytic_function1(src_x0x1x2[0][i0], src_x0x1x2[1][i1], src_x0x1x2[2][i2]);
            break;
          case FUNC2:
            src_gf[idx] = analytic_function2(src_x0x1x2[0][i0], src_x0x1x2[1][i1], src_x0x1x2[2][i2]);
            break;
          case FUNC3:
            src_gf[idx] = analytic_function3(src_x0x1x2[0][i0], src_x0x1x2[1][i1], src_x0x1x2[2][i2]);
            break;
          case FUNC4:
            src_gf[idx] = analytic_function4(src_x0x1x2[0][i0], src_x0x1x2[1][i1], src_x0x1x2[2][i2]);
            break;
        }
        // src_gf[idx] = func(src_x0x1x2[0][i0], src_x0x1x2[1][i1], src_x0x1x2[2][i2]);
      } // END LOOP: Over i0.
    } // END LOOP: Over i1.
  } // END LOOP: Over i2.
} // END FUNCTION: initialize_src_gf.

void print_device_info() {
  cudaDeviceProp deviceProp;
    int dev = 0;
    cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, dev);
    if (error_id != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties for device %d returned %s\n",
                dev, cudaGetErrorString(error_id));
    }

    printf("\n--- Device %d: %s ---\n", dev, deviceProp.name);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Max dynamic shared memory per block (opt-in): %zu bytes\n", deviceProp.sharedMemPerBlockOptin); // More relevant for configurable SM
    printf("  Total global memory: %zu bytes (%.2f GB)\n",
            deviceProp.totalGlobalMem, (double)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("  Max registers per block: %d\n", deviceProp.regsPerBlock);
    // Add more properties if you're interested
}

int main() {
  // print_device_info();
  const int N_interp_GHOSTS = 4;                    // For 9th order interpolation.
  const int INTERP_ORDER = 2 * N_interp_GHOSTS + 1; // 9th order.
  const int num_resolutions = 3;                    // Number of resolutions to test.
  const int num_dst_pts = 3000000;                  // Number of destination points.

  int N_x0_arr[num_resolutions];
  int N_x1_arr[num_resolutions];
  int N_x2_arr[num_resolutions];
  REAL h_arr[num_resolutions];
  REAL error_L2_norm[NUM_INTERP_GFS][num_resolutions];

  // Initialize the resolutions.
  N_x0_arr[0] = 16;
  N_x1_arr[0] = 16;
  N_x2_arr[0] = 16;

  N_x0_arr[1] = 32;
  N_x1_arr[1] = 32;
  N_x2_arr[1] = 32;

  N_x0_arr[2] = 64;
  N_x1_arr[2] = 64;
  N_x2_arr[2] = 64;

  // Allocate memory for destination points.
  REAL(*dst_pts)[3] = (REAL(*)[3])malloc(sizeof(REAL) * num_dst_pts * 3);
  if (dst_pts == NULL) {
    fprintf(stderr, "Memory allocation failed for destination points.\n");
    return EXIT_FAILURE;
  }
  REAL(*dst_pts_device)[3];
  BHAH_MALLOC_DEVICE(dst_pts_device, sizeof(REAL) * num_dst_pts * 3);

  // Allocate exact solution arrays for each grid function.
  REAL *f_exact[NUM_INTERP_GFS];
  for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    // f_exact[gf] = (REAL *)malloc(sizeof(REAL) * num_dst_pts);
    BHAH_MALLOC(f_exact[gf], sizeof(REAL) * num_dst_pts);
    if (f_exact[gf] == NULL) {
      fprintf(stderr, "Memory allocation failed for f_exact[%d].\n", gf);
      // Free previously allocated memory before exiting.
      for (int g = 0; g < gf; g++) {
        BHAH_FREE(f_exact[g]);
      }
      BHAH_FREE(dst_pts);
      BHAH_FREE_DEVICE(dst_pts_device);
      return EXIT_FAILURE;
    }
  } // END LOOP: Allocate exact solution arrays.

  // Loop over resolutions.
  for (int res = 0; res < num_resolutions; res++) {
    int N_x0 = N_x0_arr[res];
    int N_x1 = N_x1_arr[res];
    int N_x2 = N_x2_arr[res];

    h_arr[res] = (N_x0 > 0) ? ((REAL)(2.0 * M_PI) / N_x0) : 0.0; // Assuming src_dxx0 == src_dxx1 == src_dxx2.

    // Define source grid dimensions including ghost zones.
    int src_Nxx_plus_2NGHOSTS0 = N_x0 + 2 * N_interp_GHOSTS;
    int src_Nxx_plus_2NGHOSTS1 = N_x1 + 2 * N_interp_GHOSTS;
    int src_Nxx_plus_2NGHOSTS2 = N_x2 + 2 * N_interp_GHOSTS;

    // Allocate and initialize coordinate arrays.
    REAL *src_x0x1x2[3];
    REAL src_dxx0_val, src_dxx1_val, src_dxx2_val;

    initialize_coordinates(N_interp_GHOSTS, N_x0, N_x1, N_x2, src_x0x1x2, &src_dxx0_val, &src_dxx1_val, &src_dxx2_val, src_Nxx_plus_2NGHOSTS0,
                           src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2);
    BHAH_DEVICE_SYNC();
    // Compute safe domain for destination points
    REAL x0_min_safe;
    cudaMemcpy(&x0_min_safe, &src_x0x1x2[0][N_interp_GHOSTS], sizeof(REAL), cudaMemcpyDeviceToHost);
    x0_min_safe += 1e-6; // Add a small offset to avoid boundary issues.

    REAL x0_max_safe;
    cudaMemcpy(&x0_max_safe, &src_x0x1x2[0][src_Nxx_plus_2NGHOSTS0 - N_interp_GHOSTS - 1], sizeof(REAL), cudaMemcpyDeviceToHost);
    x0_max_safe -= 1e-6; // Subtract a small offset to avoid boundary issues.

    REAL x1_min_safe;
    cudaMemcpy(&x1_min_safe, &src_x0x1x2[1][N_interp_GHOSTS], sizeof(REAL), cudaMemcpyDeviceToHost);
    x1_min_safe += 1e-6; // Add a small offset to avoid boundary issues.

    REAL x1_max_safe;
    cudaMemcpy(&x1_max_safe, &src_x0x1x2[1][src_Nxx_plus_2NGHOSTS1 - N_interp_GHOSTS - 1], sizeof(REAL), cudaMemcpyDeviceToHost);
    x1_max_safe -= 1e-6; // Subtract a small offset to avoid boundary issues.

    REAL x2_min_safe;
    cudaMemcpy(&x2_min_safe, &src_x0x1x2[2][N_interp_GHOSTS], sizeof(REAL), cudaMemcpyDeviceToHost);
    x2_min_safe += 1e-6; // Add a small offset to avoid boundary issues.

    REAL x2_max_safe;
    cudaMemcpy(&x2_max_safe, &src_x0x1x2[2][src_Nxx_plus_2NGHOSTS2 - N_interp_GHOSTS - 1], sizeof(REAL), cudaMemcpyDeviceToHost);
    x2_max_safe -= 1e-6; // Subtract a small offset to avoid boundary issues.

    // Seed the random number generator.
    srand(42 + res); // Use different seed for each resolution if desired
    BHAH_DEVICE_SYNC();
    // Generate random destination points and compute the exact function values for each grid function.
    for (int i = 0; i < num_dst_pts; i++) {
      REAL x0 = x0_min_safe + ((REAL)rand() / RAND_MAX) * (x0_max_safe - x0_min_safe);
      REAL x1 = x1_min_safe + ((REAL)rand() / RAND_MAX) * (x1_max_safe - x1_min_safe);
      REAL x2 = x2_min_safe + ((REAL)rand() / RAND_MAX) * (x2_max_safe - x2_min_safe);
      dst_pts[i][0] = x0;
      dst_pts[i][1] = x1;
      dst_pts[i][2] = x2;
      // Compute exact values for each grid function.
      f_exact[0][i] = analytic_function1(x0, x1, x2);
      f_exact[1][i] = analytic_function2(x0, x1, x2);
      f_exact[2][i] = analytic_function3(x0, x1, x2);
      f_exact[3][i] = analytic_function4(x0, x1, x2);
    }
    cudaMemcpy(dst_pts_device, dst_pts, sizeof(REAL) * num_dst_pts * 3, cudaMemcpyHostToDevice);
    BHAH_DEVICE_SYNC();

    // Allocate and initialize grid functions.
    REAL * src_gf[NUM_INTERP_GFS];
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      BHAH_MALLOC_DEVICE(src_gf[gf], sizeof(REAL) * src_Nxx_plus_2NGHOSTS0 * src_Nxx_plus_2NGHOSTS1 * src_Nxx_plus_2NGHOSTS2);
      if (src_gf[gf] == NULL) {
        fprintf(stderr, "Memory allocation failed for src_gf[%d].\n", gf);
        // Free previously allocated memory before exiting.
        for (int g = 0; g < gf; g++) {
          BHAH_FREE_DEVICE(src_gf[g]);
        }
        for (int dim = 0; dim < 3; dim++) {
          BHAH_FREE_DEVICE(src_x0x1x2[dim]);
        }
        BHAH_FREE(dst_pts);
        BHAH_FREE_DEVICE(dst_pts_device);
        for (int g = 0; g < NUM_INTERP_GFS; g++) {
          BHAH_FREE(f_exact[g]);
        }
        return EXIT_FAILURE;
      }
    } // END LOOP: Allocate grid functions.
    REAL ** src_x0x1x2_ptrs;
    cudaMalloc(&src_x0x1x2_ptrs, 3 * sizeof(REAL*));
    cudaMemcpy(src_x0x1x2_ptrs, src_x0x1x2, 3 * sizeof(REAL*), cudaMemcpyHostToDevice);
    BHAH_DEVICE_SYNC();
    // Initialize each grid function with its respective analytic function.
    {
      int threadCount = 32;
      dim3 blockCount = (
        MAX((src_Nxx_plus_2NGHOSTS0 + threadCount - 1) / threadCount, 1),
        src_Nxx_plus_2NGHOSTS1,
        src_Nxx_plus_2NGHOSTS2
      );

      initialize_src_gf<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_x0x1x2_ptrs, src_gf[0], FUNC1);
      cudaCheckErrors(initialize_src_gf, "initialize_src_gf1 kernel failed");
      initialize_src_gf<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_x0x1x2_ptrs, src_gf[1], FUNC2);
      cudaCheckErrors(initialize_src_gf, "initialize_src_gf2 kernel failed");
      initialize_src_gf<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_x0x1x2_ptrs, src_gf[2], FUNC3);
      cudaCheckErrors(initialize_src_gf, "initialize_src_gf3 kernel failed");
      initialize_src_gf<<<blockCount, threadCount>>>(src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_x0x1x2_ptrs, src_gf[3], FUNC4);
      cudaCheckErrors(initialize_src_gf, "initialize_src_gf4 kernel failed");
    }

    // Create an array of pointers to src_gf.
    // const REAL *restrict src_gf_ptrs[NUM_INTERP_GFS] = {src_gf[0], src_gf[1], src_gf[2], src_gf[3]};
    REAL ** src_gf_ptrs;
    cudaMalloc(&src_gf_ptrs, NUM_INTERP_GFS * sizeof(REAL*));
    cudaMemcpy(src_gf_ptrs, src_gf, NUM_INTERP_GFS * sizeof(REAL*), cudaMemcpyHostToDevice);
    BHAH_DEVICE_SYNC();


    // Allocate memory for interpolated data for all grid functions.
    REAL *dst_data[NUM_INTERP_GFS];
    REAL *dst_data_device[NUM_INTERP_GFS];
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      // dst_data[gf] = (REAL *)malloc(sizeof(REAL) * num_dst_pts);
      BHAH_MALLOC(dst_data[gf], sizeof(REAL) * num_dst_pts);
      BHAH_MALLOC_DEVICE(dst_data_device[gf], sizeof(REAL) * num_dst_pts);
      if (dst_data[gf] == NULL) {
        fprintf(stderr, "Memory allocation failed for dst_data[%d].\n", gf);
        // Free previously allocated memory before exiting.
        for (int g = 0; g < gf; g++) {
          BHAH_FREE(dst_data[g]);
          BHAH_FREE_DEVICE(dst_data_device[g]);
        }
        for (int g = 0; g < NUM_INTERP_GFS; g++) {
          BHAH_FREE_DEVICE(src_gf[g]);
        }
        for (int dim = 0; dim < 3; dim++) {
          BHAH_FREE_DEVICE(src_x0x1x2[dim]);
        }
        BHAH_FREE(dst_pts);
        BHAH_FREE_DEVICE(dst_pts_device);
        for (int g = 0; g < NUM_INTERP_GFS; g++) {
          BHAH_FREE(f_exact[g]);
        }
        BHAH_FREE_DEVICE(src_gf_ptrs);
        return EXIT_FAILURE;
      }
    }
    REAL ** dst_data_ptrs;
    cudaMalloc(&dst_data_ptrs, NUM_INTERP_GFS * sizeof(REAL*));
    cudaMemcpy(dst_data_ptrs, dst_data_device, NUM_INTERP_GFS * sizeof(REAL*), cudaMemcpyHostToDevice);
    BHAH_DEVICE_SYNC();

    // Call the interpolation function.
    int error_code = interpolation_3d_general__uniform_src_grid(N_interp_GHOSTS, src_dxx0_val, src_dxx1_val, src_dxx2_val, src_Nxx_plus_2NGHOSTS0,
                                                                src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, NUM_INTERP_GFS, src_x0x1x2_ptrs,
                                                                src_gf_ptrs, num_dst_pts, dst_pts_device, dst_data_ptrs);
    BHAH_DEVICE_SYNC();
    // if (error_code != INTERP_SUCCESS) {
    //   fprintf(stderr, "Interpolation error code: %d\n", error_code);
    //   // Free allocated memory before exiting.
    //   for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    //     BHAH_FREE_DEVICE(src_gf[gf]);
    //     BHAH_FREE(dst_data[gf]);
    //     BHAH_FREE_DEVICE(dst_data_device[gf]);
    //   }
    //   for (int dim = 0; dim < 3; dim++) {
    //     BHAH_FREE_DEVICE(src_x0x1x2[dim]);
    //   }
    //   for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    //     BHAH_FREE(f_exact[gf]);
    //   }
    //   BHAH_FREE(dst_pts);
    //   BHAH_FREE_DEVICE(dst_pts_device);
    //   BHAH_FREE_DEVICE(src_gf_ptrs);
    //   return error_code;
    // }



    // Compute the L2 norm of the error for each grid function.
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      cudaMemcpy(dst_data[gf], dst_data_device[gf], sizeof(REAL) * num_dst_pts, cudaMemcpyDeviceToHost);
      BHAH_DEVICE_SYNC();
      REAL error_sum = 0.0;
      for (int i = 0; i < num_dst_pts; i++) {
        REAL error = dst_data[gf][i] - f_exact[gf][i];
        error_sum += error * error;
      }
      error_L2_norm[gf][res] = sqrt(error_sum / num_dst_pts);
    } // END LOOP: Compute L2 norms.

    // Output the error for each grid function.
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      printf("Resolution %d: N_x0 = %d, N_x1 = %d, N_x2 = %d, h = %.5e, Grid Function %d, L2 error = %.5e\n", res, N_x0, N_x1, N_x2, h_arr[res],
             gf + 1, error_L2_norm[gf][res]);
    } // END LOOP: Output errors.

    // Free allocated memory for this resolution.
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      BHAH_FREE_DEVICE(src_gf[gf]);
      BHAH_FREE(dst_data[gf]);
      BHAH_FREE_DEVICE(dst_data_device[gf]);
    }
    for (int dim = 0; dim < 3; dim++) {
      BHAH_FREE_DEVICE(src_x0x1x2[dim]);
    }
    BHAH_FREE_DEVICE(src_gf_ptrs);
    BHAH_FREE_DEVICE(dst_data_ptrs);

  } // END LOOP: Over resolutions.

  // Compute the observed order of convergence for each grid function.
  for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    for (int res = 1; res < num_resolutions; res++) {
      REAL observed_order = log(error_L2_norm[gf][res - 1] / error_L2_norm[gf][res]) / log(h_arr[res - 1] / h_arr[res]);
      printf("Observed order of convergence for Grid Function %d between resolutions %d and %d: %.2f\n", gf + 1, res - 1, res, observed_order);
    }
    // Expected order is INTERP_ORDER (since we are using 9th order interpolation).
    printf("Expected order of convergence for Grid Function %d: %d\n", gf + 1, INTERP_ORDER);
  } // END LOOP: Compute observed orders.

  // Clean up.
  for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    BHAH_FREE(f_exact[gf]);
  }
  BHAH_FREE(dst_pts);
  BHAH_FREE_DEVICE(dst_pts_device);


  return 0;
} // END FUNCTION: main.

#endif // STANDALONE

