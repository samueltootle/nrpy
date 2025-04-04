#include "../BHaH_defines.h"
/**
 * Kernel: rfm_precompute_free__deallocate_gpu.
 * Kernel to deallocate rfmstruct arrays.
 */
__global__ static void rfm_precompute_free__deallocate_gpu(rfm_struct *restrict rfmstruct) {
  // Temporary parameters
} // END FUNCTION rfm_precompute_free__deallocate_gpu

/**
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void rfm_precompute_free__rfm__Cartesian(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                         rfm_struct *restrict rfmstruct) {
  {

    const size_t threads_in_x_dir = 1;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid(1, 1, 1);
    rfm_precompute_free__deallocate_gpu<<<blocks_per_grid, threads_per_block>>>(rfmstruct);
    cudaCheckErrors(cudaKernel, "rfm_precompute_free__deallocate_gpu failure");
  }
} // END FUNCTION rfm_precompute_free__rfm__Cartesian
