#include "../BHaH_defines.h"
/*
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0_gpu.
 * GPU Kernel to precompute metric quantity f0_of_xx0.
 */
__global__ static void rfm_precompute_defines__f0_of_xx0_gpu(REAL *restrict f0_of_xx0, const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f0_of_xx0[i0] = AMAX * (exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA)) / (exp(1.0 / SINHWAA) - exp(-1 / SINHWAA));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__D0_gpu.
 * GPU Kernel to precompute metric quantity f0_of_xx0__D0.
 */
__global__ static void rfm_precompute_defines__f0_of_xx0__D0_gpu(REAL *restrict f0_of_xx0__D0, const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f0_of_xx0__D0[i0] = AMAX * (exp(xx0 / SINHWAA) / SINHWAA + exp(-xx0 / SINHWAA) / SINHWAA) / (exp(1.0 / SINHWAA) - exp(-1 / SINHWAA));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__DD00_gpu.
 * GPU Kernel to precompute metric quantity f0_of_xx0__DD00.
 */
__global__ static void rfm_precompute_defines__f0_of_xx0__DD00_gpu(REAL *restrict f0_of_xx0__DD00, const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f0_of_xx0__DD00[i0] =
        AMAX * (exp(xx0 / SINHWAA) / pow(SINHWAA, 2) - exp(-xx0 / SINHWAA) / pow(SINHWAA, 2)) / (exp(1.0 / SINHWAA) - exp(-1 / SINHWAA));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__DDD000_gpu.
 * GPU Kernel to precompute metric quantity f0_of_xx0__DDD000.
 */
__global__ static void rfm_precompute_defines__f0_of_xx0__DDD000_gpu(REAL *restrict f0_of_xx0__DDD000, const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f0_of_xx0__DDD000[i0] =
        AMAX * (exp(xx0 / SINHWAA) / pow(SINHWAA, 3) + exp(-xx0 / SINHWAA) / pow(SINHWAA, 3)) / (exp(1.0 / SINHWAA) - exp(-1 / SINHWAA));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1_gpu.
 * GPU Kernel to precompute metric quantity f1_of_xx1.
 */
__global__ static void rfm_precompute_defines__f1_of_xx1_gpu(REAL *restrict f1_of_xx1, const REAL *restrict x1) {
  // Temporary parameters
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f1_of_xx1[i1] = sin(xx1);
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1__D1_gpu.
 * GPU Kernel to precompute metric quantity f1_of_xx1__D1.
 */
__global__ static void rfm_precompute_defines__f1_of_xx1__D1_gpu(REAL *restrict f1_of_xx1__D1, const REAL *restrict x1) {
  // Temporary parameters
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f1_of_xx1__D1[i1] = cos(xx1);
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1__DD11_gpu.
 * GPU Kernel to precompute metric quantity f1_of_xx1__DD11.
 */
__global__ static void rfm_precompute_defines__f1_of_xx1__DD11_gpu(REAL *restrict f1_of_xx1__DD11, const REAL *restrict x1) {
  // Temporary parameters
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f1_of_xx1__DD11[i1] = -sin(xx1);
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f2_of_xx0_gpu.
 * GPU Kernel to precompute metric quantity f2_of_xx0.
 */
__global__ static void rfm_precompute_defines__f2_of_xx0_gpu(REAL *restrict f2_of_xx0, const REAL *restrict x0) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f2_of_xx0[i0] =
        sqrt(pow(AMAX, 2) * pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) / pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2) + pow(bScale, 2));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f2_of_xx0__D0_gpu.
 * GPU Kernel to precompute metric quantity f2_of_xx0__D0.
 */
__global__ static void rfm_precompute_defines__f2_of_xx0__D0_gpu(REAL *restrict f2_of_xx0__D0, const REAL *restrict x0) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f2_of_xx0__D0[i0] =
        (1.0 / 2.0) * pow(AMAX, 2) * (2 * exp(xx0 / SINHWAA) / SINHWAA + 2 * exp(-xx0 / SINHWAA) / SINHWAA) *
        (exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA)) /
        (sqrt(pow(AMAX, 2) * pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) / pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2) + pow(bScale, 2)) *
         pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f2_of_xx0__DD00_gpu.
 * GPU Kernel to precompute metric quantity f2_of_xx0__DD00.
 */
__global__ static void rfm_precompute_defines__f2_of_xx0__DD00_gpu(REAL *restrict f2_of_xx0__DD00, const REAL *restrict x0) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const REAL AMAX = d_params.AMAX;
  const REAL SINHWAA = d_params.SINHWAA;
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
    const REAL xx0 = x0[i0];
    f2_of_xx0__DD00[i0] =
        -1.0 / 4.0 * pow(AMAX, 4) * pow(2 * exp(xx0 / SINHWAA) / SINHWAA + 2 * exp(-xx0 / SINHWAA) / SINHWAA, 2) *
            pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) /
            (pow(pow(AMAX, 2) * pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) / pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2) + pow(bScale, 2),
                 3.0 / 2.0) *
             pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 4)) +
        (1.0 / 2.0) * pow(AMAX, 2) * (2 * exp(xx0 / SINHWAA) / pow(SINHWAA, 2) - 2 * exp(-xx0 / SINHWAA) / pow(SINHWAA, 2)) *
            (exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA)) /
            (sqrt(pow(AMAX, 2) * pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) / pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2) + pow(bScale, 2)) *
             pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2)) +
        (1.0 / 2.0) * pow(AMAX, 2) * (exp(xx0 / SINHWAA) / SINHWAA + exp(-xx0 / SINHWAA) / SINHWAA) *
            (2 * exp(xx0 / SINHWAA) / SINHWAA + 2 * exp(-xx0 / SINHWAA) / SINHWAA) /
            (sqrt(pow(AMAX, 2) * pow(exp(xx0 / SINHWAA) - exp(-xx0 / SINHWAA), 2) / pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2) + pow(bScale, 2)) *
             pow(exp(1.0 / SINHWAA) - exp(-1 / SINHWAA), 2));
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f4_of_xx1_gpu.
 * GPU Kernel to precompute metric quantity f4_of_xx1.
 */
__global__ static void rfm_precompute_defines__f4_of_xx1_gpu(REAL *restrict f4_of_xx1, const REAL *restrict x1) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f4_of_xx1[i1] = bScale * sin(xx1);
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f4_of_xx1__D1_gpu.
 * GPU Kernel to precompute metric quantity f4_of_xx1__D1.
 */
__global__ static void rfm_precompute_defines__f4_of_xx1__D1_gpu(REAL *restrict f4_of_xx1__D1, const REAL *restrict x1) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f4_of_xx1__D1[i1] = bScale * cos(xx1);
  }
}
/*
 * GPU Kernel: rfm_precompute_defines__f4_of_xx1__DD11_gpu.
 * GPU Kernel to precompute metric quantity f4_of_xx1__DD11.
 */
__global__ static void rfm_precompute_defines__f4_of_xx1__DD11_gpu(REAL *restrict f4_of_xx1__DD11, const REAL *restrict x1) {
  // Temporary parameters
  const REAL bScale = d_params.bScale;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;

  // Kernel thread/stride setup
  const int tid0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride0 = blockDim.x * gridDim.x;

  for (int i1 = tid0; i1 < Nxx_plus_2NGHOSTS1; i1 += stride0) {
    const REAL xx1 = x1[i1];
    f4_of_xx1__DD11[i1] = -bScale * sin(xx1);
  }
}

/*
 * rfm_precompute_defines: reference metric precomputed lookup arrays: defines
 */
void rfm_precompute_defines__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                            rfm_struct *restrict rfmstruct, REAL *restrict xx[3]) {
#include "../set_CodeParameters.h"
  [[maybe_unused]] const REAL *restrict x0 = xx[0];
  [[maybe_unused]] const REAL *restrict x1 = xx[1];
  [[maybe_unused]] const REAL *restrict x2 = xx[2];
  {
    REAL *restrict f0_of_xx0 = rfmstruct->f0_of_xx0;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 0 % nstreams;
    rfm_precompute_defines__f0_of_xx0_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f0_of_xx0, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f0_of_xx0_gpu failure");
  }
  {
    REAL *restrict f0_of_xx0__D0 = rfmstruct->f0_of_xx0__D0;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 1 % nstreams;
    rfm_precompute_defines__f0_of_xx0__D0_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f0_of_xx0__D0, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f0_of_xx0__D0_gpu failure");
  }
  {
    REAL *restrict f0_of_xx0__DD00 = rfmstruct->f0_of_xx0__DD00;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 2 % nstreams;
    rfm_precompute_defines__f0_of_xx0__DD00_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f0_of_xx0__DD00, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f0_of_xx0__DD00_gpu failure");
  }
  {
    REAL *restrict f0_of_xx0__DDD000 = rfmstruct->f0_of_xx0__DDD000;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 3 % nstreams;
    rfm_precompute_defines__f0_of_xx0__DDD000_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f0_of_xx0__DDD000, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f0_of_xx0__DDD000_gpu failure");
  }
  {
    REAL *restrict f1_of_xx1 = rfmstruct->f1_of_xx1;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 4 % nstreams;
    rfm_precompute_defines__f1_of_xx1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f1_of_xx1, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f1_of_xx1_gpu failure");
  }
  {
    REAL *restrict f1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 5 % nstreams;
    rfm_precompute_defines__f1_of_xx1__D1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f1_of_xx1__D1, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f1_of_xx1__D1_gpu failure");
  }
  {
    REAL *restrict f1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 6 % nstreams;
    rfm_precompute_defines__f1_of_xx1__DD11_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f1_of_xx1__DD11, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f1_of_xx1__DD11_gpu failure");
  }
  {
    REAL *restrict f2_of_xx0 = rfmstruct->f2_of_xx0;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 7 % nstreams;
    rfm_precompute_defines__f2_of_xx0_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f2_of_xx0, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f2_of_xx0_gpu failure");
  }
  {
    REAL *restrict f2_of_xx0__D0 = rfmstruct->f2_of_xx0__D0;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 8 % nstreams;
    rfm_precompute_defines__f2_of_xx0__D0_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f2_of_xx0__D0, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f2_of_xx0__D0_gpu failure");
  }
  {
    REAL *restrict f2_of_xx0__DD00 = rfmstruct->f2_of_xx0__DD00;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 9 % nstreams;
    rfm_precompute_defines__f2_of_xx0__DD00_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f2_of_xx0__DD00, x0);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f2_of_xx0__DD00_gpu failure");
  }
  {
    REAL *restrict f4_of_xx1 = rfmstruct->f4_of_xx1;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 10 % nstreams;
    rfm_precompute_defines__f4_of_xx1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f4_of_xx1, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f4_of_xx1_gpu failure");
  }
  {
    REAL *restrict f4_of_xx1__D1 = rfmstruct->f4_of_xx1__D1;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 11 % nstreams;
    rfm_precompute_defines__f4_of_xx1__D1_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f4_of_xx1__D1, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f4_of_xx1__D1_gpu failure");
  }
  {
    REAL *restrict f4_of_xx1__DD11 = rfmstruct->f4_of_xx1__DD11;

    const size_t threads_in_x_dir = 32;
    const size_t threads_in_y_dir = 1;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    dim3 blocks_per_grid((Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
                         (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
                         (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir);
    size_t sm = 0;
    size_t streamid = 12 % nstreams;
    rfm_precompute_defines__f4_of_xx1__DD11_gpu<<<blocks_per_grid, threads_per_block, sm, streams[streamid]>>>(f4_of_xx1__DD11, x1);
    cudaCheckErrors(cudaKernel, "rfm_precompute_defines__f4_of_xx1__DD11_gpu failure");
  }
}
