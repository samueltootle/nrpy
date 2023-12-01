#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Set initial data to params.time==0 corresponds to the initial data.
 */

__global__
void initial_data_gpu(const commondata_struct in_commondata,
                         params_struct *restrict params, 
                         REAL *restrict in_xx0,
                         REAL *restrict in_xx1,
                         REAL *restrict in_xx2,
                         REAL *restrict in_gfs) {
  const commondata_struct * commondata = &in_commondata;
  #include "set_CodeParameters.h"
  const int tid0  = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1  = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2  = blockIdx.z * blockDim.z + threadIdx.z;
  
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.x * gridDim.z;

  for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2+=stride2) {
    const REAL xx2 = in_xx0[i2];
    for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1+=stride1) {
      const REAL xx1 = in_xx1[i1];
      for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0+=stride0) {
        const REAL xx0 = in_xx2[i0];
        REAL const & xCart0 = xx0;
        REAL const & xCart1 = xx1;
        REAL const & xCart2 = xx2;

        exact_solution_single_Cartesian_point_gpu(commondata, params, 
                                              xCart0, xCart1, xCart2, 
                                              &in_gfs[IDX4(UUGF, i0, i1, i2)],
                                              &in_gfs[IDX4(VVGF, i0, i1, i2)]);
      }
    }
  }
}


void initial_data(const commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
    cudaMemcpy(&Nxx_plus_2NGHOSTS0, &griddata[grid].params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")
    cudaMemcpy(&Nxx_plus_2NGHOSTS1, &griddata[grid].params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")
    cudaMemcpy(&Nxx_plus_2NGHOSTS2, &griddata[grid].params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "memory failed")

    dim3 block_threads(GPU_NBLOCK0,GPU_NBLOCK1,GPU_NBLOCK2);
    dim3 grid_blocks(
      (Nxx_plus_2NGHOSTS0 + GPU_NBLOCK0 - 1) / GPU_NBLOCK0,
      (Nxx_plus_2NGHOSTS1 + GPU_NBLOCK1 - 1) / GPU_NBLOCK1,
      (Nxx_plus_2NGHOSTS2 + GPU_NBLOCK2 - 1) / GPU_NBLOCK2
    );
    initial_data_gpu<<<grid_blocks, block_threads>>>(*commondata, 
                                                      griddata[grid].params, 
                                                      griddata[grid].xx[0], 
                                                      griddata[grid].xx[1], 
                                                      griddata[grid].xx[2], 
                                                      griddata[grid].gridfuncs.y_n_gfs);
    cudaCheckErrors(initial_data_gpu, "initial data failed");
  }
}
