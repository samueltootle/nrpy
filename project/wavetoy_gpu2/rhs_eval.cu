#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
#include <stdexcept>

__host__
void rhs_eval(const commondata_struct *restrict commondata, 
              const params_struct *restrict params, 
              const REAL *restrict in_gfs,
              REAL *restrict rhs_gfs,
              REAL *restrict aux_gfs) {
#define RHS_IMP 1
  int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
  cudaMemcpy(&Nxx_plus_2NGHOSTS0, &params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS1, &params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS2, &params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
#if RHS_IMP == 1
  dim3 block(GPU_NBLOCK0,GPU_NBLOCK1,GPU_NBLOCK2);
  dim3 grid(
    (Nxx_plus_2NGHOSTS0 + GPU_NBLOCK0 - 1) / GPU_NBLOCK0,
    (Nxx_plus_2NGHOSTS1 + GPU_NBLOCK1 - 1) / GPU_NBLOCK1,
    (Nxx_plus_2NGHOSTS2 + GPU_NBLOCK2 - 1) / GPU_NBLOCK2
  );
  rhs_eval_gpu<<<grid,block>>>(commondata, params, in_gfs, rhs_gfs);
  cudaCheckErrors(rhs_eval_gpu, "kernel failed")
  // testcpy(in_gfs);
#elif RHS_IMP == 2
  // Nxx per coordinate direction
  int Nxx0, Nxx1, Nxx2;
  cudaMemcpy(&Nxx0, &params->Nxx0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx1, &params->Nxx1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx2, &params->Nxx2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")

  compute_uu_dDDxx(params, in_gfs, aux_gfs, Nxx0, Nxx1, Nxx2,Nxx_plus_2NGHOSTS0);
  compute_uu_dDDyy(params, in_gfs, aux_gfs, Nxx0, Nxx1, Nxx2,Nxx_plus_2NGHOSTS1);
  compute_uu_dDDzz(params, in_gfs, aux_gfs, Nxx0, Nxx1, Nxx2,Nxx_plus_2NGHOSTS2);

  compute_rhs(params, in_gfs, aux_gfs, rhs_gfs, Nxx0, Nxx1, Nxx2);
#else
  printf("HERE\n");
#endif
}
