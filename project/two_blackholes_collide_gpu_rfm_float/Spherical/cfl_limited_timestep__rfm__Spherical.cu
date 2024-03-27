#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
#include "../TESTS/TEST_prototypes.h"
/*
 * Output minimum gridspacing ds_min on a Spherical numerical grid.
 */
__global__
void find_local_ds_min_gpu(REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2, REAL *ds_min) {
  int const& Nxx0 = d_params.Nxx0;
  int const& Nxx1 = d_params.Nxx1;
  int const& Nxx2 = d_params.Nxx2;

  REAL const dxx0 = d_params.dxx0;
  REAL const dxx1 = d_params.dxx1;
  REAL const dxx2 = d_params.dxx2;

  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
  const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
  // Thread strides
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;
  
  for(int i2=tid2+NGHOSTS; i2 < Nxx2+NGHOSTS;i2+=stride2) {
    __attribute__((unused)) const REAL xx2 = _xx2[i2];
    for(int i1=tid1+NGHOSTS; i1 < Nxx1+NGHOSTS;i1+=stride1) {
      __attribute__((unused)) const REAL xx1 = _xx1[i1];
      for(int i0=tid0+NGHOSTS;i0 < Nxx0+NGHOSTS;i0+=stride0) {
        __attribute__((unused)) const REAL xx0 = _xx0[i0];
        
        REAL dsmin0, dsmin1, dsmin2;
          
        /*
        *  Original SymPy expressions:
        *  "[dsmin0 = dxx0]"
        *  "[dsmin1 = dxx1*xx0]"
        *  "[dsmin2 = dxx2*xx0*sin(xx1)]"
        */
        dsmin0 = dxx0;
        dsmin1 = dxx1 * xx0;
        dsmin2 = dxx2 * xx0 * sinf(xx1);
        size_t idx = (size_t)IDX3INTERIOR(i0-NGHOSTS, i1-NGHOSTS, i2-NGHOSTS);
        
        // local minimum
        ds_min[idx] = MIN(dsmin0, MIN(dsmin1, dsmin2));
      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}

__global__
void seq_min_gpu(REAL *restrict local_mins, const int N) {
  REAL seq_min = local_mins[0];
  for(int i = 1; i < N; ++i) {
    seq_min = MIN(seq_min, local_mins[i]);
  }
  printf("seq_min: %1.15f\n", seq_min * 0.5);
}

__global__
void compute_ds2(REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2, REAL *ds2) {
  int const& Nxx0 = d_params.Nxx0;
  int const& Nxx1 = d_params.Nxx1;
  int const& Nxx2 = d_params.Nxx2;

  REAL const dxx0 = d_params.dxx0;
  REAL const dxx1 = d_params.dxx1;
  REAL const dxx2 = d_params.dxx2;

  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
  const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
  // Thread strides
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;
  
  for(int i2=tid2+NGHOSTS; i2 < Nxx2+NGHOSTS;i2+=stride2) {
    __attribute__((unused)) const REAL xx2 = _xx2[i2];
    for(int i1=tid1+NGHOSTS; i1 < Nxx1+NGHOSTS;i1+=stride1) {
      __attribute__((unused)) const REAL xx1 = _xx1[i1];
      for(int i0=tid0+NGHOSTS;i0 < Nxx0+NGHOSTS;i0+=stride0) {
        __attribute__((unused)) const REAL xx0 = _xx0[i0];
        
        REAL dsmin0, dsmin1, dsmin2;
          
        /*
        *  Original SymPy expressions:
        *  "[dsmin0 = dxx0]"
        *  "[dsmin1 = dxx1*xx0]"
        *  "[dsmin2 = dxx2*xx0*sin(xx1)]"
        */
        size_t idx = (size_t)IDX3INTERIOR(i0-NGHOSTS, i1-NGHOSTS, i2-NGHOSTS);
        ds2[idx] = dxx2 * xx0 * sinf(REAL (xx1));
      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}

void cfl_limited_timestep__rfm__Spherical(commondata_struct *restrict commondata, params_struct *restrict params, REAL * xx[3],
                                          bc_struct *restrict bcstruct) {
#include "../set_CodeParameters.h"
  const int Nxx_tot = (Nxx0)*(Nxx1)*(Nxx2);
  REAL *ds_gpu;

  // We only loop over a single GF array length
  cudaMalloc(&ds_gpu,sizeof(REAL) * Nxx_tot);
  cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

  dim3 grid(GPU_NGRID0,GPU_NGRID1,GPU_NGRID2);
  dim3 block(GPU_NBLOCK0,GPU_NBLOCK1,GPU_NBLOCK2);

  // compute_ds2<<<grid, block>>>(xx[0], xx[1], xx[2], ds_gpu);
  // TEST_coord_direction(0, ds_gpu, "dsmin2", Nxx_tot);
  
  // Compute dt across grid
  find_local_ds_min_gpu<<<grid, block>>>(xx[0], xx[1], xx[2], ds_gpu);
  cudaCheckErrors(find_local_ds_min_gpu, "cudaKernel find_timestep_gpu failure"); // error checking

  // testing only
  // seq_min_gpu<<<1,1>>>(ds_gpu, Nxx_tot);
  REAL ds_min = find_min(ds_gpu, Nxx_tot);

  commondata->dt = MIN(commondata->dt, ds_min * commondata->CFL_FACTOR);
  cudaFree(ds_gpu);
}