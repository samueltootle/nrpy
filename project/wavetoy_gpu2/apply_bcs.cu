#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"

// Declare boundary condition FACE_UPDATE macro,
//          which updates a single face of the 3D grid cube
//          using quadratic polynomial extrapolation.
const int MAXFACE = -1;
const int NUL = +0;
const int MINFACE = +1;
#define FACE_UPDATE(which_gf, i0min, i0max, i1min, i1max, i2min, i2max, FACEX0, FACEX1, FACEX2)                                                      \
  for (int i2 = i2min; i2 < i2max; i2++)                                                                                                             \
    for (int i1 = i1min; i1 < i1max; i1++)                                                                                                           \
      for (int i0 = i0min; i0 < i0max; i0++) {                                                                                                       \
        gfs[IDX4(which_gf, i0, i1, i2)] = +3.0 * gfs[IDX4(which_gf, i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2)] -                            \
                                          3.0 * gfs[IDX4(which_gf, i0 + 2 * FACEX0, i1 + 2 * FACEX1, i2 + 2 * FACEX2)] +                             \
                                          1.0 * gfs[IDX4(which_gf, i0 + 3 * FACEX0, i1 + 3 * FACEX1, i2 + 3 * FACEX2)];                              \
      }

__global__
void update_face(int which_gf, const params_struct *restrict params, REAL *restrict gfs,
                   int imin0, int imax0,
                   int imin1, int imax1,
                   int imin2, int imax2,
                   const int FACEX0,
                   const int FACEX1,
                   const int FACEX2) {
  // Global data index - expecting a 1D dataset
  // Thread indices
  const int tid0  = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1  = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2  = blockIdx.z * blockDim.z + threadIdx.z;
  
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  for (int i2 = tid2+imin2; i2 < imax2; i2+=stride2) {
    for (int i1 = tid1+imin1; i1 < imax1; i1+=stride1) {
      for (int i0 = tid0+imin0; i0 < imax0; i0+=stride0) {
        gfs[IDX4(which_gf, i0, i1, i2)] = +3.0 * gfs[IDX4(which_gf, i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2)] -
                                          3.0 * gfs[IDX4(which_gf, i0 + 2 * FACEX0, i1 + 2 * FACEX1, i2 + 2 * FACEX2)] +
                                          1.0 * gfs[IDX4(which_gf, i0 + 3 * FACEX0, i1 + 3 * FACEX1, i2 + 3 * FACEX2)];
      }
    }
  }
}

/*
 * Apply (quadratic extrapolation) spatial boundary conditions to the scalar wave gridfunctions.
 * BCs are applied to all six boundary faces of the cube, filling in the innermost
 * ghost zone first, and moving outward.
 */
void apply_bcs(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict gfs) {
  
  // dim3 grid(GPU_NGRID0,GPU_NGRID1,GPU_NGRID2);
  // dim3 block(GPU_NBLOCK0,GPU_NBLOCK1,GPU_NBLOCK2);
  // Using multi-dimensional thread indexing doesn't
  // work for some reason...
  // dim3 block(1024,1,1);
  int grid = 1;
  int block = 1;

  int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
  cudaMemcpy(&Nxx_plus_2NGHOSTS0, &params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS1, &params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS2, &params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")

  for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
    int imin[3] = {(int) NGHOSTS, (int) NGHOSTS, (int) NGHOSTS};
    int imax[3] = {
      (int) (Nxx_plus_2NGHOSTS0 - NGHOSTS),
      (int) (Nxx_plus_2NGHOSTS1 - NGHOSTS),
      (int) (Nxx_plus_2NGHOSTS2 - NGHOSTS)
    };
    for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
      // After updating each face, adjust imin[] and imax[]
      //   to reflect the newly-updated face extents.
      update_face<<<grid, block>>>(which_gf, params, gfs, imin[0] - 1, imin[0], imin[1], imax[1], imin[2], imax[2], MINFACE, NUL, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imin[0]--;
      update_face<<<grid, block>>>(which_gf, params, gfs, imax[0], imax[0] + 1, imin[1], imax[1], imin[2], imax[2], MAXFACE, NUL, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imax[0]++;

      update_face<<<grid, block>>>(which_gf, params, gfs, imin[0], imax[0], imin[1] - 1, imin[1], imin[2], imax[2], NUL, MINFACE, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imin[1]--;
      update_face<<<grid, block>>>(which_gf, params, gfs, imin[0], imax[0], imax[1], imax[1] + 1, imin[2], imax[2], NUL, MAXFACE, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imax[1]++;

      update_face<<<grid, block>>>(which_gf, params, gfs, imin[0], imax[0], imin[1], imax[1], imin[2] - 1, imin[2], NUL, NUL, MINFACE);
      cudaCheckErrors(update_face, "kernel failed")
      imin[2]--;
      update_face<<<grid, block>>>(which_gf, params, gfs, imin[0], imax[0], imin[1], imax[1], imax[2], imax[2] + 1, NUL, NUL, MAXFACE);
      cudaCheckErrors(update_face, "kernel failed")
      imax[2]++;
    }
  }
}