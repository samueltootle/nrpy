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
      // printf("\n%d - %1.15f\n", IDX4(which_gf, i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2), gfs[IDX4(which_gf, i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2)]);
      }
    }
  }
  // printf("\n%d, %d - %d, %d - %d, %d\n", imin0, imax0, imin1, imax1, imin2, imax2);
}

/*
 * Apply (quadratic extrapolation) spatial boundary conditions to the scalar wave gridfunctions.
 * BCs are applied to all six boundary faces of the cube, filling in the innermost
 * ghost zone first, and moving outward.
 */
void apply_bcs(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict gfs) {

  int Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2;
  cudaMemcpy(&Nxx_plus_2NGHOSTS0, &params->Nxx_plus_2NGHOSTS0, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS1, &params->Nxx_plus_2NGHOSTS1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")
  cudaMemcpy(&Nxx_plus_2NGHOSTS2, &params->Nxx_plus_2NGHOSTS2, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "memory failed")

  dim3 grid;
  dim3 block;

  for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
    int imin[3] = {NGHOSTS, NGHOSTS, NGHOSTS};
    int imax[3] = {
      (int) (Nxx_plus_2NGHOSTS0 - NGHOSTS),
      (int) (Nxx_plus_2NGHOSTS1 - NGHOSTS),
      (int) (Nxx_plus_2NGHOSTS2 - NGHOSTS)
    };
    for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
      // After updating each face, adjust imin[] and imax[]
      //   to reflect the newly-updated face extents.
      cudaDeviceSynchronize();
      size_t tx = 1;
      uint Ny = (imax[1] - imin[1]);
      uint Nz = (imax[2] - imin[2]);
      size_t ty = MIN(1024, Ny);
      size_t tz = 1024 / ty;
      block = dim3(tx, ty, tz);
      grid = dim3(1, (Ny + ty - 1)/ty, (Nz + tz - 1)/tz);
      update_face<<<grid, block, 0, stream1>>>(which_gf, params, gfs, imin[0] - 1, imin[0], imin[1], imax[1], imin[2], imax[2], MINFACE, NUL, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imin[0]--;
      update_face<<<grid, block, 0, stream2>>>(which_gf, params, gfs, imax[0], imax[0] + 1, imin[1], imax[1], imin[2], imax[2], MAXFACE, NUL, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imax[0]++;

      uint Nx = (imax[0] - imin[0]);
      tx = MIN(1024, Nx);
      ty = 1;
      tz = 1024 / tx;
      block = dim3(tx, ty, tz);
      grid = dim3((Nx + tx - 1)/tx, 1, (Nz + tz - 1)/tz);
      update_face<<<grid, block, 0, stream3>>>(which_gf, params, gfs, imin[0], imax[0], imin[1] - 1, imin[1], imin[2], imax[2], NUL, MINFACE, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imin[1]--;
      update_face<<<grid, block, 0, stream4>>>(which_gf, params, gfs, imin[0], imax[0], imax[1], imax[1] + 1, imin[2], imax[2], NUL, MAXFACE, NUL);
      cudaCheckErrors(update_face, "kernel failed")
      imax[1]++;

      ty = tz;
      tz = 1;
      grid = dim3((Nx + tx - 1)/tx, (Ny + ty - 1)/ty, 1);
      update_face<<<grid, block, 0, stream5>>>(which_gf, params, gfs, imin[0], imax[0], imin[1], imax[1], imin[2] - 1, imin[2], NUL, NUL, MINFACE);
      cudaCheckErrors(update_face, "kernel failed")
      imin[2]--;
      update_face<<<grid, block, 0, stream6>>>(which_gf, params, gfs, imin[0], imax[0], imin[1], imax[1], imax[2], imax[2] + 1, NUL, NUL, MAXFACE);
      cudaCheckErrors(update_face, "kernel failed")
      imax[2]++;
    }
  }
}
