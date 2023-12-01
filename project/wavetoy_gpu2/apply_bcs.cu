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
void update_faces(size_t which_gf, REAL *restrict gfs, 
                  size_t i0min, size_t i0max, 
                  size_t i1min, size_t i1max, 
                  size_t i2min, size_t i2max) {
  // Global data index - expecting a 1D dataset
  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;

  uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
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
  
  int grid = 1;
  // dim3 block(GPU_NBLOCK0,GPU_NBLOCK1,GPU_NBLOCK2);
  // Using multi-dimensional thread indexing doesn't
  // work for some reason...
  int block = 1;

  for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
    size_t imin[3] = {(size_t) NGHOSTS, (size_t) NGHOSTS, (size_t) NGHOSTS};
    size_t imax[3] = {
      (size_t) (Nxx_plus_2NGHOSTS0 - NGHOSTS),
      (size_t) (Nxx_plus_2NGHOSTS1 - NGHOSTS),
      (size_t) (Nxx_plus_2NGHOSTS2 - NGHOSTS)
    };
    // After updating each face, adjust imin[] and imax[]
    //   to reflect the newly-updated face extents.
    FACE_UPDATE(which_gf, imin[0] - 1, imin[0], imin[1], imax[1], imin[2], imax[2], MINFACE, NUL, NUL);
    imin[0]--;
    FACE_UPDATE(which_gf, imax[0], imax[0] + 1, imin[1], imax[1], imin[2], imax[2], MAXFACE, NUL, NUL);
    imax[0]++;

    FACE_UPDATE(which_gf, imin[0], imax[0], imin[1] - 1, imin[1], imin[2], imax[2], NUL, MINFACE, NUL);
    imin[1]--;
    FACE_UPDATE(which_gf, imin[0], imax[0], imax[1], imax[1] + 1, imin[2], imax[2], NUL, MAXFACE, NUL);
    imax[1]++;

    FACE_UPDATE(which_gf, imin[0], imax[0], imin[1], imax[1], imin[2] - 1, imin[2], NUL, NUL, MINFACE);
    imin[2]--;
    FACE_UPDATE(which_gf, imin[0], imax[0], imin[1], imax[1], imax[2], imax[2] + 1, NUL, NUL, MAXFACE);
    imax[2]++;
  }
}
