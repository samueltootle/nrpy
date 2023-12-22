#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"

__global__
void apply_bcs_outerextrap_and_inner_only_gpu(const int num_pure_outer_boundary_points, const int which_gz, const int dirn,
  const outerpt_bc_struct *restrict pure_outer_bc_array, REAL *restrict gfs){
  int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  
  // Thread indices
  // Global data index - expecting a 1D dataset
    const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
    // Thread strides
    const int stride0 = blockDim.x * gridDim.x;

  for (int idx2d = tid0; idx2d < num_pure_outer_boundary_points; idx2d+=stride0) {
    const short i0 = pure_outer_bc_array[idx2d].i0;
    const short i1 = pure_outer_bc_array[idx2d].i1;
    const short i2 = pure_outer_bc_array[idx2d].i2;
    const short FACEX0 = pure_outer_bc_array[idx2d].FACEX0;
    const short FACEX1 = pure_outer_bc_array[idx2d].FACEX1;
    const short FACEX2 = pure_outer_bc_array[idx2d].FACEX2;
    const int idx_offset0 = IDX3(i0, i1, i2);
    const int idx_offset1 = IDX3(i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2);
    const int idx_offset2 = IDX3(i0 + 2 * FACEX0, i1 + 2 * FACEX1, i2 + 2 * FACEX2);
    const int idx_offset3 = IDX3(i0 + 3 * FACEX0, i1 + 3 * FACEX1, i2 + 3 * FACEX2);
    for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
      // *** Apply 2nd-order polynomial extrapolation BCs to all outer boundary points. ***
      gfs[IDX4pt(which_gf, idx_offset0)] =
          + 3.0 * gfs[IDX4pt(which_gf, idx_offset1)] 
          - 3.0 * gfs[IDX4pt(which_gf, idx_offset2)] 
          + 1.0 * gfs[IDX4pt(which_gf, idx_offset3)];
    }
  }
}

void apply_bcs_outerextrap_and_inner_only(const bc_struct *restrict bcstruct, REAL *restrict gfs) {
  const bc_info_struct *bc_info = &bcstruct->bc_info;
  for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
    for (int dirn = 0; dirn < 3; dirn++) {
      if (bc_info->num_pure_outer_boundary_points[which_gz][dirn] > 0) {
        int num_pure = bc_info->num_pure_outer_boundary_points[which_gz][dirn];
        size_t block_threadsx = MIN(1024,num_pure);
        size_t grid_blocks = (num_pure + block_threadsx -1) / block_threadsx;
        size_t gz_idx = dirn + (3 * which_gz);
        apply_bcs_outerextrap_and_inner_only_gpu<<<grid_blocks, block_threadsx>>>(
          num_pure, which_gz, dirn, bcstruct->pure_outer_bc_array[gz_idx], gfs);
      }
    }
  }
}

/*
 * #Suppose the outer boundary point is at the i0=max(i0) face. Then we fit known data at i0-3, i0-2, and i0-1
 * #  to the unique quadratic polynomial that passes through those points, and fill the data at
 * #  i0 with the value implied from the polynomial.
 * #As derived in nrpytutorial's Tutorial-Start_to_Finish-Curvilinear_BCs.ipynb,
 * #  the coefficients must be f_{i0} = f_{i0-3} - 3 f_{i0-2} + 3 f_{i0-1}.
 * #  To check these coefficients are correct, consider
 * #  * f(x0 = constant. Then f_{i0} = f_{i0-3} <- CHECK!
 * #  * f(x) = x. WOLOG suppose x0=0. Then f_{i0} = (-3dx) - 3(-2dx) + 3(-dx) = + dx(-3+6-3) = 0 <- CHECK!
 * #  * f(x) = x^2. WOLOG suppose x0=0. Then f_{i0} = (-3dx)^2 - 3(-2dx)^2 + 3(-dx)^2 = + dx^2(9-12+3) = 0 <- CHECK!
 */
void apply_bcs_outerextrap_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                     const bc_struct *restrict bcstruct, REAL *restrict gfs) {

  ////////////////////////////////////////////////////////
  // STEP 1 of 2: Apply BCs to pure outer boundary points.
  //              By "pure" we mean that these points are
  //              on the outer boundary and not also on
  //              an inner boundary.
  //              Here we fill in the innermost ghost zone
  //              layer first and move outward. At each
  //              layer, we fill in +/- x0 faces first,
  //              then +/- x1 faces, finally +/- x2 faces,
  //              filling in the edges as we go.
  // Spawn N OpenMP threads, either across all cores, or according to e.g., taskset.
  apply_bcs_outerextrap_and_inner_only(bcstruct, gfs);
  ///////////////////////////////////////////////////////
  // STEP 2 of 2: Apply BCs to inner boundary points.
  //              These map to either the grid interior
  //              ("pure inner") or to pure outer boundary
  //              points ("inner maps to outer"). Those
  //              that map to outer require that outer be
  //              populated first; hence this being
  //              STEP 2 OF 2.
  apply_bcs_inner_only(commondata, params, bcstruct, gfs);
}
