#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
/*
 *
 * Apply BCs to inner boundary points only,
 * using data stored in bcstruct->inner_bc_array.
 * These structs are set in bcstruct_set_up().
 * Inner boundary points map to either the grid
 * interior ("pure inner") or to pure outer
 * boundary points ("inner maps to outer").
 *
 */

void apply_bcs_inner_only_gpu(int const which_gf, int const num_inner_boundary_points, innerpt_bc_struct *restrict inner_bc_array, REAL *restrict gfs) {
  int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  for (int pt = 0; pt < num_inner_boundary_points; pt++) {
      const int dstpt = inner_bc_array[pt].dstpt;
      const int srcpt = inner_bc_array[pt].srcpt;
      gfs[IDX4pt(which_gf, dstpt)] = inner_bc_array[pt].parity[evol_gf_parity[which_gf]] * gfs[IDX4pt(which_gf, srcpt)];
  } // END for(int pt=0;pt<num_inner_pts;pt++)
}

void apply_bcs_inner_only(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct,
                          REAL *restrict gfs) {
#include "set_CodeParameters.h"

  // Unpack bc_info from bcstruct
  const bc_info_struct *bc_info = &bcstruct->bc_info;

  // collapse(2) results in a nice speedup here, esp in 2D. Two_BHs_collide goes from
  //    5550 M/hr to 7264 M/hr on a Ryzen 9 5950X running on all 16 cores with core affinity.

}
