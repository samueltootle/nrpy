#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * Compute 1st derivative finite-difference derivative with arbitrary upwind
 */
__device__ inline REAL FD1_arbitrary_upwind_x0_dirn(const REAL *restrict gf, const int i0, const int i1, const int i2, const int offset) {
  REAL const& invdxx0 = d_params.invdxx0;
  __attribute_maybe_unused__ int const& Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  __attribute_maybe_unused__ int const& Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  __attribute_maybe_unused__ int const& Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  switch (offset) {
  case 0:
    return (+ FDPart1_Rational_1_12 * gf[IDX3(i0 - 2, i1, i2)] - FDPart1_Rational_2_3  * gf[IDX3(i0 - 1, i1, i2)] 
            + FDPart1_Rational_2_3  * gf[IDX3(i0 + 1, i1, i2)] - FDPart1_Rational_1_12 * gf[IDX3(i0 + 2, i1, i2)]) *
           invdxx0;
  case 1:
    return (-FDPart1_Rational_1_4 * gf[IDX3(i0 - 1, i1, i2)] - FDPart1_Rational_5_6 * gf[IDX3(i0, i1, i2)] 
            + FDPart1_Rational_3_2 * gf[IDX3(i0 + 1, i1, i2)] - FDPart1_Rational_1_2 * gf[IDX3(i0 + 2, i1, i2)] 
            + FDPart1_Rational_1_12 * gf[IDX3(i0 + 3, i1, i2)]) *
           invdxx0;
  case -1:
    return (-FDPart1_Rational_1_12 * gf[IDX3(i0 - 3, i1, i2)] + FDPart1_Rational_1_2 * gf[IDX3(i0 - 2, i1, i2)] 
            - FDPart1_Rational_3_2 * gf[IDX3(i0 - 1, i1, i2)] + FDPart1_Rational_5_6 * gf[IDX3(i0, i1, i2)] 
            + FDPart1_Rational_1_4 * gf[IDX3(i0 + 1, i1, i2)]) *
           invdxx0;
  case 2:
    return (-FDPart1_Rational_5_2 * FDPart1_Rational_5_6 * gf[IDX3(i0, i1, i2)] + 4 * gf[IDX3(i0 + 1, i1, i2)] - 3 * gf[IDX3(i0 + 2, i1, i2)] 
            + FDPart1_Rational_4_3 * gf[IDX3(i0 + 3, i1, i2)] - FDPart1_Rational_1_4 * gf[IDX3(i0 + 4, i1, i2)]) *
           invdxx0;
  case -2:
    return (+FDPart1_Rational_1_4 * gf[IDX3(i0 - 4, i1, i2)] - FDPart1_Rational_4_3 * gf[IDX3(i0 - 3, i1, i2)] 
            + 3 * gf[IDX3(i0 - 2, i1, i2)] - 4 * gf[IDX3(i0 - 1, i1, i2)] 
            + FDPart1_Rational_5_2 * FDPart1_Rational_5_6 * gf[IDX3(i0, i1, i2)]) *
           invdxx0;
  }
  return 0.0 / 0.0; // poison output if offset computed incorrectly
}
/*
 * Compute r(xx0,xx1,xx2) and partial_r x^i.
 */
__device__ inline void r_and_partial_xi_partial_r_derivs(const REAL xx0, const REAL xx1, const REAL xx2, REAL *r, REAL *partial_x0_partial_r,
                                                     REAL *partial_x1_partial_r, REAL *partial_x2_partial_r) {
  *r = xx0;
  *partial_x0_partial_r = 1;
  *partial_x1_partial_r = 0;
  *partial_x2_partial_r = 0;
}
/*
 * Compute \partial_r f
 */
__device__ inline REAL compute_partial_r_f(const REAL *restrict gfs, const int which_gf, const int dest_i0, const int dest_i1, const int dest_i2,
                                       const int FACEi0, const int FACEi1, const int FACEi2, const REAL partial_x0_partial_r,
                                       const REAL partial_x1_partial_r, const REAL partial_x2_partial_r) {
// #include "../set_CodeParameters.h"
  int const& Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  ///////////////////////////////////////////////////////////

  // FD1_stencil_radius = radiation_BC_fd_order/2 = 2
  const int FD1_stencil_radius = 2;

  const int ntot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  ///////////////////////////////////////////////////////////
  // Next we'll compute partial_xi f, using a maximally-centered stencil.
  //   The {i0,i1,i2}_offset parameters set the offset of the maximally-centered
  //   stencil, such that an offset=0 implies a centered stencil.

  // CHECK: Nxx_plus_2NGHOSTS0=10; FD1_stencil_radius=2. Then Nxx_plus_2NGHOSTS0-FD1_stencil_radius-1 = 7
  //  if dest_i0 = 9, we get i0_offset=7-9=-2, so the (4th order) deriv
  //  stencil is: -4,-3,-2,-1,0

  // CHECK: if FD1_stencil_radius=2 and dest_i0 = 1, we get i0_offset = FD1_stencil_radius-dest_i0 = 1,
  //  so the (4th order) deriv stencil is: -1,0,1,2,3

  // CHECK: if FD1_stencil_radius=2 and dest_i0 = 0, we get i0_offset = FD1_stencil_radius-1 = 2,
  //  so the (4th order) deriv stencil is: 0,1,2,3,4
  int i0_offset = FACEi0; // Shift stencil away from the face we're updating.
  // Next adjust i0_offset so that FD stencil never goes out of bounds.
  if (dest_i0 < FD1_stencil_radius)
    i0_offset = FD1_stencil_radius - dest_i0;
  else if (dest_i0 > (Nxx_plus_2NGHOSTS0 - FD1_stencil_radius - 1))
    i0_offset = (Nxx_plus_2NGHOSTS0 - FD1_stencil_radius - 1) - dest_i0;
  const REAL partial_x0_f = FD1_arbitrary_upwind_x0_dirn(&gfs[which_gf * ntot], dest_i0, dest_i1, dest_i2, i0_offset);
  const REAL partial_x1_f = 0.0;
  const REAL partial_x2_f = 0.0;
  return partial_x0_partial_r * partial_x0_f + partial_x1_partial_r * partial_x1_f + partial_x2_partial_r * partial_x2_f;
}

/*
 * *** Apply radiation BCs to all outer boundaries. ***
 *
 */
__device__ inline REAL radiation_bcs(REAL *restrict xx[3], const REAL *restrict gfs, REAL *restrict gfs_rhss,
                                 const int which_gf, const REAL gf_wavespeed, const REAL gf_f_infinity, const int dest_i0, const int dest_i1,
                                 const int dest_i2, const short FACEi0, const short FACEi1, const short FACEi2) {

  // Nearest "interior" neighbor of this gridpoint, based on current face
  const int dest_i0_int = dest_i0 + 1 * FACEi0, dest_i1_int = dest_i1 + 1 * FACEi1, dest_i2_int = dest_i2 + 1 * FACEi2;
  REAL r, partial_x0_partial_r, partial_x1_partial_r, partial_x2_partial_r;
  REAL r_int, partial_x0_partial_r_int, partial_x1_partial_r_int, partial_x2_partial_r_int;
  r_and_partial_xi_partial_r_derivs(xx[0][dest_i0], xx[1][dest_i1], xx[2][dest_i2], &r, &partial_x0_partial_r,
                                    &partial_x1_partial_r, &partial_x2_partial_r);
  r_and_partial_xi_partial_r_derivs(xx[0][dest_i0_int], xx[1][dest_i1_int], xx[2][dest_i2_int], &r_int, &partial_x0_partial_r_int,
                                    &partial_x1_partial_r_int, &partial_x2_partial_r_int);
  const REAL partial_r_f = compute_partial_r_f(gfs, which_gf, dest_i0, dest_i1, dest_i2, FACEi0, FACEi1, FACEi2,
                                               partial_x0_partial_r, partial_x1_partial_r, partial_x2_partial_r);
  const REAL partial_r_f_int = compute_partial_r_f(gfs, which_gf, dest_i0_int, dest_i1_int, dest_i2_int, FACEi0, FACEi1,
                                                   FACEi2, partial_x0_partial_r_int, partial_x1_partial_r_int, partial_x2_partial_r_int);
  
  int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
  const int idx3 = IDX3(dest_i0, dest_i1, dest_i2);
  const int idx3_int = IDX3(dest_i0_int, dest_i1_int, dest_i2_int);

  const REAL partial_t_f_int = gfs_rhss[IDX4pt(which_gf, idx3_int)];

  const REAL c = gf_wavespeed;
  const REAL f_infinity = gf_f_infinity;
  const REAL f = gfs[IDX4pt(which_gf, idx3)];
  const REAL f_int = gfs[IDX4pt(which_gf, idx3_int)];
  const REAL partial_t_f_int_outgoing_wave = -c * (partial_r_f_int + (f_int - f_infinity) / r_int);

  const REAL k = r_int * r_int * r_int * (partial_t_f_int - partial_t_f_int_outgoing_wave);

  const REAL rinv = 1.0 / r;
  const REAL partial_t_f_outgoing_wave = -c * (partial_r_f + (f - f_infinity) * rinv);

  return partial_t_f_outgoing_wave + k * rinv * rinv * rinv;
}

/*
 * This function is responsible for applying boundary conditions (BCs) to both pure outer and inner
 * boundary points. In the first step, it parallelizes the task using OpenMP and starts by applying BCs to
 * the outer boundary points layer-by-layer, prioritizing the faces in the order x0, x1, x2. The second step
 * applies BCs to the inner boundary points, which may map either to the grid interior or to the outer boundary.
 *
 */
__global__
void apply_bcs_pure_only_gpu(const int num_pure_outer_boundary_points, const int which_gz, const int dirn,
  const outerpt_bc_struct *restrict pure_outer_bc_array, REAL *restrict gfs, REAL *restrict rhs_gfs,
  REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2,
  const REAL *restrict custom_wavespeed, const REAL *restrict custom_f_infinity) {
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
    const int idx3 = IDX3(i0, i1, i2);
    REAL* xx[3] = {_xx0, _xx1, _xx2};
    // printf("%f\n", custom_f_infinity[0]);
    for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
      // *** Apply radiation BCs to all outer boundary points. ***
      rhs_gfs[IDX4pt(which_gf, idx3)] = radiation_bcs(xx, gfs, rhs_gfs, which_gf, custom_wavespeed[which_gf],
                                                      custom_f_infinity[which_gf], i0, i1, i2, FACEX0, FACEX1, FACEX2);
    // printf("%d: %f\n", idx3, rhs_gfs[IDX4pt(which_gf, idx3)]);
        // if(idx3 == IDX3(34, 18 , 18)) {
        //     printf("GF %d: %f\n", which_gf,
        //     rhs_gfs[IDX4pt(which_gf, idx3)]);
        // }
    }
  }
}

void apply_bcs_pure_only(const bc_struct *restrict bcstruct,
  REAL *restrict xx[3], const REAL *restrict custom_wavespeed, const REAL *restrict custom_f_infinity,
    REAL *restrict gfs, REAL *restrict rhs_gfs) {
  const bc_info_struct *bc_info = &bcstruct->bc_info;
  for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
    for (int dirn = 0; dirn < 3; dirn++) {
      if (bc_info->num_pure_outer_boundary_points[which_gz][dirn] > 0) {
        int num_pure = bc_info->num_pure_outer_boundary_points[which_gz][dirn];
        size_t block_threadsx = MIN(1024,(num_pure/32U) * 32U);
        size_t grid_blocks = (num_pure + block_threadsx -1) / block_threadsx;
        size_t gz_idx = dirn + (3 * which_gz);
        apply_bcs_pure_only_gpu<<<grid_blocks, block_threadsx>>>(
        // apply_bcs_pure_only_gpu<<<1,1>>>(
          num_pure, which_gz, dirn, bcstruct->pure_outer_bc_array[gz_idx], gfs, rhs_gfs, 
          xx[0], xx[1], xx[2], custom_wavespeed, custom_f_infinity
        );
      }
    }
  }
}

void apply_bcs_outerradiation_and_inner__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                        const bc_struct *restrict bcstruct, REAL * xx[3],
                                                        const REAL *restrict custom_wavespeed, const REAL *restrict custom_f_infinity,
                                                        REAL *restrict gfs, REAL *restrict rhs_gfs) {
#include "../set_CodeParameters.h"
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
  apply_bcs_pure_only(bcstruct, xx, custom_wavespeed, custom_f_infinity, gfs, rhs_gfs);
  
  cudaDeviceSynchronize();
  // for(int i = 0; i < NUM_EVOL_GFS; ++i)
  //     print_var<<<1,1>>>(gfs, IDX4(i, 34, 10 , 10));
  // cudaDeviceSynchronize();
  // printf("**************************_pure\n");

  ///////////////////////////////////////////////////////
  // STEP 2 of 2: Apply BCs to inner boundary points.
  //              These map to either the grid interior
  //              ("pure inner") or to pure outer boundary
  //              points ("inner maps to outer"). Those
  //              that map to outer require that outer be
  //              populated first; hence this being
  //              STEP 2 OF 2.
  apply_bcs_inner_only(commondata, params, bcstruct, rhs_gfs); // <- apply inner BCs to RHS gfs only
  
  cudaDeviceSynchronize();
  // for(int i = 0; i < NUM_EVOL_GFS; ++i)
  //     print_var<<<1,1>>>(gfs, IDX4(i, 34, 10, 10));
  // cudaDeviceSynchronize();
  // printf("**************************_inner_only\n");
}
