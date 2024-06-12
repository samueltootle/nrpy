#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 *
 * Initializes a cell-centered grid in SinhSymTP coordinates based on physical dimensions (grid_physical_size).
 *
 * Inputs:
 * - Nx[] inputs: Specifies new grid dimensions, if needed.
 * - grid_is_resized: Indicates if the grid has been manually resized, triggering adjustments to grid parameters.
 * - convergence_factor: Multiplier for grid dimensions to refine resolution, applied only if grid hasn't been resized.
 *
 * Parameter outputs:
 * - Nxx: Number of grid points in each direction.
 * - Nxx_plus_2NGHOSTS: Total grid points including ghost zones.
 * - dxx: Grid spacing.
 * - invdxx: Inverse of grid spacing.
 *
 * Grid setup output:
 * - xx: Coordinate values for each (cell-centered) grid point.
 *
 */
void numerical_grid_params_Nxx_dxx_xx(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3], const int Nx[3],
                                      const bool grid_is_resized) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(commondata, params, xx, Nx, grid_is_resized);
    break;
  default:
    fprintf(stderr, "ERROR in numerical_grid_params_Nxx_dxx_xx(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
}
