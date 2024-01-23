#include "BHaH_defines.h"
/*
 * Set params_struct to default values specified within NRPy+.
 */
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Loop over params structs:
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    params_struct *restrict params = &griddata[grid].params;
    // Set params_struct variables to default
    params->Cart_originx = 0.0;       // nrpy.grid::Cart_originx
    params->Cart_originy = 0.0;       // nrpy.grid::Cart_originy
    params->Cart_originz = 0.0;       // nrpy.grid::Cart_originz
    params->Nxx0 = 72;                // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx0
    params->Nxx1 = 12;                // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx1
    params->Nxx2 = 12;                // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx2
    params->grid_physical_size = 7.5; // nrpy.reference_metric::grid_physical_size
  }
}