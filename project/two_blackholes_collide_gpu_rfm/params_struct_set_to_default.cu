#include "BHaH_defines.h"
#include "BHaH_gpu_defines.h"
/*
 * Set params_struct to default values specified within NRPy+.
 */
__global__
void params_struct_set_to_default_gpu(params_struct * params) {
  // Set params_struct variables to default
  params->Cart_originx = 0.0;       // nrpy.grid::Cart_originx
  params->Cart_originy = 0.0;       // nrpy.grid::Cart_originy
  params->Cart_originz = 0.0;       // nrpy.grid::Cart_originz
  params->Nxx0 = 72;                // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx0
  params->Nxx1 = 12;                // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx1
  params->Nxx2 = 2;                 // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx2
  params->grid_physical_size = 7.5; // nrpy.reference_metric::grid_physical_size
}

__host__
void params_struct_set_to_default_host(params_struct * params) {
  cudaMalloc(&params, sizeof(params_struct));
  cudaCheckErrors(malloc, "Malloc failed")
  params_struct_set_to_default_gpu<<<1,1>>>(params);
  cudaCheckErrors(params_default, "kernel failed")
}

void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Loop over params structs:
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    params_struct_set_to_default_host(&griddata[grid].params);
  }
}