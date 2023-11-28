#include "BHaH_defines.h"
/*
 * Set params_struct to default values specified within NRPy+.
 */
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Loop over params structs:
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    params_struct *restrict params = &griddata[grid].params;
    // Set params_struct variables to default
    params->Cart_originx = 0.0; // nrpy.grid::Cart_originx
    params->Cart_originy = 0.0; // nrpy.grid::Cart_originy
    params->Cart_originz = 0.0; // nrpy.grid::Cart_originz
    params->Nxx0 = 64;          // __main__::Nxx0
    params->Nxx1 = 64;          // __main__::Nxx1
    params->Nxx2 = 64;          // __main__::Nxx2
    params->xxmax0 = 10.0;      // __main__::xxmax0
    params->xxmax1 = 10.0;      // __main__::xxmax1
    params->xxmax2 = 10.0;      // __main__::xxmax2
    params->xxmin0 = -10.0;     // __main__::xxmin0
    params->xxmin1 = -10.0;     // __main__::xxmin1
    params->xxmin2 = -10.0;     // __main__::xxmin2
  }
}
