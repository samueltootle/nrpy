#include "BHaH_defines.h"
/*
 * Set commondata_struct to default values specified within NRPy+.
 */
void commondata_struct_set_to_default(commondata_struct *restrict commondata) {

  // Set commondata_struct variables to default
  commondata->CFL_FACTOR = 0.5;               // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
  commondata->NUMGRIDS = 1;                   // nrpy.grid::NUMGRIDS
  commondata->convergence_factor = 1.0;       // __main__::convergence_factor
  commondata->diagnostics_output_every = 0.2; // __main__::diagnostics_output_every
  commondata->sigma = 3.0;                    // nrpy.equations.wave_equation.WaveEquation_Solutions_InitialData::sigma
  commondata->t_final = 8.0;                  // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
  commondata->wavespeed = 1.0;                // nrpy.equations.wave_equation.CommonParams::wavespeed
}
