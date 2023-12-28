#include "BHaH_defines.h"
/*
 * Set commondata_struct to default values specified within NRPy+.
 */
void commondata_struct_set_to_default(commondata_struct *restrict commondata) {

  // Set commondata_struct variables to default
  commondata->BH1_mass = 0.5;                  // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_mass
  commondata->BH1_posn_x = 0.0;                // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_x
  commondata->BH1_posn_y = 0.0;                // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_y
  commondata->BH1_posn_z = 0.5;                // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_z
  commondata->BH2_mass = 0.5;                  // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_mass
  commondata->BH2_posn_x = 0.0;                // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_x
  commondata->BH2_posn_y = 0.0;                // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_y
  commondata->BH2_posn_z = -0.5;               // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_z
  commondata->CFL_FACTOR = 0.5;                // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
  commondata->NUMGRIDS = 1;                    // nrpy.grid::NUMGRIDS
  commondata->convergence_factor = 1.0;        // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
  commondata->diagnostics_output_every = 0.25; // nrpy.infrastructures.BHaH.general_relativity.BSSN_C_codegen_library::diagnostics_output_every
  commondata->eta = 1.0;                       // nrpy.equations.general_relativity.BSSN_gauge_RHSs::eta
  commondata->t_final = 7.5;                   // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
  snprintf(commondata->outer_bc_type, 50, "radiation"); // nrpy.infrastructures.BHaH.CurviBoundaryConditions.CurviBoundaryConditions::outer_bc_type
}
