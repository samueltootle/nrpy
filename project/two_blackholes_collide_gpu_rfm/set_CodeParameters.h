const REAL BH1_mass = commondata->BH1_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_mass
const REAL BH1_posn_x = commondata->BH1_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_x
const REAL BH1_posn_y = commondata->BH1_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_y
const REAL BH1_posn_z = commondata->BH1_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_z
const REAL BH2_mass = commondata->BH2_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_mass
const REAL BH2_posn_x = commondata->BH2_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_x
const REAL BH2_posn_y = commondata->BH2_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_y
const REAL BH2_posn_z = commondata->BH2_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_z
const REAL Cart_originx = params->Cart_originx;                 // nrpy.grid::Cart_originx
const REAL Cart_originy = params->Cart_originy;                 // nrpy.grid::Cart_originy
const REAL Cart_originz = params->Cart_originz;                 // nrpy.grid::Cart_originz
const REAL CFL_FACTOR = commondata->CFL_FACTOR;                 // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
const REAL convergence_factor = commondata->convergence_factor; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
const int CoordSystem_hash = params->CoordSystem_hash;          // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::CoordSystem_hash
char CoordSystemName[50];                                       // nrpy.reference_metric::CoordSystemName
{
  strncpy(CoordSystemName, params->CoordSystemName, 49);
  CoordSystemName[49] = '\0';
} // Properly null terminate char array.
const REAL diagnostics_output_every =
    commondata->diagnostics_output_every; // nrpy.infrastructures.BHaH.general_relativity.BSSN_C_codegen_library::diagnostics_output_every
const REAL dt = commondata->dt;           // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::dt
const REAL dxx0 = params->dxx0;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx0
const REAL dxx1 = params->dxx1;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx1
const REAL dxx2 = params->dxx2;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx2
const REAL eta = commondata->eta;         // nrpy.equations.general_relativity.BSSN_gauge_RHSs::eta
const REAL f0_of_xx0 = params->f0_of_xx0; // nrpy.reference_metric_Spherical::f0_of_xx0
const REAL f1_of_xx1 = params->f1_of_xx1; // nrpy.reference_metric_Spherical::f1_of_xx1
const REAL f2_of_xx0 = params->f2_of_xx0; // nrpy.reference_metric_Spherical::f2_of_xx0
const REAL f3_of_xx2 = params->f3_of_xx2; // nrpy.reference_metric_Spherical::f3_of_xx2
const REAL f4_of_xx1 = params->f4_of_xx1; // nrpy.reference_metric_Spherical::f4_of_xx1
const REAL grid_physical_size = params->grid_physical_size; // nrpy.reference_metric::grid_physical_size
const REAL invdxx0 = params->invdxx0;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx0
const REAL invdxx1 = params->invdxx1;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx1
const REAL invdxx2 = params->invdxx2;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx2
const int nn = commondata->nn;                              // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn
const int nn_0 = commondata->nn_0;                          // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn_0
const int NUMGRIDS = commondata->NUMGRIDS;                  // nrpy.grid::NUMGRIDS
const int Nxx0 = params->Nxx0;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx0
const int Nxx1 = params->Nxx1;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx1
const int Nxx2 = params->Nxx2;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx2
const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS0
const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS1
const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS2
char outer_bc_type[50]; // nrpy.infrastructures.BHaH.CurviBoundaryConditions.CurviBoundaryConditions::outer_bc_type
{
  strncpy(outer_bc_type, commondata->outer_bc_type, 49);
  outer_bc_type[49] = '\0';
} // Properly null terminate char array.
const REAL RMAX = params->RMAX;           // nrpy.reference_metric_Spherical::RMAX
const REAL t_0 = commondata->t_0;         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_0
const REAL t_final = commondata->t_final; // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
const REAL time = commondata->time;       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::time
const REAL xxmax0 = params->xxmax0;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax0
const REAL xxmax1 = params->xxmax1;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax1
const REAL xxmax2 = params->xxmax2;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax2
const REAL xxmin0 = params->xxmin0;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin0
const REAL xxmin1 = params->xxmin1;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin1
const REAL xxmin2 = params->xxmin2;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin2
