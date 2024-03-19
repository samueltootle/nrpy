[[maybe_unused]] const REAL BH1_mass = commondata->BH1_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_mass
[[maybe_unused]] const REAL BH1_posn_x = commondata->BH1_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_x
[[maybe_unused]] const REAL BH1_posn_y = commondata->BH1_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_y
[[maybe_unused]] const REAL BH1_posn_z = commondata->BH1_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_z
[[maybe_unused]] const REAL BH2_mass = commondata->BH2_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_mass
[[maybe_unused]] const REAL BH2_posn_x = commondata->BH2_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_x
[[maybe_unused]] const REAL BH2_posn_y = commondata->BH2_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_y
[[maybe_unused]] const REAL BH2_posn_z = commondata->BH2_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_z
[[maybe_unused]] const REAL Cart_originx = params->Cart_originx;                 // nrpy.grid::Cart_originx
[[maybe_unused]] const REAL Cart_originy = params->Cart_originy;                 // nrpy.grid::Cart_originy
[[maybe_unused]] const REAL Cart_originz = params->Cart_originz;                 // nrpy.grid::Cart_originz
[[maybe_unused]] const REAL CFL_FACTOR = commondata->CFL_FACTOR;                 // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
[[maybe_unused]] const REAL convergence_factor = commondata->convergence_factor; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
[[maybe_unused]] const int CoordSystem_hash = params->CoordSystem_hash;          // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::CoordSystem_hash
[[maybe_unused]] char CoordSystemName[50];                                       // nrpy.reference_metric::CoordSystemName
{
  strncpy(CoordSystemName, params->CoordSystemName, 49);
  CoordSystemName[49] = '\0';
} // Properly null terminate char array.
[[maybe_unused]] const REAL diagnostics_output_every =
    commondata->diagnostics_output_every; // nrpy.infrastructures.BHaH.general_relativity.BSSN_C_codegen_library::diagnostics_output_every
[[maybe_unused]] const REAL dt = commondata->dt;           // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::dt
[[maybe_unused]] const REAL dxx0 = params->dxx0;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx0
[[maybe_unused]] const REAL dxx1 = params->dxx1;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx1
[[maybe_unused]] const REAL dxx2 = params->dxx2;           // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx2
[[maybe_unused]] const REAL eta = commondata->eta;         // nrpy.equations.general_relativity.BSSN_gauge_RHSs::eta
[[maybe_unused]] const REAL f0_of_xx0 = params->f0_of_xx0; // nrpy.reference_metric_Spherical::f0_of_xx0
[[maybe_unused]] const REAL f1_of_xx1 = params->f1_of_xx1; // nrpy.reference_metric_Spherical::f1_of_xx1
[[maybe_unused]] const REAL f2_of_xx0 = params->f2_of_xx0; // nrpy.reference_metric_Spherical::f2_of_xx0
[[maybe_unused]] const REAL f3_of_xx2 = params->f3_of_xx2; // nrpy.reference_metric_Spherical::f3_of_xx2
[[maybe_unused]] const REAL f4_of_xx1 = params->f4_of_xx1; // nrpy.reference_metric_Spherical::f4_of_xx1
[[maybe_unused]] const REAL grid_physical_size = params->grid_physical_size; // nrpy.reference_metric::grid_physical_size
[[maybe_unused]] const REAL invdxx0 = params->invdxx0;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx0
[[maybe_unused]] const REAL invdxx1 = params->invdxx1;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx1
[[maybe_unused]] const REAL invdxx2 = params->invdxx2;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx2
[[maybe_unused]] const int nn = commondata->nn;                              // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn
[[maybe_unused]] const int nn_0 = commondata->nn_0;                          // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn_0
[[maybe_unused]] const int NUMGRIDS = commondata->NUMGRIDS;                  // nrpy.grid::NUMGRIDS
[[maybe_unused]] const int Nxx0 = params->Nxx0;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx0
[[maybe_unused]] const int Nxx1 = params->Nxx1;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx1
[[maybe_unused]] const int Nxx2 = params->Nxx2;                              // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx2
[[maybe_unused]] const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS0
[[maybe_unused]] const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS1
[[maybe_unused]] const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS2
[[maybe_unused]] char outer_bc_type[50]; // nrpy.infrastructures.BHaH.CurviBoundaryConditions.CurviBoundaryConditions::outer_bc_type
{
  strncpy(outer_bc_type, commondata->outer_bc_type, 49);
  outer_bc_type[49] = '\0';
} // Properly null terminate char array.
[[maybe_unused]] const REAL RMAX = params->RMAX;           // nrpy.reference_metric_Spherical::RMAX
[[maybe_unused]] const REAL t_0 = commondata->t_0;         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_0
[[maybe_unused]] const REAL t_final = commondata->t_final; // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
[[maybe_unused]] const REAL time = commondata->time;       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::time
[[maybe_unused]] const REAL xxmax0 = params->xxmax0;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax0
[[maybe_unused]] const REAL xxmax1 = params->xxmax1;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax1
[[maybe_unused]] const REAL xxmax2 = params->xxmax2;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax2
[[maybe_unused]] const REAL xxmin0 = params->xxmin0;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin0
[[maybe_unused]] const REAL xxmin1 = params->xxmin1;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin1
[[maybe_unused]] const REAL xxmin2 = params->xxmin2;       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin2
