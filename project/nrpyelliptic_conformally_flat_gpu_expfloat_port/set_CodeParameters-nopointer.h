[[maybe_unused]] const REAL AMAX = params.AMAX;                   // nrpy.reference_metric_SinhSymTP::AMAX
[[maybe_unused]] const REAL bare_mass_0 = commondata.bare_mass_0; // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_0
[[maybe_unused]] const REAL bare_mass_1 = commondata.bare_mass_1; // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_1
[[maybe_unused]] const REAL bScale = params.bScale;               // nrpy.reference_metric_SinhSymTP::bScale
[[maybe_unused]] const REAL Cart_originx = params.Cart_originx;   // nrpy.grid::Cart_originx
[[maybe_unused]] const REAL Cart_originy = params.Cart_originy;   // nrpy.grid::Cart_originy
[[maybe_unused]] const REAL Cart_originz = params.Cart_originz;   // nrpy.grid::Cart_originz
[[maybe_unused]] const REAL CFL_FACTOR = commondata.CFL_FACTOR;   // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::CFL_FACTOR
[[maybe_unused]] const REAL checkpoint_every =
    commondata.checkpoint_every; // nrpy.infrastructures.BHaH.checkpoints.base_checkpointing::checkpoint_every
[[maybe_unused]] const REAL convergence_factor =
    commondata.convergence_factor; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::convergence_factor
[[maybe_unused]] const int CoordSystem_hash =
    params.CoordSystem_hash;               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::CoordSystem_hash
[[maybe_unused]] char CoordSystemName[50]; // nrpy.reference_metric::CoordSystemName
{
  // Copy up to 49 characters from params.CoordSystemName to CoordSystemName
  strncpy(CoordSystemName, params.CoordSystemName, 50 - 1);
  // Explicitly null-terminate CoordSystemName to ensure it is a valid C-string
  CoordSystemName[50 - 1] = '\0'; // Properly null terminate char array.
}
[[maybe_unused]] const int diagnostics_output_every =
    commondata.diagnostics_output_every; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::diagnostics_output_every
[[maybe_unused]] const REAL dt = commondata.dt;                   // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::dt
[[maybe_unused]] const REAL dxx0 = params.dxx0;                   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx0
[[maybe_unused]] const REAL dxx1 = params.dxx1;                   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx1
[[maybe_unused]] const REAL dxx2 = params.dxx2;                   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx2
[[maybe_unused]] const REAL eta_damping = commondata.eta_damping; // nrpy.equations.nrpyelliptic.CommonParams::eta_damping
[[maybe_unused]] const REAL f0_of_xx0 = params.f0_of_xx0;         // nrpy.reference_metric_SinhSymTP::f0_of_xx0
[[maybe_unused]] const REAL f1_of_xx1 = params.f1_of_xx1;         // nrpy.reference_metric_SinhSymTP::f1_of_xx1
[[maybe_unused]] const REAL f2_of_xx0 = params.f2_of_xx0;         // nrpy.reference_metric_SinhSymTP::f2_of_xx0
[[maybe_unused]] const REAL f3_of_xx2 = params.f3_of_xx2;         // nrpy.reference_metric_SinhSymTP::f3_of_xx2
[[maybe_unused]] const REAL f4_of_xx1 = params.f4_of_xx1;         // nrpy.reference_metric_SinhSymTP::f4_of_xx1
[[maybe_unused]] const int grid_idx = params.grid_idx; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::grid_idx
[[maybe_unused]] const REAL grid_physical_size = params.grid_physical_size; // nrpy.reference_metric::grid_physical_size
[[maybe_unused]] const bool grid_rotates = params.grid_rotates;             // nrpy.grid::grid_rotates
[[maybe_unused]] char gridding_choice[200]; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::gridding_choice
{
  // Copy up to 199 characters from commondata.gridding_choice to gridding_choice
  strncpy(gridding_choice, commondata.gridding_choice, 200 - 1);
  // Explicitly null-terminate gridding_choice to ensure it is a valid C-string
  gridding_choice[200 - 1] = '\0'; // Properly null terminate char array.
}
[[maybe_unused]] const REAL invdxx0 = params.invdxx0; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx0
[[maybe_unused]] const REAL invdxx1 = params.invdxx1; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx1
[[maybe_unused]] const REAL invdxx2 = params.invdxx2; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx2
[[maybe_unused]] const REAL log10_current_residual =
    commondata.log10_current_residual; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_current_residual
[[maybe_unused]] const REAL log10_residual_tolerance =
    commondata.log10_residual_tolerance; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_residual_tolerance
[[maybe_unused]] const REAL MINIMUM_GLOBAL_WAVESPEED =
    commondata.MINIMUM_GLOBAL_WAVESPEED;                   // nrpy.equations.nrpyelliptic.CommonParams::MINIMUM_GLOBAL_WAVESPEED
[[maybe_unused]] const int nn = commondata.nn;             // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::nn
[[maybe_unused]] const int nn_0 = commondata.nn_0;         // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::nn_0
[[maybe_unused]] const int nn_max = commondata.nn_max;     // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::nn_max
[[maybe_unused]] const int NUMGRIDS = commondata.NUMGRIDS; // nrpy.grid::NUMGRIDS
[[maybe_unused]] const int Nxx0 = params.Nxx0;             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx0
[[maybe_unused]] const int Nxx1 = params.Nxx1;             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx1
[[maybe_unused]] const int Nxx2 = params.Nxx2;             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx2
[[maybe_unused]] const int Nxx_plus_2NGHOSTS0 =
    params.Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS0
[[maybe_unused]] const int Nxx_plus_2NGHOSTS1 =
    params.Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS1
[[maybe_unused]] const int Nxx_plus_2NGHOSTS2 =
    params.Nxx_plus_2NGHOSTS2;           // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS2
[[maybe_unused]] char outer_bc_type[50]; // nrpy.infrastructures.BHaH.CurviBoundaryConditions.cuda.CurviBoundaryConditions::outer_bc_type
{
  // Copy up to 49 characters from commondata.outer_bc_type to outer_bc_type
  strncpy(outer_bc_type, commondata.outer_bc_type, 50 - 1);
  // Explicitly null-terminate outer_bc_type to ensure it is a valid C-string
  outer_bc_type[50 - 1] = '\0'; // Properly null terminate char array.
}
[[maybe_unused]] const REAL P0_x = commondata.P0_x;   // nrpy.equations.nrpyelliptic.CommonParams::P0_x
[[maybe_unused]] const REAL P0_y = commondata.P0_y;   // nrpy.equations.nrpyelliptic.CommonParams::P0_y
[[maybe_unused]] const REAL P0_z = commondata.P0_z;   // nrpy.equations.nrpyelliptic.CommonParams::P0_z
[[maybe_unused]] const REAL P1_x = commondata.P1_x;   // nrpy.equations.nrpyelliptic.CommonParams::P1_x
[[maybe_unused]] const REAL P1_y = commondata.P1_y;   // nrpy.equations.nrpyelliptic.CommonParams::P1_y
[[maybe_unused]] const REAL P1_z = commondata.P1_z;   // nrpy.equations.nrpyelliptic.CommonParams::P1_z
[[maybe_unused]] const REAL S0_x = commondata.S0_x;   // nrpy.equations.nrpyelliptic.CommonParams::S0_x
[[maybe_unused]] const REAL S0_y = commondata.S0_y;   // nrpy.equations.nrpyelliptic.CommonParams::S0_y
[[maybe_unused]] const REAL S0_z = commondata.S0_z;   // nrpy.equations.nrpyelliptic.CommonParams::S0_z
[[maybe_unused]] const REAL S1_x = commondata.S1_x;   // nrpy.equations.nrpyelliptic.CommonParams::S1_x
[[maybe_unused]] const REAL S1_y = commondata.S1_y;   // nrpy.equations.nrpyelliptic.CommonParams::S1_y
[[maybe_unused]] const REAL S1_z = commondata.S1_z;   // nrpy.equations.nrpyelliptic.CommonParams::S1_z
[[maybe_unused]] const REAL SINHWAA = params.SINHWAA; // nrpy.reference_metric_SinhSymTP::SINHWAA
[[maybe_unused]] const bool stop_relaxation =
    commondata.stop_relaxation;                   // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::stop_relaxation
[[maybe_unused]] const REAL t_0 = commondata.t_0; // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_0
[[maybe_unused]] const REAL t_final = commondata.t_final; // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_final
[[maybe_unused]] const REAL time = commondata.time;       // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::time
[[maybe_unused]] const REAL xxmax0 = params.xxmax0;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax0
[[maybe_unused]] const REAL xxmax1 = params.xxmax1;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax1
[[maybe_unused]] const REAL xxmax2 = params.xxmax2;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax2
[[maybe_unused]] const REAL xxmin0 = params.xxmin0;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin0
[[maybe_unused]] const REAL xxmin1 = params.xxmin1;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin1
[[maybe_unused]] const REAL xxmin2 = params.xxmin2;       // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin2
[[maybe_unused]] const REAL zPunc = commondata.zPunc;     // nrpy.equations.nrpyelliptic.CommonParams::zPunc
