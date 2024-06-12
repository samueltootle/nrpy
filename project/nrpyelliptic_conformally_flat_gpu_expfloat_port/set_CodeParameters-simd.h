const REAL NOSIMDAMAX = params->AMAX;                               // nrpy.reference_metric_SinhSymTP::AMAX
const REAL_SIMD_ARRAY AMAX = ConstSIMD(NOSIMDAMAX);                 // nrpy.reference_metric_SinhSymTP::AMAX
const REAL NOSIMDbare_mass_0 = commondata->bare_mass_0;             // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_0
const REAL_SIMD_ARRAY bare_mass_0 = ConstSIMD(NOSIMDbare_mass_0);   // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_0
const REAL NOSIMDbare_mass_1 = commondata->bare_mass_1;             // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_1
const REAL_SIMD_ARRAY bare_mass_1 = ConstSIMD(NOSIMDbare_mass_1);   // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_1
const REAL NOSIMDbScale = params->bScale;                           // nrpy.reference_metric_SinhSymTP::bScale
const REAL_SIMD_ARRAY bScale = ConstSIMD(NOSIMDbScale);             // nrpy.reference_metric_SinhSymTP::bScale
const REAL NOSIMDCart_originx = params->Cart_originx;               // nrpy.grid::Cart_originx
const REAL_SIMD_ARRAY Cart_originx = ConstSIMD(NOSIMDCart_originx); // nrpy.grid::Cart_originx
const REAL NOSIMDCart_originy = params->Cart_originy;               // nrpy.grid::Cart_originy
const REAL_SIMD_ARRAY Cart_originy = ConstSIMD(NOSIMDCart_originy); // nrpy.grid::Cart_originy
const REAL NOSIMDCart_originz = params->Cart_originz;               // nrpy.grid::Cart_originz
const REAL_SIMD_ARRAY Cart_originz = ConstSIMD(NOSIMDCart_originz); // nrpy.grid::Cart_originz
const REAL NOSIMDCFL_FACTOR = commondata->CFL_FACTOR;               // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::CFL_FACTOR
const REAL_SIMD_ARRAY CFL_FACTOR = ConstSIMD(NOSIMDCFL_FACTOR);     // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::CFL_FACTOR
const REAL NOSIMDcheckpoint_every = commondata->checkpoint_every;   // nrpy.infrastructures.BHaH.checkpoints.base_checkpointing::checkpoint_every
const REAL_SIMD_ARRAY checkpoint_every =
    ConstSIMD(NOSIMDcheckpoint_every); // nrpy.infrastructures.BHaH.checkpoints.base_checkpointing::checkpoint_every
const REAL NOSIMDconvergence_factor =
    commondata->convergence_factor; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::convergence_factor
const REAL_SIMD_ARRAY convergence_factor =
    ConstSIMD(NOSIMDconvergence_factor); // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::convergence_factor
const int CoordSystem_hash =
    params->CoordSystem_hash; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::CoordSystem_hash
const int diagnostics_output_every =
    commondata->diagnostics_output_every; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::diagnostics_output_every
const REAL NOSIMDdt = commondata->dt;     // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::dt
const REAL_SIMD_ARRAY dt = ConstSIMD(NOSIMDdt);                   // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::dt
const REAL NOSIMDdxx0 = params->dxx0;                             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx0
const REAL_SIMD_ARRAY dxx0 = ConstSIMD(NOSIMDdxx0);               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx0
const REAL NOSIMDdxx1 = params->dxx1;                             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx1
const REAL_SIMD_ARRAY dxx1 = ConstSIMD(NOSIMDdxx1);               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx1
const REAL NOSIMDdxx2 = params->dxx2;                             // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx2
const REAL_SIMD_ARRAY dxx2 = ConstSIMD(NOSIMDdxx2);               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::dxx2
const REAL NOSIMDeta_damping = commondata->eta_damping;           // nrpy.equations.nrpyelliptic.CommonParams::eta_damping
const REAL_SIMD_ARRAY eta_damping = ConstSIMD(NOSIMDeta_damping); // nrpy.equations.nrpyelliptic.CommonParams::eta_damping
const REAL NOSIMDf0_of_xx0 = params->f0_of_xx0;                   // nrpy.reference_metric_SinhSymTP::f0_of_xx0
const REAL_SIMD_ARRAY f0_of_xx0 = ConstSIMD(NOSIMDf0_of_xx0);     // nrpy.reference_metric_SinhSymTP::f0_of_xx0
const REAL NOSIMDf1_of_xx1 = params->f1_of_xx1;                   // nrpy.reference_metric_SinhSymTP::f1_of_xx1
const REAL_SIMD_ARRAY f1_of_xx1 = ConstSIMD(NOSIMDf1_of_xx1);     // nrpy.reference_metric_SinhSymTP::f1_of_xx1
const REAL NOSIMDf2_of_xx0 = params->f2_of_xx0;                   // nrpy.reference_metric_SinhSymTP::f2_of_xx0
const REAL_SIMD_ARRAY f2_of_xx0 = ConstSIMD(NOSIMDf2_of_xx0);     // nrpy.reference_metric_SinhSymTP::f2_of_xx0
const REAL NOSIMDf3_of_xx2 = params->f3_of_xx2;                   // nrpy.reference_metric_SinhSymTP::f3_of_xx2
const REAL_SIMD_ARRAY f3_of_xx2 = ConstSIMD(NOSIMDf3_of_xx2);     // nrpy.reference_metric_SinhSymTP::f3_of_xx2
const REAL NOSIMDf4_of_xx1 = params->f4_of_xx1;                   // nrpy.reference_metric_SinhSymTP::f4_of_xx1
const REAL_SIMD_ARRAY f4_of_xx1 = ConstSIMD(NOSIMDf4_of_xx1);     // nrpy.reference_metric_SinhSymTP::f4_of_xx1
const int grid_idx = params->grid_idx; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::grid_idx
const REAL NOSIMDgrid_physical_size = params->grid_physical_size;               // nrpy.reference_metric::grid_physical_size
const REAL_SIMD_ARRAY grid_physical_size = ConstSIMD(NOSIMDgrid_physical_size); // nrpy.reference_metric::grid_physical_size
const bool grid_rotates = params->grid_rotates;                                 // nrpy.grid::grid_rotates
const REAL NOSIMDinvdxx0 = params->invdxx0;               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx0
const REAL_SIMD_ARRAY invdxx0 = ConstSIMD(NOSIMDinvdxx0); // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx0
const REAL NOSIMDinvdxx1 = params->invdxx1;               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx1
const REAL_SIMD_ARRAY invdxx1 = ConstSIMD(NOSIMDinvdxx1); // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx1
const REAL NOSIMDinvdxx2 = params->invdxx2;               // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx2
const REAL_SIMD_ARRAY invdxx2 = ConstSIMD(NOSIMDinvdxx2); // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::invdxx2
const REAL NOSIMDlog10_current_residual =
    commondata->log10_current_residual; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_current_residual
const REAL_SIMD_ARRAY log10_current_residual =
    ConstSIMD(NOSIMDlog10_current_residual); // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_current_residual
const REAL NOSIMDlog10_residual_tolerance =
    commondata->log10_residual_tolerance; // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_residual_tolerance
const REAL_SIMD_ARRAY log10_residual_tolerance = ConstSIMD(
    NOSIMDlog10_residual_tolerance); // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::log10_residual_tolerance
const REAL NOSIMDMINIMUM_GLOBAL_WAVESPEED =
    commondata->MINIMUM_GLOBAL_WAVESPEED; // nrpy.equations.nrpyelliptic.CommonParams::MINIMUM_GLOBAL_WAVESPEED
const REAL_SIMD_ARRAY MINIMUM_GLOBAL_WAVESPEED =
    ConstSIMD(NOSIMDMINIMUM_GLOBAL_WAVESPEED); // nrpy.equations.nrpyelliptic.CommonParams::MINIMUM_GLOBAL_WAVESPEED
const int nn = commondata->nn;                 // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::nn
const int nn_0 = commondata->nn_0;             // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::nn_0
const int nn_max = commondata->nn_max;         // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::nn_max
const int NUMGRIDS = commondata->NUMGRIDS;     // nrpy.grid::NUMGRIDS
const int Nxx0 = params->Nxx0;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx0
const int Nxx1 = params->Nxx1;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx1
const int Nxx2 = params->Nxx2;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx2
const int Nxx_plus_2NGHOSTS0 =
    params->Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS0
const int Nxx_plus_2NGHOSTS1 =
    params->Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS1
const int Nxx_plus_2NGHOSTS2 =
    params->Nxx_plus_2NGHOSTS2;                     // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS2
const REAL NOSIMDP0_x = commondata->P0_x;           // nrpy.equations.nrpyelliptic.CommonParams::P0_x
const REAL_SIMD_ARRAY P0_x = ConstSIMD(NOSIMDP0_x); // nrpy.equations.nrpyelliptic.CommonParams::P0_x
const REAL NOSIMDP0_y = commondata->P0_y;           // nrpy.equations.nrpyelliptic.CommonParams::P0_y
const REAL_SIMD_ARRAY P0_y = ConstSIMD(NOSIMDP0_y); // nrpy.equations.nrpyelliptic.CommonParams::P0_y
const REAL NOSIMDP0_z = commondata->P0_z;           // nrpy.equations.nrpyelliptic.CommonParams::P0_z
const REAL_SIMD_ARRAY P0_z = ConstSIMD(NOSIMDP0_z); // nrpy.equations.nrpyelliptic.CommonParams::P0_z
const REAL NOSIMDP1_x = commondata->P1_x;           // nrpy.equations.nrpyelliptic.CommonParams::P1_x
const REAL_SIMD_ARRAY P1_x = ConstSIMD(NOSIMDP1_x); // nrpy.equations.nrpyelliptic.CommonParams::P1_x
const REAL NOSIMDP1_y = commondata->P1_y;           // nrpy.equations.nrpyelliptic.CommonParams::P1_y
const REAL_SIMD_ARRAY P1_y = ConstSIMD(NOSIMDP1_y); // nrpy.equations.nrpyelliptic.CommonParams::P1_y
const REAL NOSIMDP1_z = commondata->P1_z;           // nrpy.equations.nrpyelliptic.CommonParams::P1_z
const REAL_SIMD_ARRAY P1_z = ConstSIMD(NOSIMDP1_z); // nrpy.equations.nrpyelliptic.CommonParams::P1_z
const REAL NOSIMDS0_x = commondata->S0_x;           // nrpy.equations.nrpyelliptic.CommonParams::S0_x
const REAL_SIMD_ARRAY S0_x = ConstSIMD(NOSIMDS0_x); // nrpy.equations.nrpyelliptic.CommonParams::S0_x
const REAL NOSIMDS0_y = commondata->S0_y;           // nrpy.equations.nrpyelliptic.CommonParams::S0_y
const REAL_SIMD_ARRAY S0_y = ConstSIMD(NOSIMDS0_y); // nrpy.equations.nrpyelliptic.CommonParams::S0_y
const REAL NOSIMDS0_z = commondata->S0_z;           // nrpy.equations.nrpyelliptic.CommonParams::S0_z
const REAL_SIMD_ARRAY S0_z = ConstSIMD(NOSIMDS0_z); // nrpy.equations.nrpyelliptic.CommonParams::S0_z
const REAL NOSIMDS1_x = commondata->S1_x;           // nrpy.equations.nrpyelliptic.CommonParams::S1_x
const REAL_SIMD_ARRAY S1_x = ConstSIMD(NOSIMDS1_x); // nrpy.equations.nrpyelliptic.CommonParams::S1_x
const REAL NOSIMDS1_y = commondata->S1_y;           // nrpy.equations.nrpyelliptic.CommonParams::S1_y
const REAL_SIMD_ARRAY S1_y = ConstSIMD(NOSIMDS1_y); // nrpy.equations.nrpyelliptic.CommonParams::S1_y
const REAL NOSIMDS1_z = commondata->S1_z;           // nrpy.equations.nrpyelliptic.CommonParams::S1_z
const REAL_SIMD_ARRAY S1_z = ConstSIMD(NOSIMDS1_z); // nrpy.equations.nrpyelliptic.CommonParams::S1_z
const REAL NOSIMDSINHWAA = params->SINHWAA;         // nrpy.reference_metric_SinhSymTP::SINHWAA
const REAL_SIMD_ARRAY SINHWAA = ConstSIMD(NOSIMDSINHWAA); // nrpy.reference_metric_SinhSymTP::SINHWAA
const bool stop_relaxation =
    commondata->stop_relaxation;                  // nrpy.infrastructures.BHaH.nrpyelliptic.base_conformally_flat_C_codegen_library::stop_relaxation
const REAL NOSIMDt_0 = commondata->t_0;           // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_0
const REAL_SIMD_ARRAY t_0 = ConstSIMD(NOSIMDt_0); // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_0
const REAL NOSIMDt_final = commondata->t_final;   // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_final
const REAL_SIMD_ARRAY t_final = ConstSIMD(NOSIMDt_final); // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::t_final
const REAL NOSIMDtime = commondata->time;                 // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::time
const REAL_SIMD_ARRAY time = ConstSIMD(NOSIMDtime);       // nrpy.infrastructures.BHaH.MoLtimestepping.cuda.MoL::time
const REAL NOSIMDxxmax0 = params->xxmax0;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax0
const REAL_SIMD_ARRAY xxmax0 = ConstSIMD(NOSIMDxxmax0);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax0
const REAL NOSIMDxxmax1 = params->xxmax1;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax1
const REAL_SIMD_ARRAY xxmax1 = ConstSIMD(NOSIMDxxmax1);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax1
const REAL NOSIMDxxmax2 = params->xxmax2;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax2
const REAL_SIMD_ARRAY xxmax2 = ConstSIMD(NOSIMDxxmax2);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmax2
const REAL NOSIMDxxmin0 = params->xxmin0;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin0
const REAL_SIMD_ARRAY xxmin0 = ConstSIMD(NOSIMDxxmin0);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin0
const REAL NOSIMDxxmin1 = params->xxmin1;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin1
const REAL_SIMD_ARRAY xxmin1 = ConstSIMD(NOSIMDxxmin1);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin1
const REAL NOSIMDxxmin2 = params->xxmin2;                 // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin2
const REAL_SIMD_ARRAY xxmin2 = ConstSIMD(NOSIMDxxmin2);   // nrpy.infrastructures.BHaH.grid_management.cuda.numerical_grids_and_timestep::xxmin2
const REAL NOSIMDzPunc = commondata->zPunc;               // nrpy.equations.nrpyelliptic.CommonParams::zPunc
const REAL_SIMD_ARRAY zPunc = ConstSIMD(NOSIMDzPunc);     // nrpy.equations.nrpyelliptic.CommonParams::zPunc
