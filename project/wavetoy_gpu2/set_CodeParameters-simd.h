const REAL NOSIMDCart_originx = params->Cart_originx;                             // nrpy.grid::Cart_originx
const REAL_SIMD_ARRAY Cart_originx = ConstSIMD(NOSIMDCart_originx);               // nrpy.grid::Cart_originx
const REAL NOSIMDCart_originy = params->Cart_originy;                             // nrpy.grid::Cart_originy
const REAL_SIMD_ARRAY Cart_originy = ConstSIMD(NOSIMDCart_originy);               // nrpy.grid::Cart_originy
const REAL NOSIMDCart_originz = params->Cart_originz;                             // nrpy.grid::Cart_originz
const REAL_SIMD_ARRAY Cart_originz = ConstSIMD(NOSIMDCart_originz);               // nrpy.grid::Cart_originz
const REAL NOSIMDCFL_FACTOR = commondata->CFL_FACTOR;                             // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
const REAL_SIMD_ARRAY CFL_FACTOR = ConstSIMD(NOSIMDCFL_FACTOR);                   // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
const REAL NOSIMDconvergence_factor = commondata->convergence_factor;             // __main__::convergence_factor
const REAL_SIMD_ARRAY convergence_factor = ConstSIMD(NOSIMDconvergence_factor);   // __main__::convergence_factor
const REAL NOSIMDdiagnostics_output_every = commondata->diagnostics_output_every; // __main__::diagnostics_output_every
const REAL_SIMD_ARRAY diagnostics_output_every = ConstSIMD(NOSIMDdiagnostics_output_every); // __main__::diagnostics_output_every
const REAL NOSIMDdt = commondata->dt;                                                       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::dt
const REAL_SIMD_ARRAY dt = ConstSIMD(NOSIMDdt);                                             // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::dt
const REAL NOSIMDdxx0 = params->dxx0;                                                       // __main__::dxx0
const REAL_SIMD_ARRAY dxx0 = ConstSIMD(NOSIMDdxx0);                                         // __main__::dxx0
const REAL NOSIMDdxx1 = params->dxx1;                                                       // __main__::dxx1
const REAL_SIMD_ARRAY dxx1 = ConstSIMD(NOSIMDdxx1);                                         // __main__::dxx1
const REAL NOSIMDdxx2 = params->dxx2;                                                       // __main__::dxx2
const REAL_SIMD_ARRAY dxx2 = ConstSIMD(NOSIMDdxx2);                                         // __main__::dxx2
const REAL NOSIMDinvdxx0 = params->invdxx0;                                                 // __main__::invdxx0
const REAL_SIMD_ARRAY invdxx0 = ConstSIMD(NOSIMDinvdxx0);                                   // __main__::invdxx0
const REAL NOSIMDinvdxx1 = params->invdxx1;                                                 // __main__::invdxx1
const REAL_SIMD_ARRAY invdxx1 = ConstSIMD(NOSIMDinvdxx1);                                   // __main__::invdxx1
const REAL NOSIMDinvdxx2 = params->invdxx2;                                                 // __main__::invdxx2
const REAL_SIMD_ARRAY invdxx2 = ConstSIMD(NOSIMDinvdxx2);                                   // __main__::invdxx2
const int nn = commondata->nn;                                                              // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn
const int nn_0 = commondata->nn_0;                                                          // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn_0
const int NUMGRIDS = commondata->NUMGRIDS;                                                  // nrpy.grid::NUMGRIDS
const int Nxx0 = params->Nxx0;                                                              // __main__::Nxx0
const int Nxx1 = params->Nxx1;                                                              // __main__::Nxx1
const int Nxx2 = params->Nxx2;                                                              // __main__::Nxx2
const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;                                  // __main__::Nxx_plus_2NGHOSTS0
const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;                                  // __main__::Nxx_plus_2NGHOSTS1
const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;                                  // __main__::Nxx_plus_2NGHOSTS2
const REAL NOSIMDsigma = commondata->sigma;                   // nrpy.equations.wave_equation.WaveEquation_Solutions_InitialData::sigma
const REAL_SIMD_ARRAY sigma = ConstSIMD(NOSIMDsigma);         // nrpy.equations.wave_equation.WaveEquation_Solutions_InitialData::sigma
const REAL NOSIMDt_0 = commondata->t_0;                       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_0
const REAL_SIMD_ARRAY t_0 = ConstSIMD(NOSIMDt_0);             // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_0
const REAL NOSIMDt_final = commondata->t_final;               // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
const REAL_SIMD_ARRAY t_final = ConstSIMD(NOSIMDt_final);     // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
const REAL NOSIMDtime = commondata->time;                     // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::time
const REAL_SIMD_ARRAY time = ConstSIMD(NOSIMDtime);           // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::time
const REAL NOSIMDwavespeed = commondata->wavespeed;           // nrpy.equations.wave_equation.CommonParams::wavespeed
const REAL_SIMD_ARRAY wavespeed = ConstSIMD(NOSIMDwavespeed); // nrpy.equations.wave_equation.CommonParams::wavespeed
const REAL NOSIMDxxmax0 = params->xxmax0;                     // __main__::xxmax0
const REAL_SIMD_ARRAY xxmax0 = ConstSIMD(NOSIMDxxmax0);       // __main__::xxmax0
const REAL NOSIMDxxmax1 = params->xxmax1;                     // __main__::xxmax1
const REAL_SIMD_ARRAY xxmax1 = ConstSIMD(NOSIMDxxmax1);       // __main__::xxmax1
const REAL NOSIMDxxmax2 = params->xxmax2;                     // __main__::xxmax2
const REAL_SIMD_ARRAY xxmax2 = ConstSIMD(NOSIMDxxmax2);       // __main__::xxmax2
const REAL NOSIMDxxmin0 = params->xxmin0;                     // __main__::xxmin0
const REAL_SIMD_ARRAY xxmin0 = ConstSIMD(NOSIMDxxmin0);       // __main__::xxmin0
const REAL NOSIMDxxmin1 = params->xxmin1;                     // __main__::xxmin1
const REAL_SIMD_ARRAY xxmin1 = ConstSIMD(NOSIMDxxmin1);       // __main__::xxmin1
const REAL NOSIMDxxmin2 = params->xxmin2;                     // __main__::xxmin2
const REAL_SIMD_ARRAY xxmin2 = ConstSIMD(NOSIMDxxmin2);       // __main__::xxmin2
