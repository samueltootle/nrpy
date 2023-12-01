// BHaH core header file, automatically generated from output_BHaH_defines_h within BHaH_defines_h.py,
//    DO NOT EDIT THIS FILE BY HAND.

//********************************************
// Basic definitions for module general:
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __cplusplus
#define restrict __restrict__
#endif
#define REAL double

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

//********************************************
// Basic definitions for module nrpy.infrastructures.BHaH.diagnostics.progress_indicator:
#ifdef __linux__
// Timer with nanosecond resolution. Only on Linux.
#define TIMEVAR struct timespec
#define CURRTIME_FUNC(currtime) clock_gettime(CLOCK_REALTIME, currtime)
#define TIME_IN_NS(start, end) (REAL)(1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
#else
// Low-resolution timer, 1-second resolution. Widely available.
#define TIMEVAR time_t
#define CURRTIME_FUNC(currtime) time(currtime)
#define TIME_IN_NS(start, end) (REAL)(difftime(end, start) * 1.0e9 + 1e-6) // Round up to avoid divide-by-zero.
#endif

//********************************************
// Basic definitions for module commondata_struct:
typedef struct __commondata_struct__ {
  REAL CFL_FACTOR;               // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::CFL_FACTOR
  REAL convergence_factor;       // __main__::convergence_factor
  REAL diagnostics_output_every; // __main__::diagnostics_output_every
  REAL dt;                       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::dt
  REAL sigma;                    // nrpy.equations.wave_equation.WaveEquation_Solutions_InitialData::sigma
  REAL t_0;                      // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_0
  REAL t_final;                  // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::t_final
  REAL time;                     // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::time
  REAL wavespeed;                // nrpy.equations.wave_equation.CommonParams::wavespeed
  TIMEVAR start_wallclock_time;  // nrpy.infrastructures.BHaH.diagnostics.progress_indicator::start_wallclock_time
  int NUMGRIDS;                  // nrpy.grid::NUMGRIDS
  int nn;                        // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn
  int nn_0;                      // nrpy.infrastructures.BHaH.MoLtimestepping.MoL::nn_0
} commondata_struct;

//********************************************
// Basic definitions for module params_struct:
typedef struct __params_struct__ {
  REAL Cart_originx;      // nrpy.grid::Cart_originx
  REAL Cart_originy;      // nrpy.grid::Cart_originy
  REAL Cart_originz;      // nrpy.grid::Cart_originz
  REAL dxx0;              // __main__::dxx0
  REAL dxx1;              // __main__::dxx1
  REAL dxx2;              // __main__::dxx2
  REAL invdxx0;           // __main__::invdxx0
  REAL invdxx1;           // __main__::invdxx1
  REAL invdxx2;           // __main__::invdxx2
  REAL xxmax0;            // __main__::xxmax0
  REAL xxmax1;            // __main__::xxmax1
  REAL xxmax2;            // __main__::xxmax2
  REAL xxmin0;            // __main__::xxmin0
  REAL xxmin1;            // __main__::xxmin1
  REAL xxmin2;            // __main__::xxmin2
  int Nxx0;               // __main__::Nxx0
  int Nxx1;               // __main__::Nxx1
  int Nxx2;               // __main__::Nxx2
  int Nxx_plus_2NGHOSTS0; // __main__::Nxx_plus_2NGHOSTS0
  int Nxx_plus_2NGHOSTS1; // __main__::Nxx_plus_2NGHOSTS1
  int Nxx_plus_2NGHOSTS2; // __main__::Nxx_plus_2NGHOSTS2
} params_struct;

//********************************************
// Basic definitions for module finite_difference:

// Set the number of ghost zones
// Note that upwinding in e.g., BSSN requires that NGHOSTS = fd_order/2 + 1 <- Notice the +1.
#define NGHOSTS 2

// When enable_simd = False, this is the UPWIND_ALG() macro:
#define UPWIND_ALG(UpwindVecU) UpwindVecU > 0.0 ? 1.0 : 0.0

//********************************************
// Basic definitions for module nrpy.infrastructures.BHaH.MoLtimestepping.MoL:
typedef struct __MoL_gridfunctions_struct__ {
  REAL *y_n_gfs;
  REAL *y_nplus1_running_total_gfs;
  REAL *k_odd_gfs;
  REAL *k_even_gfs;
  REAL *auxevol_gfs;
  REAL *diagnostic_output_gfs;
  REAL *diagnostic_output_gfs2;
} MoL_gridfunctions_struct;

#define LOOP_ALL_GFS_GPS(ii)                                                                                                                         \
  _Pragma("omp parallel for") for (int(ii) = 0; (ii) < Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS; (ii)++)

//********************************************
// Basic definitions for module grid:

// EVOL VARIABLES:
#define NUM_EVOL_GFS 2
#define UUGF 0
#define VVGF 1

// SET gridfunctions_f_infinity[i] = evolved gridfunction i's value in the limit r->infinity:
static const REAL gridfunctions_f_infinity[NUM_EVOL_GFS] = {2.0, 0.0};

// SET gridfunctions_wavespeed[i] = evolved gridfunction i's characteristic wave speed:
static const REAL gridfunctions_wavespeed[NUM_EVOL_GFS] = {1.0, 1.0};

// AUX VARIABLES:
#define NUM_AUX_GFS 0

// AUXEVOL VARIABLES:
#define NUM_AUXEVOL_GFS 0

// Declare the IDX4(gf,i,j,k) macro, which enables us to store 4-dimensions of
//   data in a 1D array. In this case, consecutive values of "i"
//   (all other indices held to a fixed value) are consecutive in memory, where
//   consecutive values of "j" (fixing all other indices) are separated by
//   Nxx_plus_2NGHOSTS0 elements in memory. Similarly, consecutive values of
//   "k" are separated by Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1 in memory, etc.
#define IDX4(g, i, j, k) ((i) + Nxx_plus_2NGHOSTS0 * ((j) + Nxx_plus_2NGHOSTS1 * ((k) + Nxx_plus_2NGHOSTS2 * (g))))
#define IDX4pt(g, idx) ((idx) + (Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2) * (g))
#define IDX3(i, j, k) ((i) + Nxx_plus_2NGHOSTS0 * ((j) + Nxx_plus_2NGHOSTS1 * ((k))))
#define LOOP_REGION(i0min, i0max, i1min, i1max, i2min, i2max)                                                                                        \
  for (int i2 = i2min; i2 < i2max; i2++)                                                                                                             \
    for (int i1 = i1min; i1 < i1max; i1++)                                                                                                           \
      for (int i0 = i0min; i0 < i0max; i0++)
#define LOOP_OMP(__OMP_PRAGMA__, i0, i0min, i0max, i1, i1min, i1max, i2, i2min, i2max)                                                               \
  _Pragma(__OMP_PRAGMA__) for (int(i2) = (i2min); (i2) < (i2max); (i2)++) for (int(i1) = (i1min); (i1) < (i1max);                                    \
                                                                               (i1)++) for (int(i0) = (i0min); (i0) < (i0max); (i0)++)
#define LOOP_NOOMP(i0, i0min, i0max, i1, i1min, i1max, i2, i2min, i2max)                                                                             \
  for (int(i2) = (i2min); (i2) < (i2max); (i2)++)                                                                                                    \
    for (int(i1) = (i1min); (i1) < (i1max); (i1)++)                                                                                                  \
      for (int(i0) = (i0min); (i0) < (i0max); (i0)++)
#define LOOP_BREAKOUT(i0, i1, i2, i0max, i1max, i2max)                                                                                               \
  {                                                                                                                                                  \
    i0 = (i0max);                                                                                                                                    \
    i1 = (i1max);                                                                                                                                    \
    i2 = (i2max);                                                                                                                                    \
    break;                                                                                                                                           \
  }
#define IS_IN_GRID_INTERIOR(i0i1i2, Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2, NG)                                                  \
  (i0i1i2[0] >= (NG) && i0i1i2[0] < (Nxx_plus_2NGHOSTS0) - (NG) && i0i1i2[1] >= (NG) && i0i1i2[1] < (Nxx_plus_2NGHOSTS1) - (NG) &&                   \
   i0i1i2[2] >= (NG) && i0i1i2[2] < (Nxx_plus_2NGHOSTS2) - (NG))

typedef struct __griddata__ {
  // griddata_struct stores data needed on each grid
  // xx[3] stores the uniform grid coordinates.
  REAL * xx[3];
  // NRPy+ MODULE: nrpy.infrastructures.BHaH.MoLtimestepping.MoL
  MoL_gridfunctions_struct gridfuncs; // <- MoL gridfunctions
  // NRPy+ MODULE: params
  params_struct* params; // <- BHaH parameters, generated from NRPy+'s CodeParameters
} griddata_struct;
