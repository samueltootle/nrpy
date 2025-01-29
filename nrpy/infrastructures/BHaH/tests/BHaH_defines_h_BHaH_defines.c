#ifndef __BHAH_DEFINES_H__
#define __BHAH_DEFINES_H__
// BHaH core header file, automatically generated from output_BHaH_defines_h within BHaH_defines_h.py,
//    DO NOT EDIT THIS FILE BY HAND.

//********************************************
// Basic definitions for module general:
#include <ctype.h>   // Character type functions, such as isdigit, isalpha, etc.
#include <errno.h>   // Error number definitions
#include <math.h>    // Transcendental functions, etc.
#include <stdbool.h> // bool-typed variables
#include <stdint.h>  // int8_t-typed variables
#include <stdio.h>   // Basic input/output functions, such as *printf, fopen, fwrite, etc.
#include <stdlib.h>  // malloc/free, etc.
#include <string.h>  // String handling functions, such as strlen, strcmp, etc.
#include <time.h>    // Time-related functions and types, such as time(), clock(),
// output_BHaH_defines_h(...,enable_intrinsics=True) was called so we intrinsics headers:
#include "intrinsics/simd_intrinsics.h"
#define REAL double
#define DOUBLE double

// These macros for MIN(), MAX(), and SQR() ensure that if the arguments inside
//   are a function/complex expression, the function/expression is evaluated
//   *only once* per argument. See https://lwn.net/Articles/983965/ for details.
// They are improvements over the original implementations:
// #define MIN(A, B) ( ((A) < (B)) ? (A) : (B) )
// #define MAX(A, B) ( ((A) > (B)) ? (A) : (B) )
// #define SQR(A) ((A) * (A))
#define MIN(A, B)                                                                                                                                    \
  ({                                                                                                                                                 \
    __typeof__(A) _a = (A);                                                                                                                          \
    __typeof__(B) _b = (B);                                                                                                                          \
    _a < _b ? _a : _b;                                                                                                                               \
  })
#define MAX(A, B)                                                                                                                                    \
  ({                                                                                                                                                 \
    __typeof__(A) _a = (A);                                                                                                                          \
    __typeof__(B) _b = (B);                                                                                                                          \
    _a > _b ? _a : _b;                                                                                                                               \
  })
#define SQR(A)                                                                                                                                       \
  ({                                                                                                                                                 \
    __typeof__(A) _a = (A);                                                                                                                          \
    _a *_a;                                                                                                                                          \
  })
#ifndef MAYBE_UNUSED
#if __cplusplus >= 201703L
#define MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif // END check for GCC, Clang, or NVCC
#endif // END MAYBE_UNUSED
// START: CodeParameters declared as #define.
#ifndef MAXNUMGRIDS
#define MAXNUMGRIDS 15 // nrpy.grid
#endif
// END: CodeParameters declared as #define.

//********************************************
// Basic definitions for module commondata_struct:
typedef struct __commondata_struct__ {
  REAL CFL_FACTOR; // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  REAL dt;         // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  REAL t_0;        // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  REAL t_final;    // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  REAL time;       // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  int NUMGRIDS;    // (nrpy.grid)
  int nn;          // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
  int nn_0;        // (nrpy.infrastructures.BHaH.MoLtimestepping.MoL)
} commondata_struct;

//********************************************
// Basic definitions for module params_struct:
typedef struct __params_struct__ {
  REAL Cart_originx; // (nrpy.grid)
  REAL Cart_originy; // (nrpy.grid)
  REAL Cart_originz; // (nrpy.grid)
  bool grid_rotates; // (nrpy.grid)
} params_struct;

//********************************************
// Basic definitions for module finite_difference:

// Set the number of ghost zones
// Note that upwinding in e.g., BSSN requires that NGHOSTS = fd_order/2 + 1 <- Notice the +1.
#define NGHOSTS 2

// Declare NO_INLINE macro, used in FD functions. GCC v10+ compilations hang on complex RHS expressions (like BSSN) without this.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define NO_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NO_INLINE __declspec(noinline)
#else
#define NO_INLINE // Fallback for unknown compilers
#endif

//********************************************
// Basic definitions for module nrpy.infrastructures.BHaH.MoLtimestepping.MoL:
typedef struct __MoL_gridfunctions_struct__ {
  REAL *restrict y_n_gfs;
  REAL *restrict y_nplus1_running_total_gfs;
  REAL *restrict k_odd_gfs;
  REAL *restrict k_even_gfs;
  REAL *restrict auxevol_gfs;
  REAL *restrict diagnostic_output_gfs;
  REAL *restrict diagnostic_output_gfs2;
} MoL_gridfunctions_struct;

//********************************************
// Basic definitions for module grid:

// EVOL VARIABLES:
#define NUM_EVOL_GFS 0

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
#define IDX4(gf, i, j, k) ((i) + Nxx_plus_2NGHOSTS0 * ((j) + Nxx_plus_2NGHOSTS1 * ((k) + Nxx_plus_2NGHOSTS2 * (gf))))
#define IDX4pt(gf, idx) ((idx) + (Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2) * (gf))
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
  REAL *restrict xx[3];
  // NRPy+ MODULE: nrpy.infrastructures.BHaH.MoLtimestepping.MoL
  MoL_gridfunctions_struct gridfuncs; // <- MoL gridfunctions
  // NRPy+ MODULE: params
  params_struct params; // <- BHaH parameters, generated from NRPy+'s CodeParameters
} griddata_struct;

#endif
