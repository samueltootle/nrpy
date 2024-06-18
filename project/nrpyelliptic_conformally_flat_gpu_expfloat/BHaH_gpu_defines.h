// BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
//    DO NOT EDIT THIS FILE BY HAND.

// Standard macro definitions
// We include the macro definition nstreams since it is used for calculations in various
// algorithms in addition to defining the streams array
#define nstreams 3

// Standard declarations
// Device storage for grid parameters
extern __constant__ params_struct d_params;
// Device storage for commondata
extern __constant__ commondata_struct d_commondata;
// Device storage for grid function parity
extern __constant__ int8_t d_evol_gf_parity[24];
extern cudaStream_t streams[nstreams];
extern size_t GPU_N_SMS;
extern __constant__ REAL d_gridfunctions_wavespeed[NUM_EVOL_GFS];
extern __constant__ REAL d_gridfunctions_f_infinity[NUM_EVOL_GFS];

// CUDA Error checking macro only active if compiled with -DDEBUG
// Otherwise additional synchronization overhead will occur
#define DEBUG
#ifdef DEBUG
#define cudaCheckErrors(v, msg)                                                                                                                      \
  do {                                                                                                                                               \
    cudaError_t __err = cudaGetLastError();                                                                                                          \
    if (__err != cudaSuccess) {                                                                                                                      \
      fprintf(stderr, "Fatal error: %s %s (%s at %s:%d)\n", #v, msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                 \
      fprintf(stderr, "*** FAILED - ABORTING\n");                                                                                                    \
      exit(1);                                                                                                                                       \
    }                                                                                                                                                \
  } while (0);
#else
#define cudaCheckErrors(v, msg)
#endif