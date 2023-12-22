// Note this can only be #include by a *.cu file
extern __constant__ params_struct d_params;
extern __constant__ int8_t d_evol_gf_parity[24];
extern __constant__ REAL FDPart1_Rational_1_2;
extern __constant__ REAL FDPart1_Rational_1_4;
extern __constant__ REAL FDPart1_Rational_1_12;
extern __constant__ REAL FDPart1_Rational_1_18;
extern __constant__ REAL FDPart1_Rational_1_144;
extern __constant__ REAL FDPart1_Rational_2_3;
extern __constant__ REAL FDPart1_Rational_3_2;
extern __constant__ REAL FDPart1_Rational_4_3;
extern __constant__ REAL FDPart1_Rational_4_9;
extern __constant__ REAL FDPart1_Rational_5_2;
extern __constant__ REAL FDPart1_Rational_5_6;
extern __constant__ REAL d_gridfunctions_wavespeed[NUM_EVOL_GFS];
extern __constant__ REAL d_gridfunctions_f_infinity[NUM_EVOL_GFS];
extern cudaStream_t stream1, stream2, stream3;


typedef void (*ID_pfunc)(const commondata_struct *restrict commondata, const REAL xCart[3],
  const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data);


#define GPU_NGRID0 8
#define GPU_NGRID1 8
#define GPU_NGRID2 8
#define GPU_NBLOCK0 32
#define GPU_NBLOCK1 4
#define GPU_NBLOCK2 2
#define SHARED_SIZE_LIMIT 1024U
#define PENCIL_SIZEY 4

// error checking macro
#define cudaCheckErrors(v, msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s %s (%s at %s:%d)\n", \
                #v, msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0);


