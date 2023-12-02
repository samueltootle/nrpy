// Note this can only be #include by a *.cu file

extern __constant__ REAL wavespeed;
extern __constant__ REAL FDPart1_Rational_5_2; 
extern __constant__ REAL FDPart1_Rational_1_12;
extern __constant__ REAL FDPart1_Rational_4_3;
extern __constant__ params_struct d_params;



#define GPU_NGRID0 8
#define GPU_NGRID1 8
#define GPU_NGRID2 8
#define GPU_NBLOCK0 32
#define GPU_NBLOCK1 4
#define GPU_NBLOCK2 2
#define SHARED_SIZE_LIMIT 1024U

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

