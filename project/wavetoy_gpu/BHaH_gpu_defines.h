// Note this can only be #include by a *.cu file

extern __constant__ REAL wavespeed;
extern __constant__ REAL FDPart1_Rational_5_2; 
extern __constant__ REAL FDPart1_Rational_1_12;
extern __constant__ REAL FDPart1_Rational_4_3;

__device__ __host__
void exact_solution_single_Cartesian_point(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const REAL xCart0, const REAL xCart1, const REAL xCart2,  REAL *restrict exact_soln_UUGF, REAL *restrict exact_soln_VVGF
);

// __global__
// void rhs_eval(const commondata_struct *restrict commondata, 
//               const params_struct *restrict params, 
//               const REAL *restrict in_gfs,
//               REAL *restrict rhs_gfs);

__global__
void rk_substep_gpu(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                MoL_gridfunctions_struct *restrict gridfuncs,
                REAL rk_weight,
                REAL dt_step_factor);
__host__
void rk_substep(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                MoL_gridfunctions_struct *restrict gridfuncs,
                REAL rk_weight,
                REAL dt_step_factor);
__host__
void set_fd_constants();

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


template<class T>
__global__
void print_var(T * data, size_t index) {
    printf("%f\n", data[index]);
}