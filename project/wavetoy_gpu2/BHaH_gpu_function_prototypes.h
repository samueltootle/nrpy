#define RHS_IMP 4
// #define DEBUG_RHS
// #define DEBUG_IDX

template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

__host__
REAL find_min(REAL * data, uint const data_length);

__host__
void testcpy(REAL const * const xx, size_t idx = 43);

__device__
void exact_solution_single_Cartesian_point_gpu(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const REAL xCart0, const REAL xCart1, const REAL xCart2,  REAL *restrict exact_soln_UUGF, REAL *restrict exact_soln_VVGF
);

__global__
void rk_substep1_gpu(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N);
__host__
void rk_substep1(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) ;

__global__
void rk_substep2_gpu(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N);
__host__
void rk_substep2(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) ;

__global__
void rk_substep3_gpu(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N);
__host__
void rk_substep3(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) ;

__global__
void rk_substep4_gpu(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs,
                REAL const dt,
                size_t const N);
__host__
void rk_substep4(params_struct *restrict params,
                REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                REAL *restrict k_even_gfs,
                REAL *restrict auxevol_gfs, REAL const dt) ;

__host__
void set_fd_constants();
__host__
void set_param_constants(params_struct *restrict params);
__host__
void set_commondata_constants(commondata_struct *restrict commondata);

#if RHS_IMP == 1
// Original RHS
__global__
void rhs_eval_gpu(const commondata_struct *restrict commondata, 
              const params_struct *restrict params, 
              const REAL *restrict in_gfs,
              REAL *restrict rhs_gfs);

#elif RHS_IMP == 2
// Shared Memory setup
__global__ void compute_uu_dDDxx_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDxx(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS0);

__global__ void compute_uu_dDDyy_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDyy(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS1);

__global__ void compute_uu_dDDzz_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDzz(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS2);

__global__ 
void compute_rhs_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 const REAL *restrict in_gfs_derivatives,
                                 REAL *restrict out_gfs);

__host__ 
void compute_rhs(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          const REAL *restrict aux_gfs,
                          REAL *restrict out_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2);
#elif RHS_IMP == 2
// Shared Memory setup
__global__ void compute_uu_dDDxx_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDxx(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS0);
                          
__global__ void compute_uu_dDDyy_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDyy(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS1);

__global__ void compute_uu_dDDzz_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDDzz(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS2);

__global__ 
void compute_rhs_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 const REAL *restrict in_gfs_derivatives,
                                 REAL *restrict out_gfs);

__host__ 
void compute_rhs(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          const REAL *restrict aux_gfs,
                          REAL *restrict out_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2);
#else
// Shared Memory setup
__global__ void compute_uu_dDD_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);

__host__ 
void compute_uu_dDD(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          REAL *restrict aux_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2,
                          const int Nxx_plus_2NGHOSTS0,
                          const int Nxx_plus_2NGHOSTS1,
                          const int Nxx_plus_2NGHOSTS2);

__global__ 
void compute_rhs_gpu(const params_struct *restrict params, 
                                 const REAL *restrict in_gfs,
                                 const REAL *restrict in_gfs_derivatives,
                                 REAL *restrict out_gfs);

__host__ 
void compute_rhs(const params_struct *restrict params, 
                          const REAL *restrict in_gfs,
                          const REAL *restrict aux_gfs,
                          REAL *restrict out_gfs,
                          const int Nxx0,
                          const int Nxx1,
                          const int Nxx2);
#endif