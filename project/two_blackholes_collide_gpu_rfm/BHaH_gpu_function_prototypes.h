// #define RHS_IMP 4
// #define DEBUG_RHS
// #define DEBUG_IDX

__host__
REAL find_min(REAL * data, uint const data_length);

template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

template<class T>
__global__
void print_data(T * data, uint const length) {
    for(int i = 0; i < length; ++i) {
        printf("%1.15e\n", data[i]);
    }
}

__host__
REAL reduction_sum(REAL * data, uint const data_length);

__host__
uint reduction_sum(uint * data, uint const data_length);

template<class T>
__global__
void reduction_sum_gpu(T * data, T * sum, uint const data_length);

// __host__
// void testcpy(REAL const * const xx, size_t idx = 43);

__host__
void set_fd_constants();
__host__
void set_param_constants(const params_struct *restrict params);
// __host__
// void set_commondata_constants(commondata_struct *restrict commondata);

__global__
void print_params();

template<class T>
__global__
void print_var(T * data, size_t index) {
    printf("%1.15e\n", data[index]);
}

__host__ __device__
void BrillLindquist(const commondata_struct * commondata, const REAL xCart[3], const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data);

__device__
void xx_to_Cart(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);

__device__
void xx_to_Cart__rfm__Spherical(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);

__device__
void Cart_to_xx_and_nearest_i0i1i2(const commondata_struct *restrict commondata, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
__device__
void Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical(const commondata_struct *restrict commondata, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);

__device__ REAL fd_function_dD0_fdorder4(const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1, const REAL FDPROTO_i0p2,
                                     const REAL invdxx0);

__device__ REAL fd_function_dD1_fdorder4(const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1, const REAL FDPROTO_i1p2,
                                     const REAL invdxx1);

__device__ REAL fd_function_dD2_fdorder4(const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1, const REAL FDPROTO_i2p2,
                                     const REAL invdxx2);

__device__ REAL fd_function_dDD00_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i0m1, const REAL FDPROTO_i0m2, const REAL FDPROTO_i0p1,
                                       const REAL FDPROTO_i0p2, const REAL invdxx0);

__device__ REAL fd_function_dDD01_fdorder4(const REAL FDPROTO_i0m1_i1m1, const REAL FDPROTO_i0m1_i1m2, const REAL FDPROTO_i0m1_i1p1,
                                       const REAL FDPROTO_i0m1_i1p2, const REAL FDPROTO_i0m2_i1m1, const REAL FDPROTO_i0m2_i1m2,
                                       const REAL FDPROTO_i0m2_i1p1, const REAL FDPROTO_i0m2_i1p2, const REAL FDPROTO_i0p1_i1m1,
                                       const REAL FDPROTO_i0p1_i1m2, const REAL FDPROTO_i0p1_i1p1, const REAL FDPROTO_i0p1_i1p2,
                                       const REAL FDPROTO_i0p2_i1m1, const REAL FDPROTO_i0p2_i1m2, const REAL FDPROTO_i0p2_i1p1,
                                       const REAL FDPROTO_i0p2_i1p2, const REAL invdxx0, const REAL invdxx1);

__device__ REAL fd_function_dDD02_fdorder4(const REAL FDPROTO_i0m1_i2m1, const REAL FDPROTO_i0m1_i2m2, const REAL FDPROTO_i0m1_i2p1,
                                       const REAL FDPROTO_i0m1_i2p2, const REAL FDPROTO_i0m2_i2m1, const REAL FDPROTO_i0m2_i2m2,
                                       const REAL FDPROTO_i0m2_i2p1, const REAL FDPROTO_i0m2_i2p2, const REAL FDPROTO_i0p1_i2m1,
                                       const REAL FDPROTO_i0p1_i2m2, const REAL FDPROTO_i0p1_i2p1, const REAL FDPROTO_i0p1_i2p2,
                                       const REAL FDPROTO_i0p2_i2m1, const REAL FDPROTO_i0p2_i2m2, const REAL FDPROTO_i0p2_i2p1,
                                       const REAL FDPROTO_i0p2_i2p2, const REAL invdxx0, const REAL invdxx2);

__device__ REAL fd_function_dDD11_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i1m1, const REAL FDPROTO_i1m2, const REAL FDPROTO_i1p1,
                                       const REAL FDPROTO_i1p2, const REAL invdxx1);

__device__ REAL fd_function_dDD12_fdorder4(const REAL FDPROTO_i1m1_i2m1, const REAL FDPROTO_i1m1_i2m2, const REAL FDPROTO_i1m1_i2p1,
                                       const REAL FDPROTO_i1m1_i2p2, const REAL FDPROTO_i1m2_i2m1, const REAL FDPROTO_i1m2_i2m2,
                                       const REAL FDPROTO_i1m2_i2p1, const REAL FDPROTO_i1m2_i2p2, const REAL FDPROTO_i1p1_i2m1,
                                       const REAL FDPROTO_i1p1_i2m2, const REAL FDPROTO_i1p1_i2p1, const REAL FDPROTO_i1p1_i2p2,
                                       const REAL FDPROTO_i1p2_i2m1, const REAL FDPROTO_i1p2_i2m2, const REAL FDPROTO_i1p2_i2p1,
                                       const REAL FDPROTO_i1p2_i2p2, const REAL invdxx1, const REAL invdxx2);

__device__ REAL fd_function_dDD22_fdorder4(const REAL FDPROTO, const REAL FDPROTO_i2m1, const REAL FDPROTO_i2m2, const REAL FDPROTO_i2p1,
                                       const REAL FDPROTO_i2p2, const REAL invdxx2);


__global__
void rk_substep1_gpu(const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
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
void rk_substep2_gpu(const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                const REAL *restrict k_odd_gfs,
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
void rk_substep3_gpu(const REAL *restrict y_n_gfs,
                REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
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
void rk_substep4_gpu(REAL *restrict y_n_gfs,
                const REAL *restrict y_nplus1_running_total_gfs,
                REAL *restrict k_odd_gfs,
                const REAL *restrict k_even_gfs,
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