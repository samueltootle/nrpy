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
    printf("val: %1.15e\n", data[index]);
}

__host__ __device__
void BrillLindquist(const commondata_struct * commondata, const REAL xCart[3], const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data);

__device__
void xx_to_Cart(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);

__device__ __host__
void xx_to_Cart__rfm__Spherical(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);

__device__
void Cart_to_xx_and_nearest_i0i1i2(const commondata_struct *restrict commondata, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
__device__
void Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical(const commondata_struct *restrict commondata, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);



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

__host__
void cpyDevicetoHost__grid(const commondata_struct *restrict commondata, 
                          griddata_struct *restrict gd_host,
                          const griddata_struct *restrict gd_gpu);
__host__
size_t cpyDevicetoHost__gf(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        REAL * gf_host,
                        const REAL *gf_gpu,
                        const int host_GF_IDX,
                        const int gpu_GF_IDX);

void cpyDevicetoHost__malloc_y_n_gfs(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs);

__host__
void cpyDevicetoHost__malloc_diag_gfs(const commondata_struct *restrict commondata,
                        const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs);

__host__
void cpyDevicetoHost__free_gfs(MoL_gridfunctions_struct *restrict gfs_host);

__host__
void freeHostgrid(griddata_struct *restrict gd_host);