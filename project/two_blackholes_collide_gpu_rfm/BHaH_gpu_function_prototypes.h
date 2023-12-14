// #define RHS_IMP 4
// #define DEBUG_RHS
// #define DEBUG_IDX

__host__
REAL find_min(REAL * data, uint const data_length);

template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

__global__
void print_data(REAL * data, uint const length);

__host__
REAL reduction_sum(REAL * data, uint const data_length);

__host__
uint reduction_sum(uint * data, uint const data_length);

template<class T>
__global__
void reduction_sum_gpu(T * data, T * sum, uint const data_length);

// __host__
// void testcpy(REAL const * const xx, size_t idx = 43);

// __host__
// void set_fd_constants();
__host__
void set_param_constants(params_struct *restrict params);
// __host__
// void set_commondata_constants(commondata_struct *restrict commondata);

__global__
void print_params();

__device__
void BrillLindquist(const commondata_struct * commondata, const REAL xCart[3], const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data);

__device__
void xx_to_Cart(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);

__device__
void xx_to_Cart__rfm__Spherical(REAL *xx[3],const int i0,const int i1,const int i2, REAL xCart[3]);