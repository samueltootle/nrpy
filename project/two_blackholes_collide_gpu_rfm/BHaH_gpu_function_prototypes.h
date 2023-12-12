// #define RHS_IMP 4
// #define DEBUG_RHS
// #define DEBUG_IDX

// template<class T>
// __global__
// void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

// __host__
// REAL find_min(REAL * data, uint const data_length);

// __host__
// void testcpy(REAL const * const xx, size_t idx = 43);

// __host__
// void set_fd_constants();
__host__
void set_param_constants(params_struct *restrict params);
// __host__
// void set_commondata_constants(commondata_struct *restrict commondata);
