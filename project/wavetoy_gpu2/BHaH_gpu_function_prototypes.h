template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

__host__
REAL find_min(REAL * data, uint const data_length);

__host__
void testcpy(REAL const * const xx);

__device__
void exact_solution_single_Cartesian_point_gpu(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const REAL xCart0, const REAL xCart1, const REAL xCart2,  REAL *restrict exact_soln_UUGF, REAL *restrict exact_soln_VVGF
);