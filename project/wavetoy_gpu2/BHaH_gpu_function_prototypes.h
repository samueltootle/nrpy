template<class T>
__global__
void find_min_cu(T * data, unsigned long long int * min, uint const data_length);

__host__
REAL find_min(REAL * data, uint const data_length);

__host__
void testcpy(REAL* xx);