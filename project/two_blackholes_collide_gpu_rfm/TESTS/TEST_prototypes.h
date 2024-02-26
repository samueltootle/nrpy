#ifdef GPU_TESTS
void TEST_commondata(const commondata_struct *restrict commondata, const char* suffix);
void TEST_param_struct(const int grid, const params_struct *restrict commondata, const char* suffix);
void TEST_coord_direction(const int grid, const REAL *restrict xx, const char* dir, const int Nxx);
// void dump_gf_array(const int grid, const params_struct *restrict params, const REAL *restrict gfs, 
//   const char* prefix, const char* suffix, const int numgfs);
#endif