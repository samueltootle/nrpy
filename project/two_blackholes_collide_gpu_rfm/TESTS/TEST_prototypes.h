#ifdef GPU_TESTS
void TEST_commondata(const commondata_struct *restrict commondata, const char* suffix);
void TEST_param_struct(const int grid, const params_struct *restrict commondata, const char* suffix);
// void dump_gf_array(const int grid, const params_struct *restrict params, const REAL *restrict gfs, 
//   const char* prefix, const char* suffix, const int numgfs);
#endif