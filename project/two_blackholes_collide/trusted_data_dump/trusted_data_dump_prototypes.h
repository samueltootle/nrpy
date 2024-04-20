#ifdef GPU_TESTS
void dump_common_data(const commondata_struct *restrict commondata, const char* suffix);
void dump_param_struct(const int grid, const params_struct *restrict commondata, const char* suffix);
void dump_gf_array(const int grid, const params_struct *restrict params, const REAL *restrict gfs, 
  const char* prefix, const char* suffix, const int numgfs);
void dump_coord_direction(const int grid, const REAL *restrict xx, const char* dir, const int Nxx);
void dump_bcstruct(const bc_struct *restrict bcstruct, const char* suffix);
#endif