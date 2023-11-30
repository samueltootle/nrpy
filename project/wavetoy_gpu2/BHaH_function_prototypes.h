void MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_free_memory_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void apply_bcs(const commondata_struct *restrict commondata, const params_struct *restrict params,REAL *restrict gfs);
void cmdline_input_and_parfile_parser(commondata_struct *restrict commondata, int argc, const char *argv[]);
void commondata_struct_set_to_default(commondata_struct *restrict commondata);
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void exact_solution_single_Cartesian_point(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const REAL xCart0, const REAL xCart1, const REAL xCart2,  REAL *restrict exact_soln_UUGF, REAL *restrict exact_soln_VVGF
);
void initial_data(const commondata_struct *restrict commondata, griddata_struct *restrict griddata);
int main(int argc, const char *argv[]);
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, bool calling_for_first_time);
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void progress_indicator(commondata_struct *restrict commondata, const griddata_struct *restrict griddata);
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *restrict in_gfs, REAL *restrict rhs_gfs);
