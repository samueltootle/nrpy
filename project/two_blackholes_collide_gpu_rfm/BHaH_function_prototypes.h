void Cart_to_xx_and_nearest_i0i1i2(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void CoordSystem_hash_setup(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_free_memory_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void Ricci_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, REAL *restrict auxevol_gfs);
void Ricci_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, REAL *restrict auxevol_gfs);
void apply_bcs_inner_only(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct, REAL *restrict gfs);
void apply_bcs_outerextrap_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct, REAL *restrict gfs);
void apply_bcs_outerradiation_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const bc_struct *restrict bcstruct, REAL *xx[3],
    const REAL custom_wavespeed[NUM_EVOL_GFS],
    const REAL custom_f_infinity[NUM_EVOL_GFS],
    REAL *restrict gfs, REAL *restrict rhs_gfs);
void apply_bcs_outerradiation_and_inner__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
    const bc_struct *restrict bcstruct, REAL *xx[3],
    const REAL custom_wavespeed[NUM_EVOL_GFS],
    const REAL custom_f_infinity[NUM_EVOL_GFS],
    REAL *restrict gfs, REAL *restrict rhs_gfs);
void bcstruct_set_up(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], bc_struct *restrict bcstruct);
void bcstruct_set_up__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], bc_struct *restrict bcstruct);
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3], bc_struct *restrict bcstruct);
void cfl_limited_timestep__rfm__Spherical(commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3], bc_struct *restrict bcstruct);
void cmdline_input_and_parfile_parser(commondata_struct *restrict commondata, int argc, const char *argv[]);
void commondata_struct_set_to_default(commondata_struct *restrict commondata);
void constraints_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, const REAL *restrict auxevol_gfs, REAL *restrict diagnostic_output_gfs);
void constraints_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, const REAL *restrict auxevol_gfs, REAL *restrict diagnostic_output_gfs);
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void diagnostics_nearest_1d_y_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_y_axis__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_grid_center(commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_grid_center__rfm__Spherical(commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs);
void enforce_detgammabar_equals_detgammahat(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, REAL *restrict in_gfs);
void enforce_detgammabar_equals_detgammahat__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, REAL *restrict in_gfs);
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void initial_data_reader__convert_ADM_Cartesian_to_BSSN(const commondata_struct *restrict commondata, const params_struct *restrict params,
    REAL *xx[3], bc_struct *restrict bcstruct, MoL_gridfunctions_struct *restrict gridfuncs,
    ID_persist_struct *restrict ID_persist,
    void ID_function(const commondata_struct *restrict commondata, const REAL xCart[3],
                     const ID_persist_struct *restrict ID_persist,
                     initial_data_struct *restrict initial_data));
void initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params,
    REAL *xx[3], bc_struct *restrict bcstruct, MoL_gridfunctions_struct *restrict gridfuncs,
    ID_persist_struct *restrict ID_persist,
    void ID_function(const commondata_struct *restrict commondata, const REAL xCart[3],
                     const ID_persist_struct *restrict ID_persist,
                     initial_data_struct *restrict initial_data));
int main(int argc, const char *argv[]);
void numerical_grid_params_Nxx_dxx_xx(commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3]);
void numerical_grid_params_Nxx_dxx_xx__rfm__Spherical(commondata_struct *restrict commondata, params_struct *restrict params, REAL * xx[3]);
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, bool calling_for_first_time);
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void progress_indicator(commondata_struct *restrict commondata, const griddata_struct *restrict griddata);
void rfm_precompute_defines(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct, REAL *xx[3]);
void rfm_precompute_defines__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct, REAL *xx[3]);
void rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_free__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs);
void rhs_eval__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs);
