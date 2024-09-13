__host__ void CUDA__free_host_gfs(MoL_gridfunctions_struct *gridfuncs);
__host__ void CUDA__malloc_host_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                    MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_free_memory_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                            MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void apply_bcs_inner_only(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct,
                          REAL *restrict gfs);
void apply_bcs_outerextrap_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                     const bc_struct *restrict bcstruct, REAL *restrict gfs);
void apply_bcs_outerradiation_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3], const REAL custom_wavespeed[NUM_EVOL_GFS],
                                        const REAL custom_f_infinity[NUM_EVOL_GFS], REAL *restrict gfs, REAL *restrict rhs_gfs);
void apply_bcs_outerradiation_and_inner__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3],
                                                        const REAL custom_wavespeed[NUM_EVOL_GFS], const REAL custom_f_infinity[NUM_EVOL_GFS],
                                                        REAL *restrict gfs, REAL *restrict rhs_gfs);
void auxevol_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
__device__ void auxevol_gfs_single_point(const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict psi_background, REAL *restrict ADD_times_AUU);
void bcstruct_set_up(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                     bc_struct *restrict bcstruct_gpu);
void bcstruct_set_up__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     bc_struct *restrict bcstruct_gpu);
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]);
void cfl_limited_timestep__rfm__SinhSymTP(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]);
void check_stop_conditions(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void cmdline_input_and_parfile_parser(commondata_struct *restrict commondata, int argc, const char *argv[]);
void commondata_struct_set_to_default(commondata_struct *restrict commondata);
REAL compute_L2_norm_of_gridfunction(commondata_struct *restrict commondata, griddata_struct *restrict griddata, const REAL integration_radius,
                                     const int gf_index, const REAL *restrict in_gf);
void compute_residual_all_points(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                 const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);
__host__ size_t cpyDevicetoHost__gf(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *gf_host,
                                    const REAL *gf_gpu, const int host_GF_IDX, const int gpu_GF_IDX);
__host__ void cpyDevicetoHost__grid(const commondata_struct *restrict commondata, griddata_struct *restrict gd_host,
                                    const griddata_struct *restrict gd_gpu);
__host__ size_t cpyHosttoDevice__gf(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *gf_host,
                                    REAL *gf_gpu, const int host_GF_IDX, const int gpu_GF_IDX);
__host__ void cpyHosttoDevice_commondata__constant(const commondata_struct *restrict commondata);
__host__ void cpyHosttoDevice_params__constant(const params_struct *restrict params);
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host);
void diagnostics_nearest_1d_y_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_y_axis__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL *restrict xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL *restrict xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
REAL find_global__minimum(REAL *data, uint const data_length);
REAL find_global__sum(REAL *data, uint const data_length);
void griddata_free(const commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host,
                   const bool enable_free_non_y_n_gfs);
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
__device__ __host__ void initial_guess_single_point(const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID);
void initialize_constant_auxevol(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
int main(int argc, const char *argv[]);
void numerical_grid_params_Nxx_dxx_xx(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3], const int Nx[3],
                                      const bool grid_is_resized);
void numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *xx[3],
                                                      const int Nx[3], const bool grid_is_resized);
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host,
                                  bool calling_for_first_time);
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void progress_indicator(commondata_struct *restrict commondata, const griddata_struct *restrict griddata);
int read_checkpoint(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_GPU);
void rfm_precompute_defines(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                            REAL *restrict xx[3]);
void rfm_precompute_defines__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                            rfm_struct *restrict rfmstruct, REAL *restrict xx[3]);
void rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_free__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                         rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct);
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
              const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs);
void variable_wavespeed_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void write_checkpoint(const commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_GPU);
void xx_to_Cart(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const int i0, const int i1,
                const int i2, REAL xCart[3]);
__host__ void xx_to_Cart__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                         const int i0, const int i1, const int i2, REAL xCart[3]);
