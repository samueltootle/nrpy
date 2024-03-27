CC = nvcc  # assigns the value CC to gcc only if environment variable CC is not already set
# CFLAGS = -O2 -march=native -g -Wall -Wno-unused-variable -std=gnu99
#CFLAGS = -O3 -funroll-loops -march=native -g -Wall -Wno-unused-variable -std=gnu99 -std=gnu99
#CFLAGS = -O2 -g -Wall -Wno-unused-variable -Wno-unknown-pragmas -std=gnu99
CFLAGS = -Xcompiler -g -O2 -arch=native -O2 -Xcompiler=-march=native -Xcompiler -Wall --forward-unknown-to-host-compiler --Werror cross-execution-space-call --relocatable-device-code=true -allow-unsupported-compiler

INCLUDEDIRS =
LDFLAGS = -lm

# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

# ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)  # -lgomp does not work with clang in termux
# endif

#OBJ_FILES = BrillLindquist.o Cart_to_xx_and_nearest_i0i1i2.o CoordSystem_hash_setup.o MoL_free_memory_non_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_malloc_y_n_gfs.o MoL_step_forward_in_time.o Ricci_eval.o Spherical/Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical.o Spherical/Ricci_eval__rfm__Spherical.o Spherical/apply_bcs_outerradiation_and_inner__rfm__Spherical.o Spherical/bcstruct_set_up__rfm__Spherical.o Spherical/cfl_limited_timestep__rfm__Spherical.o Spherical/constraints_eval__rfm__Spherical.o Spherical/diagnostics_nearest_1d_y_axis__rfm__Spherical.o Spherical/diagnostics_nearest_1d_z_axis__rfm__Spherical.o Spherical/diagnostics_nearest_2d_xy_plane__rfm__Spherical.o Spherical/diagnostics_nearest_2d_yz_plane__rfm__Spherical.o Spherical/diagnostics_nearest_grid_center__rfm__Spherical.o Spherical/enforce_detgammabar_equals_detgammahat__rfm__Spherical.o Spherical/initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical.o Spherical/numerical_grid_params_Nxx_dxx_xx__rfm__Spherical.o Spherical/rfm_precompute_defines__rfm__Spherical.o Spherical/rfm_precompute_free__rfm__Spherical.o Spherical/rfm_precompute_malloc__rfm__Spherical.o Spherical/rhs_eval__rfm__Spherical.o Spherical/xx_to_Cart__rfm__Spherical.o apply_bcs_inner_only.o apply_bcs_outerextrap_and_inner.o apply_bcs_outerradiation_and_inner.o bcstruct_set_up.o cfl_limited_timestep.o cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o constraints_eval.o diagnostics.o diagnostics_nearest_1d_y_axis.o diagnostics_nearest_1d_z_axis.o diagnostics_nearest_2d_xy_plane.o diagnostics_nearest_2d_yz_plane.o diagnostics_nearest_grid_center.o enforce_detgammabar_equals_detgammahat.o initial_data.o initial_data_reader__convert_ADM_Cartesian_to_BSSN.o main.o numerical_grid_params_Nxx_dxx_xx.o numerical_grids_and_timestep.o params_struct_set_to_default.o progress_indicator.o rfm_precompute_defines.o rfm_precompute_free.o rfm_precompute_malloc.o rhs_eval.o xx_to_Cart.o
OBJ_FILES  = apply_bcs_outerradiation_and_inner.o rhs_eval.o
OBJ_FILES += diagnostics_nearest_grid_center.o constraints_eval.o Ricci_eval.o 
OBJ_FILES += initial_data_reader__convert_ADM_Cartesian_to_BSSN.o cfl_limited_timestep.o 
OBJ_FILES += bcstruct_set_up.o rfm_precompute_free.o rfm_precompute_malloc.o rfm_precompute_defines.o 
OBJ_FILES += numerical_grid_params_Nxx_dxx_xx.o params_struct_set_to_default.o CoordSystem_hash_setup.o 
OBJ_FILES += cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o enforce_detgammabar_equals_detgammahat.o
OBJ_FILES += diagnostics_nearest_1d_y_axis.o diagnostics_nearest_1d_z_axis.o progress_indicator.o
OBJ_FILES += diagnostics_nearest_2d_xy_plane.o diagnostics_nearest_2d_yz_plane.o
CU_OBJ_FILES  = Spherical/rhs_eval__rfm__Spherical.o Spherical/diagnostics_nearest_grid_center__rfm__Spherical.o 
CU_OBJ_FILES += Spherical/constraints_eval__rfm__Spherical.o Spherical/Ricci_eval__rfm__Spherical.o diagnostics.o 
CU_OBJ_FILES += MoL_free_memory_non_y_n_gfs.o MoL_malloc_non_y_n_gfs.o apply_bcs_inner_only.o 
CU_OBJ_FILES += Spherical/Cart_to_xx_and_nearest_i0i1i2__rfm__Spherical.o Cart_to_xx_and_nearest_i0i1i2.o 
CU_OBJ_FILES += initial_data.o Spherical/xx_to_Cart__rfm__Spherical.o xx_to_Cart.o BrillLindquist.o 
CU_OBJ_FILES += Spherical/initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical.o MoL_malloc_y_n_gfs.o 
CU_OBJ_FILES += MoL_free_memory_y_n_gfs.o reduction_utils.o Spherical/cfl_limited_timestep__rfm__Spherical.o 
CU_OBJ_FILES += Spherical/bcstruct_set_up__rfm__Spherical.o Spherical/rfm_precompute_defines__rfm__Spherical.o 
CU_OBJ_FILES += Spherical/rfm_precompute_free__rfm__Spherical.o Spherical/rfm_precompute_malloc__rfm__Spherical.o 
CU_OBJ_FILES += numerical_grids_and_timestep.o Spherical/numerical_grid_params_Nxx_dxx_xx__rfm__Spherical.o 
CU_OBJ_FILES += set__constant__.o main.o MoL_step_forward_in_time.o 
CU_OBJ_FILES += Spherical/apply_bcs_outerradiation_and_inner__rfm__Spherical.o MoL_RK_substeps_in_time.o
CU_OBJ_FILES += apply_bcs_outerextrap_and_inner.o Spherical/enforce_detgammabar_equals_detgammahat__rfm__Spherical.o
CU_OBJ_FILES += Spherical/diagnostics_nearest_1d_y_axis__rfm__Spherical.o Spherical/diagnostics_nearest_1d_z_axis__rfm__Spherical.o
CU_OBJ_FILES += Spherical/rhs_eval__rfm__Spherical_reimagine.o Spherical/diagnostics_nearest_2d_xy_plane__rfm__Spherical.o Spherical/diagnostics_nearest_2d_yz_plane__rfm__Spherical.o
CU_OBJ_FILES += TESTS/commondata_test.o TESTS/paramstruct_test.o TESTS/coord_direction_test.o cpyDevicetoHost_griddata.o

all: two_blackholes_collide_nosimd_rfm

two_blackholes_collide_nosimd_rfm: $(OBJ_FILES)  $(CU_OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES)  $(CU_OBJ_FILES) -o $@ $(LDFLAGS)

$(CU_OBJ_FILES): %.o: %.cu $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

$(OBJ_FILES) : %.o: %.cpp $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o *~ */*~ ./TESTS/reldiff/* *.txt two_blackholes_collide_nosimd_rfm #* *.txt *.dat *.avi *.png two_blackholes_collide_nosimd_rfm
