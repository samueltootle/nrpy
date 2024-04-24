CC = nvcc  # Locally overwrites CC to nvcc\n
CFLAGS = -Xcompiler -fopenmp -Xcompiler -g -O2 -arch=native -O2 -Xcompiler=-march=native -Xcompiler -Wall --forward-unknown-to-host-compiler --Werror cross-execution-space-call --relocatable-device-code=true -allow-unsupported-compiler
#CFLAGS = -O2 -march=native -g -Wall -Wno-unused-variable
#CFLAGS = -O3 -funroll-loops -march=native -g -Wall -Wno-unused-variable -std=gnu99
#CFLAGS = -O2 -g -Wall -Wno-unused-variable -Wno-unknown-pragmas

INCLUDEDIRS =
LDFLAGS = -lm

OPENMP_FLAG = -fopenmp
CFLAGS += $(OPENMP_FLAG)
LDFLAGS += $(OPENMP_FLAG)

OBJ_FILES = CUDA_utils/cpyDevicetoHost__grid.o CUDA_utils/cpyHosttoDevice_commondata__constant.o CUDA_utils/cpyHosttoDevice_params__constant.o CUDA_utils/find_global__min.o MoL_free_memory_non_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_malloc_y_n_gfs.o MoL_step_forward_in_time.o SinhSymTP/apply_bcs_outerradiation_and_inner__rfm__SinhSymTP.o SinhSymTP/bcstruct_set_up__rfm__SinhSymTP.o SinhSymTP/cfl_limited_timestep__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_1d_y_axis__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_1d_z_axis__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_2d_xy_plane__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP.o SinhSymTP/numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_defines__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_free__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_malloc__rfm__SinhSymTP.o SinhSymTP/xx_to_Cart__rfm__SinhSymTP.o apply_bcs_inner_only.o apply_bcs_outerextrap_and_inner.o apply_bcs_outerradiation_and_inner.o auxevol_gfs_all_points.o auxevol_gfs_single_point.o bcstruct_set_up.o cfl_limited_timestep.o check_stop_conditions.o cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o diagnostics.o diagnostics_nearest_1d_y_axis.o diagnostics_nearest_1d_z_axis.o diagnostics_nearest_2d_xy_plane.o diagnostics_nearest_2d_yz_plane.o griddata_free.o initial_data.o initial_guess_single_point.o initialize_constant_auxevol.o main.o numerical_grid_params_Nxx_dxx_xx.o numerical_grids_and_timestep.o params_struct_set_to_default.o progress_indicator.o read_checkpoint.o rfm_precompute_defines.o rfm_precompute_free.o rfm_precompute_malloc.o rhs_eval.o variable_wavespeed_gfs_all_points.o write_checkpoint.o xx_to_Cart.o

all: nrpyelliptic_conformally_flat_gpu

%.o: %.cu $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

nrpyelliptic_conformally_flat_gpu: $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o *~ */*~ ./#* *.txt *.dat *.avi *.png nrpyelliptic_conformally_flat_gpu
