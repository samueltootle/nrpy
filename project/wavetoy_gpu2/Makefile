CC=nvcc  # assigns the value CC to gcc only if environment variable CC is not already set
# CFLAGS = -O2 -march=native -g -Wall -Wno-unused-variable -std=gnu99
CFLAGS = -Xcompiler -g -Xptxas=-O3 -arch=native  -O3 -Xcompiler=-march=native -Xcompiler -Wall --forward-unknown-to-host-compiler --Werror cross-execution-space-call --relocatable-device-code=true -allow-unsupported-compiler
#CFLAGS = -O3 -funroll-loops -march=native -g -Wall -Wno-unused-variable -std=gnu99 -std=gnu99
#CFLAGS = -O2 -g -Wall -Wno-unused-variable -Wno-unknown-pragmas -std=gnu99

INCLUDEDIRS =
LDFLAGS = -lm

# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

#ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += -Xcompiler $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)  # -lgomp does not work with clang in termux
#endif

#OBJ_FILES = MoL_free_memory_non_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_malloc_y_n_gfs.o MoL_step_forward_in_time.o apply_bcs.o cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o diagnostics.o exact_solution_single_Cartesian_point.o initial_data.o params_struct_set_to_default.o progress_indicator.o rhs_eval.o
OBJ_FILES= cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o progress_indicator.o
CU_OBJ_FILES = apply_bcs.o MoL_RK_substeps_in_time.o rhs_eval.o MoL_step_forward_in_time.o diagnostics.o exact_solution_single_Cartesian_point.o initial_data.o MoL_free_memory_non_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_malloc_y_n_gfs.o numerical_grids_and_timestep.o params_struct_set_to_default.o set_fd_constants.o main.o

all: wavetoy_gpu

wavetoy_gpu: $(OBJ_FILES) $(CU_OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS)

$(CU_OBJ_FILES): %.o: %.cu $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

$(OBJ_FILES) : %.o: %.cpp $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o *~ */*~ ./#* *.txt *.dat *.avi *.png wavetoy_gpu
