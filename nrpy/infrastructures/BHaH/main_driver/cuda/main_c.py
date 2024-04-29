"""
Generate the C main() function for all codes in the BHaH infrastructure using CUDA.

Author: Samuel D. Tootle
        sdtootle **at** gmail **dot** com
        Zachariah B. Etienne
        zachetie **at** gmail **dot** com
"""

import nrpy.c_function as cfc
import nrpy.infrastructures.BHaH.main_driver.base_main_c as base_main


class register_CFunction_main_c(base_main.base_register_CFunction_main_c):
    """
    Generate the "generic" C main() function for all simulation codes in the BHaH infrastructure.

    :param MoL_method: Method of Lines algorithm used to step forward in time.
    :param initial_data_desc: Description for initial data, default is an empty string.
    :param boundary_conditions_desc: Description of the boundary conditions, default is an empty string.
    :param prefunc: String that appears before main(). DO NOT populate this, except when debugging, default is an empty string.
    :param initialize_constant_auxevol: If set to True, `initialize_constant_auxevol` function will be called during the simulation initialization phase to set these constants. Default is False.
    :param pre_MoL_step_forward_in_time: Code for handling pre-right-hand-side operations, default is an empty string.
    :param post_MoL_step_forward_in_time: Code for handling post-right-hand-side operations, default is an empty string.
    :param clang_format_options: Clang formatting options, default is "-style={BasedOnStyle: LLVM, ColumnLimit: 150}".
    :raises ValueError: Raised if any required function for BHaH main() is not registered.
    """

    def __init__(
        self,
        MoL_method: str,
        initial_data_desc: str = "",
        boundary_conditions_desc: str = "",
        prefunc: str = "",
        initialize_constant_auxevol: bool = False,
        pre_MoL_step_forward_in_time: str = "",
        post_MoL_step_forward_in_time: str = "",
        clang_format_options: str = "-style={BasedOnStyle: LLVM, ColumnLimit: 150}",
    ) -> None:
        super().__init__(
            MoL_method,
            initial_data_desc=initial_data_desc,
            boundary_conditions_desc=boundary_conditions_desc,
            prefunc=prefunc,
            initialize_constant_auxevol=initialize_constant_auxevol,
            pre_MoL_step_forward_in_time=pre_MoL_step_forward_in_time,
            post_MoL_step_forward_in_time=post_MoL_step_forward_in_time,
            clang_format_options=clang_format_options,
        )
        self.includes += ["BHaH_gpu_global_defines.h"]
        self.body = r"""
#include "BHaH_gpu_global_init.h"
commondata_struct commondata; // commondata contains parameters common to all grids.
griddata_struct *restrict griddata; // griddata contains data specific to an individual grid.
griddata_struct *restrict griddata_host; // stores only the host data needed for diagnostics

// Step 1.a: Set each commondata CodeParameter to default.
commondata_struct_set_to_default(&commondata);

// Step 1.b: Overwrite default values to parfile values. Then overwrite parfile values with values set at cmd line.
cmdline_input_and_parfile_parser(&commondata, argc, argv);

// Step 1.c: Allocate NUMGRIDS griddata arrays, each containing data specific to an individual grid.
griddata = (griddata_struct *)malloc(sizeof(griddata_struct) * commondata.NUMGRIDS);
griddata_host = (griddata_struct *)malloc(sizeof(griddata_struct) * commondata.NUMGRIDS);

// Step 1.d: Set up numerical grids: xx[3], masks, Nxx, dxx, invdxx, bcstruct, rfm_precompute, timestep, etc.
{
  // if calling_for_first_time, then initialize commondata time=nn=t_0=nn_0 = 0
  const bool calling_for_first_time = true;
  numerical_grids_and_timestep(&commondata, griddata, griddata_host, calling_for_first_time);
}

for(int grid=0; grid<commondata.NUMGRIDS; grid++) {
  // Step 2: Initial data are set on y_n_gfs gridfunctions. Allocate storage for them first.
  MoL_malloc_y_n_gfs(&commondata, &griddata[grid].params, &griddata[grid].gridfuncs);
  //cpyDevicetoHost__malloc_y_n_gfs(&commondata, &griddata[grid].params, &griddata_host[grid].gridfuncs);
}

// Step 3: Finalize initialization: set up initial data, etc.
initial_data(&commondata, griddata);

// Step 4: Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping
for(int grid=0; grid<commondata.NUMGRIDS; grid++) {
  MoL_malloc_non_y_n_gfs(&commondata, &griddata[grid].params, &griddata[grid].gridfuncs);
  cpyDevicetoHost__malloc_host_diag_gfs(&commondata, &griddata[grid].params, &griddata_host[grid].gridfuncs);
}
"""
        if self.initialize_constant_auxevol:
            self.body += """// Step 4.a: Set AUXEVOL gridfunctions that will never change in time.
initialize_constant_auxevol(&commondata, griddata);
"""
        self.body += """
// Step 5: MAIN SIMULATION LOOP
while(commondata.time < commondata.t_final) { // Main loop to progress forward in time.
  // Step 5.a: Main loop, part 1: Output diagnostics
  diagnostics(&commondata, griddata, griddata_host);

  // Step 5.b: Main loop, part 2 (pre_MoL_step_forward_in_time): Prepare to step forward in time
"""
        if self.pre_MoL_step_forward_in_time != "":
            self.body += self.pre_MoL_step_forward_in_time
        else:
            self.body += "  // (nothing here; specify by setting pre_MoL_step_forward_in_time string in register_CFunction_main_c().)\n"
        self.body += f"""
  // Step 5.c: Main loop, part 3: Step forward in time using Method of Lines with {MoL_method} algorithm,
  //           applying {self.boundary_conditions_desc} boundary conditions.
  MoL_step_forward_in_time(&commondata, griddata);

  // Step 5.d: Main loop, part 4 (post_MoL_step_forward_in_time): Finish up step in time
"""
        if self.post_MoL_step_forward_in_time != "":
            self.body += self.post_MoL_step_forward_in_time
        else:
            self.body += "  // (nothing here; specify by setting post_MoL_step_forward_in_time string in register_CFunction_main_c().)\n"
        self.body += r"""
} // End main loop to progress forward in time.
// Make sure all workers are done
cudaDeviceSynchronize();
for(int i = 0; i < nstreams; ++i) {
    cudaStreamDestroy(streams[i]);
}
// Step 6: Free all allocated memory
{
  const bool enable_free_non_y_n_gfs=true;
  griddata_free(&commondata, griddata, griddata_host, enable_free_non_y_n_gfs);
}
return 0;
"""

        cfc.register_CFunction(
            includes=self.includes,
            prefunc=self.prefunc,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=self.params,
            body=self.body,
            clang_format_options=clang_format_options,
        )
