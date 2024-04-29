"""
Module that provides the utilities for generating GPU Kernels

Authors: Samuel D. Tootle; sdtootle **at** gmail **dot** com
"""

from typing import Union, Dict, Any
import nrpy.c_function as cfc


class GPU_Kernel:
    """
    Class to Generate GPU Kernel code.

    :param body: Kernel body
    :param params_dict: Dictionary storing function arguments as keys and types as Dictionary entry
    :param c_function_name: Kernel function name
    :param cfunc_type: C Function return type
    :param decorators: Function decorators i.e. Kernel type, templates, etc
    :param fp_type: Floating point type, i.e. double, float, long double
    :param comments: Additional comments to add to Function description
    :param launch_dict: Dictionary that stores kernel launch settings

    >>> kernel = GPU_Kernel(
    ... "*x = in;",
    ... {'x' : 'REAL *restrict', 'in' : 'const REAL'},
    ... 'basic_assignment_gpu',
    ... launch_dict = {
    ... 'blocks_per_grid' : [32],
    ... 'threads_per_block' : [128,28,1],
    ... },
    ... )
    >>> print(kernel.c_function_call())
    basic_assignment_gpu<<<blocks_per_grid,threads_per_block>>>(x, in);
    <BLANKLINE>
    >>> print(kernel.CFunction.full_function)
    /*
     * GPU Kernel: basic_assignment_gpu.
     *
     */
    __global__ static void basic_assignment_gpu([ 'x', 'in' ]) { *x = in; }
    <BLANKLINE>
    >>> print(kernel.launch_block)
    <BLANKLINE>
    const size_t threads_in_x_dir = 128;
    const size_t threads_in_y_dir = 28;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    <BLANKLINE>
    dim3 blocks_per_grid(32,1,1);
    <BLANKLINE>
    >>> kernel = GPU_Kernel(
    ... "*x = in;",
    ... {'x' : 'REAL *restrict', 'in' : 'const REAL'},
    ... 'basic_assignment_gpu',
    ... launch_dict = {
    ... 'blocks_per_grid' : [],
    ... 'threads_per_block' : [128,28,1],
    ... },
    ... )
    >>> print(kernel.launch_block)
    <BLANKLINE>
    const size_t threads_in_x_dir = 128;
    const size_t threads_in_y_dir = 28;
    const size_t threads_in_z_dir = 1;
    dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);
    <BLANKLINE>
    dim3 grid_blocks(
        (Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
        (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
        (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir
    );
    <BLANKLINE>

    """

    def __init__(
        self,
        body: str,
        params_dict: Dict[str, Any],
        c_function_name: str,
        cfunc_type: str = "static void",
        decorators: str = "__global__",
        fp_type: str = "double",
        comments: str = "",
        launch_dict: Union[Dict[str, Any], None] = None,
    ) -> None:
        self.body = body
        self.params_dict = params_dict
        self.name = c_function_name
        self.cfunc_type = f"{decorators} {cfunc_type}"
        self.decorators = decorators
        self.fp_type = fp_type

        self.CFunction: cfc.CFunction
        self.desc: str = f"GPU Kernel: {self.name}.\n" + comments
        self.launch_dict = launch_dict
        self.launch_block: str = ""
        self.launch_settings: str = "("

        if self.decorators == "__global__" and launch_dict is None:
            raise ValueError(f"Error: {self.decorators} requires a launch_dict")
        self.generate_launch_block()

        self.param_list = [f"{v} {k}" for k, v in self.params_dict.items()]
        # Store CFunction
        self.CFunction = cfc.CFunction(
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=",".join(self.param_list),
            body=self.body,
        )

    def generate_launch_block(self):
        "Generate preceding launch block definitions for kernel function call."
        if not self.launch_dict is None:
            threads_per_block = self.launch_dict["threads_per_block"]
            for _ in range(3 - len(threads_per_block)):
                threads_per_block += ["1"]
            block_def_str = f"""
const size_t threads_in_x_dir = {threads_per_block[0]};
const size_t threads_in_y_dir = {threads_per_block[1]};
const size_t threads_in_z_dir = {threads_per_block[2]};
dim3 threads_per_block(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);"""

            blocks_per_grid = self.launch_dict["blocks_per_grid"]
            if len(blocks_per_grid) > 0:
                for _ in range(3 - len(blocks_per_grid)):
                    blocks_per_grid += [1]
                blocks_per_grid_str = ",".join(map(str, blocks_per_grid))
                grid_def_str = f"dim3 blocks_per_grid({blocks_per_grid_str});"
            else:
                grid_def_str = f"""dim3 blocks_per_grid(
    (Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
    (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
    (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir
);"""

            # Determine if the stream needs to be added to launch
            stream_def_str = None
            if "stream" in self.launch_dict:
                if (
                    self.launch_dict["stream"] == ""
                    or self.launch_dict["stream"] == "default"
                ):
                    stream_def_str = f"size_t streamid = params->grid_idx % nstreams;"
                else:
                    stream_def_str = f"size_t streamid = {self.launch_dict['stream']};"

            # Determine if the shared memory size needs to be added to launch
            # If a stream is specified, we need to at least set SM to 0
            sm_def_str = None
            if "sm" in self.launch_dict or not stream_def_str is None:
                if (
                    not "sm" in self.launch_dict
                    or self.launch_dict["sm"] == ""
                    or self.launch_dict["sm"] == "default"
                ):
                    sm_def_str = "size_t sm = 0;"
                    self.launch_dict["sm"] = 0
                else:
                    sm_def_str = f"size_t sm = {self.launch_dict['sm']};"

            self.launch_block = f"""{block_def_str}
{grid_def_str}
"""
            if not sm_def_str is None:
                self.launch_block += f"{sm_def_str}"
            if not stream_def_str is None:
                self.launch_block += f"{stream_def_str}"

            self.launch_settings = f"<<<blocks_per_grid,threads_per_block"
            if not sm_def_str is None:
                self.launch_settings += f",sm"
            if not stream_def_str is None:
                self.launch_settings += f",streams[streamid]"
            self.launch_settings += ">>>("

    def c_function_call(self) -> str:
        """
        Generate the C function call for a given Kernel.

        :return: The C function call as a string.
        """

        c_function_call: str = self.name + self.launch_settings
        for p in self.params_dict:
            c_function_call += f"{p}, "
        c_function_call = c_function_call[:-2] + ");\n"
        msg = f"{self.name} failure"
        msg = f'cudaCheckErrors(cudaKernel, "{msg}")'
        c_function_call += f"{msg};\n"

        return c_function_call


# Define functions to copy params to device
def register_CFunction_cpyHosttoDevice_params__constant() -> None:
    """
    Register C function for copying params to __constant__ space on device.

    :return: None.
    """

    includes = ["BHaH_defines.h"]

    desc = r"""Copy parameters to GPU __constant__."""
    cfunc_type = "__host__ void"
    name = "cpyHosttoDevice_params__constant"
    params = r"""const params_struct *restrict params"""
    body = "cudaMemcpyToSymbol(d_params, params, sizeof(params_struct));"
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        subdirectory="CUDA_utils",
    )

# Define functions to copy params to device
def register_CFunction_cpyHosttoDevice_commondata__constant() -> None:
    """
    Register C function for copying commondata to __constant__ space on device.

    :return: None.
    """

    includes = ["BHaH_defines.h"]

    desc = r"""Copy parameters to GPU __constant__."""
    cfunc_type = "__host__ void"
    name = "cpyHosttoDevice_commondata__constant"
    params = r"""const commondata_struct *restrict commondata"""
    body = "cudaMemcpyToSymbol(d_commondata, commondata, sizeof(commondata_struct));"
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        subdirectory="CUDA_utils",
    )


def generate_CFunction_mallocHostgrid() -> GPU_Kernel:

    desc = r"""Allocate griddata_struct[grid].xx for host."""
    name = "mallocHostgrid"
    params_dict = {
        "commondata": "const commondata_struct *restrict",
        "params": "const params_struct *restrict",
        "gd_host": "griddata_struct *restrict",
        "gd_gpu": "const griddata_struct *restrict",
    }
    body = """
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Set up cell-centered Cartesian coordinate grid, centered at the origin.
  gd_host->xx[0] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  gd_host->xx[1] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  gd_host->xx[2] = (REAL*) malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);
"""
    kernel = GPU_Kernel(body, params_dict, name, decorators="__host__", comments=desc)
    return kernel.CFunction.full_function


def register_CFunction_cpyDevicetoHost__grid() -> None:
    """
    Register C function for copying grid from device to host.

    :return: None.
    """
    includes = ["BHaH_defines.h"]
    prefunc = generate_CFunction_mallocHostgrid()
    desc = r"""Copy griddata_struct[grid].xx from GPU to host."""
    cfunc_type = "__host__ void"
    name = "cpyDevicetoHost__grid"
    params = "const commondata_struct *restrict commondata, griddata_struct *restrict gd_host, "
    params += "const griddata_struct *restrict gd_gpu"
    body = """for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
  const params_struct *restrict params = &gd_gpu[grid].params;
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  
  mallocHostgrid(commondata, params, &gd_host[grid], gd_gpu);
  cudaMemcpy(gd_host[grid].xx[0], gd_gpu[grid].xx[0], sizeof(REAL) * Nxx_plus_2NGHOSTS0, cudaMemcpyDeviceToHost);
  cudaMemcpy(gd_host[grid].xx[1], gd_gpu[grid].xx[1], sizeof(REAL) * Nxx_plus_2NGHOSTS1, cudaMemcpyDeviceToHost);
  cudaMemcpy(gd_host[grid].xx[2], gd_gpu[grid].xx[2], sizeof(REAL) * Nxx_plus_2NGHOSTS2, cudaMemcpyDeviceToHost);
}
"""
    cfc.register_CFunction(
        prefunc=prefunc,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        subdirectory="CUDA_utils",
    )

def register_CFunction_cpyDevicetoHost__malloc_host_gfs() -> None:
    """
    Register C function for allocating sufficient Host storage for diagnostics GFs.

    :return: None.
    """
    includes = ["BHaH_defines.h"]
    desc = r"""Allocate Host storage for diagnostics GFs."""
    cfunc_type = "__host__ void"
    name = "cpyDevicetoHost__malloc_host_diag_gfs"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, "
    params += "MoL_gridfunctions_struct *restrict gridfuncs"
    body = """
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  
  cudaMallocHost((void**)&gridfuncs->y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_EVOL_GFS);
  cudaCheckErrors(cudaMallocHost, "Malloc y_n diagnostic GFs failed.");
  
  cudaMallocHost((void**)&gridfuncs->diagnostic_output_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_AUX_GFS);
  cudaCheckErrors(cudaMallocHost, "Malloc diagnostic GFs failed.")
"""
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        subdirectory="CUDA_utils",
    )
    
def register_CFunction_cpyDevicetoHost__gf() -> None:
    """
    Register C function for asynchronously copying data from device to host.

    :return: None.
    """
    includes = ["BHaH_defines.h"]
    desc = r"""Asynchronously copying a grid function from device to host."""
    cfunc_type = "__host__ size_t"
    name = "cpyDevicetoHost__gf"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, "
    params += "REAL * gf_host, const REAL * gf_gpu, const int host_GF_IDX, const int gpu_GF_IDX"
    body = """
  int const Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  size_t streamid = (params->grid_idx + gpu_GF_IDX) % nstreams;
  int offset_gpu  = Nxx_plus_2NGHOSTS_tot * gpu_GF_IDX;
  int offset_host = Nxx_plus_2NGHOSTS_tot * host_GF_IDX;
  cudaMemcpyAsync(&gf_host[offset_host], 
                  &gf_gpu[offset_gpu], 
                  sizeof(REAL) * Nxx_plus_2NGHOSTS_tot, 
                  cudaMemcpyDeviceToHost, streams[streamid]);
  cudaCheckErrors(cudaMemcpyAsync, "Copy of gf data failed");
  return streamid;
"""
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        subdirectory="CUDA_utils",
    )

def register_CFunctions_HostDevice__operations() -> None:
    register_CFunction_cpyHosttoDevice_commondata__constant()
    register_CFunction_cpyHosttoDevice_params__constant()

    register_CFunction_cpyDevicetoHost__grid()
    register_CFunction_cpyDevicetoHost__gf()
    register_CFunction_cpyDevicetoHost__malloc_host_gfs()

class CUDA_reductions:
    def __init__ (
        self,
        reduction_type="minimum",
        cfunc_decorators="__host__",
        cfunc_type="REAL",
        fp_type = "double",
    )->None:
        self.reduction_type = reduction_type
        self.cfunc_decorators=cfunc_decorators
        self.cfunc_type=cfunc_type
        self.fp_type = fp_type
        
        self.type_dict = {
            "double" : 'unsigned long long',
            "float" : 'unsigned',
        }
        
        self.initial_value_dict = {
            'sum' : '0U',
            'maximum' : '0U',
            'minimum' : '0xFFFFFFFFU',            
        }
        
        # Define local reductions
        def minimum_reduction(cmp_var):
            return f"""
if(local_reduced > {cmp_var}) {{
    local_reduced = {cmp_var};
}}
"""
        def maximum_reduction(cmp_var):
            return f"""
if(local_reduced < {cmp_var}) {{
    local_reduced = {cmp_var};
}}
"""
        def sum_reduction(cmp_var):
            return f"local_reduced += {cmp_var};"
        # End local reduction definitions
        
        # Store local reductions for later use in substitution
        self.reduction_dict = {
            'sum' : sum_reduction,
            'minimum' : minimum_reduction,
            'maximum' : maximum_reduction,
        }
        
        self.atomic_operations = {
            'sum' : 'atomicAdd',
            'minimum' : 'atomicMin',
            'maximum' : 'atomicMax'
        }
        
        if self.reduction_type not in self.initial_value_dict:
            raise ValueError(f"{self.reduction_type} is not a defined reduction. Choose from {self.initial_value_dict.keys()}")
        self.includes = ["BHaH_defines.h"]
        self.desc = f"""Find array global {self.reduction_type}."""
        self.name = f"find_global__{self.reduction_type}"
        self.params = "REAL * data, uint const data_length"
        
        self.kernel_name = f"{self.name}__cuda"
        self.recast_type = f"({self.type_dict[self.fp_type]} int *)" if not self.reduction_type == "sum" else ""
        self.prefunc = f"""
__global__
static void {self.kernel_name}(REAL * data, REAL * min, uint const data_length) {{
    // shared data between all warps
    // Assumes one block = 32 warps = 32 * 32 threads
    // As of today, the standard maximum threads per
    // block is 1024 = 32 * 32
    __shared__ REAL shared_data[32];

    // Some initial value
    REAL REDUCTION_LIMIT = (REAL) {self.initial_value_dict[self.reduction_type]};

    // Global data index - expecting a 1D dataset
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;

    // thread index
    uint tid = threadIdx.x;

    // local thread reduction
    REAL local_reduced = REDUCTION_LIMIT;

    // warp mask - says all threads are involved in shuffle
    // 0xFFFFFFFFU in binary is 32 1's.
    unsigned mask = 0xFFFFFFFFU;

    // lane = which thread am I in the warp
    uint lane = threadIdx.x % warpSize;
    // warpID = which warp am I in the block
    uint warpID = threadIdx.x / warpSize;

    // Stride through data for each thread
    while(idx < data_length) {{
        {self.reduction_dict[self.reduction_type]("data[idx]")}
        // idx stride
        idx += gridDim.x * blockDim.x;
    }}
    // Shuffle down kernel
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {{
        REAL shfl = __shfl_down_sync(mask, local_reduced, offset);
        {self.reduction_dict[self.reduction_type]("shfl")}
    }}
    // Shuffle results in lane 0 have the shuffle result
    if(lane == 0) {{
        shared_data[warpID] = local_reduced;
    }}
    
    // Make sure all warps in the block are synchronized
    __syncthreads();
    
    // Since there is only 32 partial reductions, we only
    // have one warp worth of work
    if(warpID == 0) {{
        // Check to make sure we had 32 blocks of data
        if(tid < blockDim.x / warpSize) {{
            local_reduced = shared_data[lane];
        }} else {{
            local_reduced = REDUCTION_LIMIT;
        }}
        
        // Shuffle down kernel
        for(int offset = warpSize / 2; offset > 0; offset >>= 1) {{
            REAL shfl = __shfl_down_sync(mask, local_reduced, offset);
            {self.reduction_dict[self.reduction_type]("shfl")}
        }}
        if(tid == 0) {{
            {self.atomic_operations[self.reduction_type]}({self.recast_type}min, *({self.recast_type}&local_reduced));
        }}
    }}
}}
"""
        
        self.body = f"""
    // This can be tested up to 1024
    uint threadCount = 32;
    
    // Number of blocks
    uint blockCount = (data_length + threadCount - 1) / threadCount;
    
    // CUDA atomics other than cas are only
    // compatible with (u)int.  To be generic
    // we use {self.type_dict[self.fp_type]} to be able to handle
    // {self.fp_type} precision variables
    using ull = REAL;
    ull * h_reduced = (ull*)malloc(sizeof(ull));
    ull * d_reduced;
    *h_reduced = (ull){self.initial_value_dict[self.reduction_type]};
    
    cudaMalloc(&d_reduced, sizeof(ull));
    cudaCheckErrors(cudaMalloc, "cudaMalloc failure"); // error checking

    cudaMemcpy(d_reduced, h_reduced, sizeof(ull), cudaMemcpyHostToDevice);
    cudaCheckErrors(cudaMemcpy, "cudaCopyTo failure"); // error checking
    
    {self.name}__cuda<<<blockCount, threadCount>>>(data, d_reduced, data_length);
    cudaCheckErrors(find_min_cu, "cudaKernel - find_min_cu failed"); // error checking
    
    cudaMemcpy(h_reduced, d_reduced, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaCheckErrors(cudaMemcpy, "cudaCopyFrom failure"); // error checking

    cudaFree(d_reduced);
    cudaCheckErrors(cudaFree, "cudaFree failure"); // error checking

    // Recast back to result pointer type
    REAL * res = (REAL *) h_reduced;
    return *res;
"""
        self.CFunction = None
        
    def generate_CFunction(self) -> None:
        "Generate CFunction from initialized parameters."
        
        self.CFunction = cfc.CFunction(
            prefunc=self.prefunc,
            includes=self.includes,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            CoordSystem_for_wrapper_func="",
            name=self.name,
            params=self.params,
            include_CodeParameters_h=False,
            body=self.body,
            subdirectory="CUDA_utils",
        )

def register_CFunction_find_global_minimum(fp_type = "double") -> None:
    """
    Register C function for finding the global minimum of an array

    :return: None.
    """
    reduction = CUDA_reductions(
        reduction_type="minimum",
        cfunc_decorators="__host__",
        cfunc_type="REAL",
        fp_type = fp_type,
    )

    cfc.register_CFunction(
        prefunc=reduction.prefunc,
        includes=reduction.includes,
        desc=reduction.desc,
        cfunc_type=reduction.cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=reduction.name,
        params=reduction.params,
        include_CodeParameters_h=False,
        body=reduction.body,
        subdirectory="CUDA_utils",
    )
    
def register_CFunction_find_global_sum(fp_type = "double") -> None:
    """
    Register C function for finding the global sum of an array

    :return: None.
    """
    reduction = CUDA_reductions(
        reduction_type="sum",
        cfunc_decorators="__host__",
        cfunc_type="REAL",
        fp_type = fp_type,
    )

    cfc.register_CFunction(
        prefunc=reduction.prefunc,
        includes=reduction.includes,
        desc=reduction.desc,
        cfunc_type=reduction.cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=reduction.name,
        params=reduction.params,
        include_CodeParameters_h=False,
        body=reduction.body,
        subdirectory="CUDA_utils",
    )

if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")