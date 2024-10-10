"""
Construct BHaH_defines.h and gpu_defines from data registered to griddata_commondata, CodeParameters, and NRPyParameters.

Author: Samuel D. Tootle
        sdtootle **at** gmail **dot** com
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Union, Any


import nrpy.grid as gri
from nrpy.helpers.generic import clang_format
from nrpy.infrastructures.gpu.header_definitions.base_output_BHaH_defines_h import (
    base_output_BHaH_defines_h,
)


def generate_declaration_str(
    decl_dict: Dict[str, Dict[str, str]], prefix: str = ""
) -> str:
    """
    Generate block string of header definitions.

    :param decl_dict: Dictionary of definitions and their properties
    :param prefix: optional prefix
    :returns: str
    """
    if prefix != "":
        prefix += " "
    decl_str: str = ""
    for var, sub_dict in decl_dict.items():
        # Standard declarations str
        decl_str += f"{sub_dict['comment']}"
        decl_str += f"{prefix} {sub_dict['type']} {var}{sub_dict['suffix']};\n"
    return decl_str


class output_BHaH_gpu_defines_h:
    r"""
    Generate and write to file the BHaH_gpu_defines.h file.

    :param project_dir: Location to write file to
    :param additional_declarations_dict: Dictionary storing additional declaration dictionaries
    :param additional_macros_str: Block string of additional macro definitions
    :param clang_format_options: Options for clang formatting.
    :param num_streams: Number of CUDA streams to use
    :param nghosts: Number of ghost zones for the FD stencil

    >>> from nrpy.infrastructures.BHaH.MoLtimestepping.openmp import MoL
    >>> import nrpy.params as par
    >>> import nrpy.c_function as cfc
    >>> par.glb_extras_dict.clear()
    >>> cfc.CFunction_dict.clear()
    >>> _ = MoL.register_CFunctions(register_MoL_step_forward_in_time=False)
    >>> project_dir = Path("/tmp", "tmp_BHaH_defines_h")
    >>> d = output_BHaH_gpu_defines_h(project_dir)
    >>> d.generate_output_str()
    >>> print(d.file_output_str)
    // BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
    //    DO NOT EDIT THIS FILE BY HAND.
    <BLANKLINE>
    // Standard macro definitions
    // We include the macro definition nstreams since it is used for calculations in various
    // algorithms in addition to defining the streams array
    #define nstreams 3
    <BLANKLINE>
    // Standard declarations
    // Device storage for grid function parity
    extern __constant__ int8_t d_evol_gf_parity[24];
    extern cudaStream_t streams[nstreams];
    extern size_t GPU_N_SMS;
    <BLANKLINE>
    // CUDA Error checking macro only active if compiled with -DDEBUG
    // Otherwise additional synchronization overhead will occur
    #ifdef DEBUG
    #define cudaCheckErrors(v, msg)                                                                                                                      \
      do {                                                                                                                                               \
        cudaError_t __err = cudaGetLastError();                                                                                                          \
        if (__err != cudaSuccess) {                                                                                                                      \
          fprintf(stderr, "Fatal error: %s %s (%s at %s:%d)\n", #v, msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                 \
          fprintf(stderr, "*** FAILED - ABORTING\n");                                                                                                    \
          exit(1);                                                                                                                                       \
        }                                                                                                                                                \
      } while (0);
    #else
    #define cudaCheckErrors(v, msg)
    #endif
    <BLANKLINE>
    """

    def __init__(
        self,
        project_dir: str,
        additional_declarations_dict: Union[Dict[str, Any], None] = None,
        additional_macros_str: Union[str, None] = None,
        clang_format_options: str = "-style={BasedOnStyle: LLVM, ColumnLimit: 150}",
        num_streams: int = 3,
        nghosts: Union[int, None] = None,
        **_: Any,
    ) -> None:
        self.project_Path = Path(project_dir)
        self.num_streams = num_streams
        self.additional_decl_dict = additional_declarations_dict
        self.additional_macros_str = additional_macros_str
        self.clang_format_options = clang_format_options
        self.bhah_gpu_defines_filename = "BHaH_gpu_defines.h"
        self.NGHOSTS = nghosts

        # Standard macros str
        self.macro_str = f"""
// Standard macro definitions
// We include the macro definition nstreams since it is used for calculations in various
// algorithms in addition to defining the streams array
#define nstreams {self.num_streams}
"""
        standard_decl_dict = {
            "d_params": {
                "type": "__constant__ params_struct",
                "suffix": "",
                "comment": "// Device storage for grid parameters\n",
            },
            "d_commondata": {
                "type": "__constant__ commondata_struct",
                "suffix": "",
                "comment": "// Device storage for commondata\n",
            },
            "d_evol_gf_parity": {
                "type": "__constant__ int8_t",
                "suffix": "[24]",
                "comment": "// Device storage for grid function parity\n",
            },
            "streams": {
                "type": "cudaStream_t",
                "suffix": "[nstreams]",
                "comment": "",
            },
            "GPU_N_SMS": {
                "type": "size_t",
                "suffix": "",
                "comment": "",
            },
        }
        evolved_variables_list: list[str]
        (
            evolved_variables_list,
            _auxiliary_variables_list,
            _auxevol_variables_list,
        ) = gri.BHaHGridFunction.gridfunction_lists()
        # This device storage is only needed by some problems
        if evolved_variables_list:
            standard_decl_dict["d_gridfunctions_wavespeed"] = {
                "type": "__constant__ REAL",
                "suffix": "[NUM_EVOL_GFS]",
                "comment": "",
            }
            standard_decl_dict["d_gridfunctions_f_infinity"] = {
                "type": "__constant__ REAL",
                "suffix": "[NUM_EVOL_GFS]",
                "comment": "",
            }
        self.combined_decl_dict = standard_decl_dict
        self.decl_str: str = "// Standard declarations\n"
        self.decl_str += generate_declaration_str(
            self.combined_decl_dict, prefix="extern"
        )

        self.file_output_str = ""
        self.generate_output_str()
        self.write_to_file()

    def combine_declarations_dicts(self) -> None:
        """Add additional_decl_dict to combined_decl_dict."""
        if not self.additional_decl_dict is None:
            for k, v in self.additional_decl_dict.items():
                self.combined_decl_dict[k] = v

    def generate_output_str(self) -> None:
        """Generate block output str to prepare writing to file."""
        self.file_output_str = """// BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
//    DO NOT EDIT THIS FILE BY HAND.\n\n"""

        self.file_output_str += self.macro_str
        if self.additional_macros_str:
            self.file_output_str += (
                "\n\n// Additional Macros\n" + self.additional_macros_str
            )

        self.file_output_str += "\n\n" + self.decl_str

        if self.additional_decl_dict:
            self.file_output_str += "\n\n// Additional Declarations\n"
            self.file_output_str += generate_declaration_str(
                self.additional_decl_dict, prefix="extern"
            )
            self.combine_declarations_dicts()

        self.file_output_str += (
            "\n\n"
            + r"""
// CUDA Error checking macro only active if compiled with -DDEBUG
// Otherwise additional synchronization overhead will occur
#ifdef DEBUG
#define cudaCheckErrors(v, msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s %s (%s at %s:%d)\n", \
                #v, msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0);
#else
#define cudaCheckErrors(v, msg)
#endif
"""
        )
        self.file_output_str = clang_format(
            self.file_output_str, clang_format_options=self.clang_format_options
        )
        self.write_to_file()

    def write_to_file(self) -> None:
        """Write file_output_str to header file."""
        bhah_gpu_defines_file = self.project_Path / self.bhah_gpu_defines_filename
        with bhah_gpu_defines_file.open("w", encoding="utf-8") as file:
            file.write(self.file_output_str)


class output_BHaH_gpu_global_init_h:
    r"""
    Generate and write to file the BHaH_gpu_global_init.h file.

    :param project_dir: Location to write file to
    :param declarations_dict: Dictionary storing declaration dictionaries
    :param clang_format_options: Options for clang formatting.

    >>> from nrpy.infrastructures.BHaH.MoLtimestepping.openmp import MoL
    >>> import nrpy.params as par
    >>> import nrpy.c_function as cfc
    >>> par.glb_extras_dict.clear()
    >>> cfc.CFunction_dict.clear()
    >>> _ = MoL.register_CFunctions(register_MoL_step_forward_in_time=False)
    >>> project_dir = Path("/tmp", "tmp_BHaH_defines_h")
    >>> gpu_d = output_BHaH_gpu_defines_h(project_dir)
    >>> gpu_init = output_init_gpu_defines_h(project_dir,gpu_d.combined_decl_dict)
    >>> print(gpu_init.file_output_str)
    // BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
    //    DO NOT EDIT THIS FILE BY HAND.
    <BLANKLINE>
    // Device storage for grid function parity
    __constant__ int8_t d_evol_gf_parity[24];
    cudaStream_t streams[nstreams];
    size_t GPU_N_SMS;
    <BLANKLINE>
    // Initialize streams
    for (int i = 0; i < nstreams; ++i) {
      cudaStreamCreate(&streams[i]);
    }
    // Copy parity array to device __constant__ memory
    cudaMemcpyToSymbol(d_evol_gf_parity, evol_gf_parity, 24 * sizeof(int8_t));
    <BLANKLINE>

    """

    def __init__(
        self,
        project_dir: str,
        declarations_dict: Dict[str, Dict[str, str]],
        clang_format_options: str = "-style={BasedOnStyle: LLVM, ColumnLimit: 150}",
        **_: Any,
    ) -> None:
        self.project_Path = Path(project_dir)
        self.declarations_dict = declarations_dict
        self.clang_format_options = clang_format_options
        self.filename = "BHaH_gpu_global_init.h"

        self.file_output_str = """// BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
//    DO NOT EDIT THIS FILE BY HAND.\n\n"""

        self.file_output_str += """\n\n
// Initialize streams
for(int i = 0; i < nstreams; ++i) {
    cudaStreamCreate(&streams[i]);
}
// Copy parity array to device __constant__ memory
cudaMemcpyToSymbol(d_evol_gf_parity, evol_gf_parity, 24 * sizeof(int8_t));
cudaCheckErrors(copy, "Copy to d_evol_gf_parity failed");
"""
        if "d_gridfunctions_wavespeed" in declarations_dict.keys():
            self.file_output_str += """
// Copy gridfunctions_wavespeed array to device memory
cudaMemcpyToSymbol(d_gridfunctions_wavespeed, gridfunctions_wavespeed, NUM_EVOL_GFS * sizeof(REAL));
cudaCheckErrors(copy, "Copy to d_gridfunctions_wavespeed failed");

// Copy gridfunctions_f_infinity array to device memory
cudaMemcpyToSymbol(d_gridfunctions_f_infinity, gridfunctions_f_infinity, NUM_EVOL_GFS * sizeof(REAL));
cudaCheckErrors(copy, "Copy to d_gridfunctions_f_infinity failed");
"""

        self.file_output_str = clang_format(
            self.file_output_str, clang_format_options=self.clang_format_options
        )
        self.write_to_file()

    def write_to_file(self) -> None:
        """Write file_output_str to header file."""
        output_file = self.project_Path / self.filename
        with output_file.open("w", encoding="utf-8") as file:
            file.write(self.file_output_str)


class output_BHaH_gpu_global_defines_h:
    r"""
    Generate and write to file the BHaH_gpu_global_defines.h file.

    :param project_dir: Location to write file to
    :param declarations_dict: Dictionary storing declaration dictionaries
    :param clang_format_options: Options for clang formatting.

    >>> from nrpy.infrastructures.BHaH.MoLtimestepping.openmp import MoL
    >>> import nrpy.params as par
    >>> import nrpy.c_function as cfc
    >>> par.glb_extras_dict.clear()
    >>> cfc.CFunction_dict.clear()
    >>> _ = MoL.register_CFunctions(register_MoL_step_forward_in_time=False)
    >>> project_dir = Path("/tmp", "tmp_BHaH_defines_h")
    >>> gpu_d = output_BHaH_gpu_defines_h(project_dir)
    >>> gpu_init = output_init_gpu_defines_h(project_dir,gpu_d.combined_decl_dict)
    >>> print(gpu_init.file_output_str)
    // BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
    //    DO NOT EDIT THIS FILE BY HAND.
    <BLANKLINE>
    // Device storage for grid function parity
    __constant__ int8_t d_evol_gf_parity[24];
    cudaStream_t streams[nstreams];
    size_t GPU_N_SMS;
    <BLANKLINE>
    // Initialize streams
    for (int i = 0; i < nstreams; ++i) {
      cudaStreamCreate(&streams[i]);
    }
    // Copy parity array to device __constant__ memory
    cudaMemcpyToSymbol(d_evol_gf_parity, evol_gf_parity, 24 * sizeof(int8_t));
    <BLANKLINE>

    """

    def __init__(
        self,
        project_dir: str,
        declarations_dict: Dict[str, Dict[str, str]],
        clang_format_options: str = "-style={BasedOnStyle: LLVM, ColumnLimit: 150}",
        **_: Any,
    ) -> None:
        self.project_Path = Path(project_dir)
        self.declarations_dict = declarations_dict
        self.clang_format_options = clang_format_options
        self.filename = "BHaH_gpu_global_defines.h"

        self.file_output_str = """// BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
//    DO NOT EDIT THIS FILE BY HAND.\n\n"""

        self.file_output_str += generate_declaration_str(self.declarations_dict)

        self.file_output_str = clang_format(
            self.file_output_str, clang_format_options=self.clang_format_options
        )
        self.write_to_file()

    def write_to_file(self) -> None:
        """Write file_output_str to header file."""
        output_file = self.project_Path / self.filename
        with output_file.open("w", encoding="utf-8") as file:
            file.write(self.file_output_str)


class output_BHaH_defines_h(base_output_BHaH_defines_h):
    r"""
    Output C code header file with macro definitions and other configurations for the project.

    :param project_dir: Directory where the project C code is output
    :param additional_includes: Additional header files to be included in the output
    :param REAL_means: The floating-point type to be used in the C code (default is "double")
    :param enable_rfm_precompute: A boolean value reflecting whether reference metric precomputation is enabled.
    :param fin_NGHOSTS_add_one_for_upwinding_or_KO: Option to add one extra ghost zone for upwinding
    :param supplemental_defines_dict: Additional key-value pairs to be included in the output file
    :param clang_format_options: Options for clang formatting.
    :param kwargs: Capture extra keyword arguments

    >>> from nrpy.infrastructures.BHaH.MoLtimestepping.openmp import MoL
    >>> import nrpy.finite_difference as fin
    >>> from nrpy.helpers.generic import compress_string_to_base64, decompress_base64_to_string, diff_strings
    >>> import nrpy.c_function as cfc
    >>> import nrpy.params as par
    >>> par.glb_extras_dict.clear()
    >>> cfc.CFunction_dict.clear()
    >>> _ = MoL.register_CFunctions(register_MoL_step_forward_in_time=False)
    >>> project_dir = Path("/tmp", "tmp_BHaH_defines_h")
    >>> project_dir.mkdir(parents=True, exist_ok=True)
    >>> C=output_BHaH_defines_h(project_dir=str(project_dir))

    # >>> expected_string = decompress_base64_to_string("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")
    # >>> returned_string = (project_dir / "BHaH_defines.h").read_text()
    # >>> if returned_string != expected_string:
    # ...    compressed_str = compress_string_to_base64(returned_string)
    # ...    error_message = "Trusted BHaH_defines.h string changed!\n"
    # ...    error_message += "Here's the diff:\n" + diff_strings(expected_string, returned_string) + "\n"
    # ...    raise ValueError(error_message + f"base64-encoded output: {compressed_str}")
    """

    def __init__(
        self,
        project_dir: str,
        additional_includes: Optional[List[str]] = None,
        REAL_means: str = "double",
        enable_rfm_precompute: bool = True,
        fin_NGHOSTS_add_one_for_upwinding_or_KO: bool = False,
        supplemental_defines_dict: Optional[Dict[str, str]] = None,
        clang_format_options: str = "-style={BasedOnStyle: LLVM, ColumnLimit: 150}",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            project_dir,
            additional_includes=additional_includes,
            REAL_means=REAL_means,
            enable_simd=False,
            enable_rfm_precompute=enable_rfm_precompute,
            fin_NGHOSTS_add_one_for_upwinding_or_KO=fin_NGHOSTS_add_one_for_upwinding_or_KO,
            supplemental_defines_dict=supplemental_defines_dict,
            clang_format_options=clang_format_options,
        )
        self.BHd_definitions_str += "\n#define restrict __restrict__\n"
        # Generate and write BHaH_gpu_defines.h to file
        gpu_defines = output_BHaH_gpu_defines_h(
            self.project_dir,
            clang_format_options=self.clang_format_options,
            nghosts=self.NGHOSTS,
            **kwargs,
        )

        self.gri_BHd_str += r"""
#define IDX3INTERIOR(i,j,k) \
  ( (i) + Nxx0 * ( (j) + Nxx1 * (k) ) )
"""
        self.gri_BHd_struct_str = self.gri_BHd_struct_str.replace("*restrict", "*")
        self.register_define_blocks()
        self.generate_output_str()

        # Add GPU header to the end of BHaH Defines header
        self.file_output_str += (
            f'\n#include "{gpu_defines.bhah_gpu_defines_filename}"\n'
        )

        self.write_to_file()

        _ = output_BHaH_gpu_global_defines_h(
            self.project_dir,
            gpu_defines.combined_decl_dict,
            clang_format_options=self.clang_format_options,
        )

        _ = output_BHaH_gpu_global_init_h(
            self.project_dir,
            gpu_defines.combined_decl_dict,
            clang_format_options=self.clang_format_options,
        )


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
