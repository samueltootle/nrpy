"""
Module for producing C codes related to MoL timestepping within the BHaH infrastructure.
This includes implementation details and functions for allocating and deallocating the necessary memory.

Authors: Brandon Clark
         Zachariah B. Etienne (maintainer)
         zachetie **at** gmail **dot* com
"""

from typing import List, Union, Dict, Tuple
import warnings
import sympy as sp  # Import SymPy, a computer algebra system written entirely in Python
import nrpy.c_function as cfc
from nrpy.c_codegen import c_codegen
from nrpy.infrastructures.BHaH.MoLtimestepping.RK_Butcher_Table_Dictionary import (
    generate_Butcher_tables,
)
from nrpy.helpers.generic import (
    superfast_uniq,
)


def is_diagonal_Butcher(
    Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
    key: str,
) -> bool:
    """
    Check if a given Butcher table (from Butcher_dict) is diagonal.

    :param Butcher_dict: A dictionary containing Butcher tables. Each key maps to a table (list of lists).
    :param key: The key to retrieve the Butcher table from the Butcher_dict.

    :return: True if the table is diagonal, False otherwise.

    Note:
    A Butcher table is diagonal if all its non-diagonal elements are zero.
    """
    # Get the Butcher table corresponding to the provided key
    Butcher = Butcher_dict[key][0]

    # Establish the number of rows to check for diagonal trait, all but the last row
    L = len(Butcher) - 1

    # Check all the desired rows
    for i in range(L):
        # Check each element before the diagonal element in a row
        for j in range(1, i):
            # If any non-diagonal coefficient is non-zero, then the table is not diagonal
            if Butcher[i][j] != sp.sympify(0):
                return False

    # If all non-diagonal elements are zero, the table is diagonal
    return True


def generate_gridfunction_names(
    Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
    MoL_method: str = "RK4",
) -> Tuple[str, List[str], str, Union[str, None]]:
    """
    Generate gridfunction names for the specified Method of Lines (MoL) method.
    Used for setting up MoL_malloc, MoL_step_forward_in_time, MoL_free, and BHaH_defines.h

    :param Butcher_dict: Dictionary of Butcher tables for the MoL method.
    :param MoL_method: The MoL method to generate gridfunction names for.
    :return: A tuple containing y_n_gridfunctions, non_y_n_gridfunctions_list,
             diagnostic_gridfunctions_point_to, and diagnostic_gridfunctions2_point_to.

    Doctests:
    >>> Butcher_dict = generate_Butcher_tables()
    >>> generate_gridfunction_names(Butcher_dict, "RK2 Heun")
    ('y_n_gfs', ['y_nplus1_running_total_gfs', 'k_odd_gfs', 'k_even_gfs', 'auxevol_gfs'], 'y_nplus1_running_total_gfs', 'k_odd_gfs')
    >>> generate_gridfunction_names(Butcher_dict, "RK3")
    ('y_n_gfs', ['next_y_input_gfs', 'k1_gfs', 'k2_gfs', 'k3_gfs', 'auxevol_gfs'], 'k1_gfs', 'k2_gfs')
    >>> generate_gridfunction_names(Butcher_dict, "RK3 Ralston")
    ('y_n_gfs', ['k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs', 'k2_or_y_nplus_a32_k2_gfs', 'auxevol_gfs'], 'k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs', 'k2_or_y_nplus_a32_k2_gfs')
    >>> generate_gridfunction_names(Butcher_dict, "RK4")
    ('y_n_gfs', ['y_nplus1_running_total_gfs', 'k_odd_gfs', 'k_even_gfs', 'auxevol_gfs'], 'y_nplus1_running_total_gfs', 'k_odd_gfs')
    """
    # y_n_gridfunctions store data for the vector of gridfunctions y_i at t_n,
    # the start of each MoL timestep.
    y_n_gridfunctions = "y_n_gfs"

    # non_y_n_gridfunctions are needed to compute the data at t_{n+1}.
    # Often labeled with k_i in the name, these gridfunctions are *not*
    # needed at the start of each timestep.
    non_y_n_gridfunctions_list = []

    # Diagnostic output gridfunctions diagnostic_output_gfs & diagnostic_output_gfs2.
    if is_diagonal_Butcher(Butcher_dict, MoL_method) and "RK3" in MoL_method:
        non_y_n_gridfunctions_list.extend(
            [
                "k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs",
                "k2_or_y_nplus_a32_k2_gfs",
            ]
        )
        diagnostic_gridfunctions_point_to = (
            "k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs"
        )
        diagnostic_gridfunctions2_point_to = "k2_or_y_nplus_a32_k2_gfs"
    else:
        if not is_diagonal_Butcher(Butcher_dict, MoL_method):
            # Allocate memory for non-diagonal Butcher tables
            # Determine the number of k_i steps based on length of Butcher Table
            num_k = len(Butcher_dict[MoL_method][0]) - 1
            # For non-diagonal tables, an intermediate gridfunction "next_y_input" is used for rhs evaluations
            non_y_n_gridfunctions_list.append("next_y_input_gfs")
            for i in range(num_k):  # Need to allocate all k_i steps for a given method
                non_y_n_gridfunctions_list.append(f"k{i + 1}_gfs")
            diagnostic_gridfunctions_point_to = "k1_gfs"
            if "k2_gfs" in non_y_n_gridfunctions_list:
                diagnostic_gridfunctions2_point_to = "k2_gfs"
            else:
                print(
                    "MoL WARNING: No gridfunction group available for diagnostic_output_gfs2"
                )
                diagnostic_gridfunctions2_point_to = None
        else:
            # Allocate memory for diagonal Butcher tables, which use a "y_nplus1_running_total gridfunction"
            non_y_n_gridfunctions_list.append("y_nplus1_running_total_gfs")
            if MoL_method != "Euler":
                # Allocate memory for diagonal Butcher tables that aren't Euler
                # Need k_odd for k_1,3,5... and k_even for k_2,4,6...
                non_y_n_gridfunctions_list.extend(["k_odd_gfs", "k_even_gfs"])
            diagnostic_gridfunctions_point_to = "y_nplus1_running_total_gfs"
            diagnostic_gridfunctions2_point_to = "k_odd_gfs"

    non_y_n_gridfunctions_list.append("auxevol_gfs")

    return (
        y_n_gridfunctions,
        non_y_n_gridfunctions_list,
        diagnostic_gridfunctions_point_to,
        diagnostic_gridfunctions2_point_to,
    )


class base_register_CFunction_MoL_malloc:
    
    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        which_gfs: str,
    ) -> None:
        """
        Base class to generate MoL_malloc_y_n_gfs() and MoL_malloc_non_y_n_gfs(), allocating memory for the gridfunctions indicated.

        :param Butcher_dict: Dictionary of Butcher tables for the MoL method.
        :param MoL_method: Method for the Method of Lines.
        :param which_gfs: Specifies which gridfunctions to consider ("y_n_gfs" or "non_y_n_gfs").

        :raises ValueError: If the which_gfs parameter is neither "y_n_gfs" nor "non_y_n_gfs".

        Doctest: FIXME
        # >>> register_CFunction_MoL_malloc("Euler", "y_n_gfs")
        """
        self.Butcher_dict=Butcher_dict
        self.MoL_method=MoL_method
        self.which_gfs=which_gfs
        self.includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

        # Create a description for the function
        self.desc = f"""Method of Lines (MoL) for "{self.MoL_method}" method: Allocate memory for "{self.which_gfs}" gridfunctions
   - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
   - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method"""

        self.cfunc_type = "void"

        (
            self.y_n_gridfunctions,
            self.non_y_n_gridfunctions_list,
            self.diagnostic_gridfunctions_point_to,
            self.diagnostic_gridfunctions2_point_to,
        ) = generate_gridfunction_names(self.Butcher_dict, MoL_method=self.MoL_method)

        # Determine which gridfunctions to allocate memory for
        if self.which_gfs == "y_n_gfs":
            self.gridfunctions_list = [self.y_n_gridfunctions]
        elif self.which_gfs == "non_y_n_gfs":
            self.gridfunctions_list = self.non_y_n_gridfunctions_list
        else:
            raise ValueError(f'ERROR: which_gfs = "{self.which_gfs}" unrecognized.')

        self.name = f"MoL_malloc_{self.which_gfs}"
        self.params = "const commondata_struct *restrict commondata, const params_struct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs"
        
        # Generate the body of the function        
        self.body = "const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;\n"


# single_RK_substep_input_symbolic() performs necessary replacements to
#   define C code for a single RK substep
#   (e.g., computing k_1 and then updating the outer boundaries)
def single_RK_substep_input_symbolic(
    comment_block: str,
    substep_time_offset_dt: Union[sp.Basic, int, str],
    rhs_str: str,
    rhs_input_expr: sp.Basic,
    rhs_output_expr: sp.Basic,
    RK_lhs_list: Union[sp.Basic, List[sp.Basic]],
    RK_rhs_list: Union[sp.Basic, List[sp.Basic]],
    post_rhs_list: Union[str, List[str]],
    post_rhs_output_list: Union[sp.Basic, List[sp.Basic]],
    enable_simd: bool = False,
    gf_aliases: str = "",
    post_post_rhs_string: str = "",
    fp_type: str = "double",
) -> str:
    """
    Generate C code for a given Runge-Kutta substep.

    :param comment_block: Block of comments for the generated code.
    :param substep_time_offset_dt: Time offset for the RK substep.
    :param rhs_str: Right-hand side string of the C code.
    :param rhs_input_expr: Input expression for the RHS.
    :param rhs_output_expr: Output expression for the RHS.
    :param RK_lhs_list: List of LHS expressions for RK.
    :param RK_rhs_list: List of RHS expressions for RK.
    :param post_rhs_list: List of post-RHS expressions.
    :param post_rhs_output_list: List of outputs for post-RHS expressions.
    :param enable_simd: Whether SIMD optimization is enabled.
    :param gf_aliases: Additional aliases for grid functions.
    :param post_post_rhs_string: String to be used after the post-RHS phase.
    :param fp_type: Floating point type, e.g., "double".

    :return: A string containing the generated C code.

    :raises ValueError: If substep_time_offset_dt cannot be extracted from the Butcher table.
    """
    # Ensure all input lists are lists
    RK_lhs_list = [RK_lhs_list] if not isinstance(RK_lhs_list, list) else RK_lhs_list
    RK_rhs_list = [RK_rhs_list] if not isinstance(RK_rhs_list, list) else RK_rhs_list
    post_rhs_list = (
        [post_rhs_list] if not isinstance(post_rhs_list, list) else post_rhs_list
    )
    post_rhs_output_list = (
        [post_rhs_output_list]
        if not isinstance(post_rhs_output_list, list)
        else post_rhs_output_list
    )

    return_str = f"{comment_block}\n"
    if isinstance(substep_time_offset_dt, (int, sp.Rational)):
        substep_time_offset_str = f"{float(substep_time_offset_dt):.17e}"
    else:
        raise ValueError(
            f"Could not extract substep_time_offset_dt={substep_time_offset_dt} from Butcher table"
        )
    return_str += "for(int grid=0; grid<commondata->NUMGRIDS; grid++) {\n"
    return_str += (
        f"commondata->time = time_start + {substep_time_offset_str} * commondata->dt;\n"
    )
    return_str += gf_aliases

    # Part 1: RHS evaluation
    updated_rhs_str = (
        str(rhs_str)
        .replace("RK_INPUT_GFS", str(rhs_input_expr).replace("gfsL", "gfs"))
        .replace("RK_OUTPUT_GFS", str(rhs_output_expr).replace("gfsL", "gfs"))
    )
    return_str += updated_rhs_str + "\n"

    # Part 2: RK update
    if enable_simd:
        warnings.warn(
            "enable_simd in MoL is not properly supported -- MoL update loops are not properly bounds checked."
        )
        return_str += "#pragma omp parallel for\n"
        return_str += "for(int i=0;i<Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;i+=simd_width) {{\n"
    else:
        return_str += "LOOP_ALL_GFS_GPS(i) {\n"

    var_type = "REAL_SIMD_ARRAY" if enable_simd else "REAL"

    RK_lhs_str_list = [
        (
            f"const REAL_SIMD_ARRAY __rhs_exp_{i}"
            if enable_simd
            else f"{str(el).replace('gfsL', 'gfs[i]')}"
        )
        for i, el in enumerate(RK_lhs_list)
    ]

    read_list = [
        read for el in RK_rhs_list for read in list(sp.ordered(el.free_symbols))
    ]
    read_list_unique = superfast_uniq(read_list)

    for el in read_list_unique:
        if str(el) != "commondata->dt":
            if enable_simd:
                simd_el = str(el).replace("gfsL", "gfs[i]")
                return_str += f"const {var_type} {el} = ReadSIMD(&{simd_el});\n"
            else:
                return_str += (
                    f"const {var_type} {el} = {str(el).replace('gfsL', 'gfs[i]')};\n"
                )

    if enable_simd:
        return_str += "const REAL_SIMD_ARRAY DT = ConstSIMD(commondata->dt);\n"

    kernel = c_codegen(
        RK_rhs_list,
        RK_lhs_str_list,
        include_braces=False,
        verbose=False,
        enable_simd=enable_simd,
        fp_type=fp_type,
    )

    if enable_simd:
        return_str += kernel.replace("commondata->dt", "DT")
        for i, el in enumerate(RK_lhs_list):
            return_str += (
                f"  WriteSIMD(&{str(el).replace('gfsL', 'gfs[i]')}, __rhs_exp_{i});\n"
            )

    else:
        return_str += kernel

    return_str += "}\n"

    # Part 3: Call post-RHS functions
    for post_rhs, post_rhs_output in zip(post_rhs_list, post_rhs_output_list):
        return_str += post_rhs.replace(
            "RK_OUTPUT_GFS", str(post_rhs_output).replace("gfsL", "gfs")
        )

    return_str += "}\n"

    for post_rhs, post_rhs_output in zip(post_rhs_list, post_rhs_output_list):
        return_str += post_post_rhs_string.replace(
            "RK_OUTPUT_GFS", str(post_rhs_output).replace("gfsL", "gfs")
        )

    return return_str


########################################################################################################################
# EXAMPLE
# ODE: y' = f(t,y), y(t_0) = y_0
# Starting at time t_n with solution having value y_n and trying to update to y_nplus1 with timestep dt

# Example of scheme for RK4 with k_1, k_2, k_3, k_4 (Using non-diagonal algorithm) Notice this requires storage of
# y_n, y_nplus1, k_1 through k_4

# k_1      = dt*f(t_n, y_n)
# k_2      = dt*f(t_n + 1/2*dt, y_n + 1/2*k_1)
# k_3      = dt*f(t_n + 1/2*dt, y_n + 1/2*k_2)
# k_4      = dt*f(t_n + dt, y_n + k_3)
# y_nplus1 = y_n + 1/3k_1 + 1/6k_2 + 1/6k_3 + 1/3k_4

# Example of scheme RK4 using only k_odd and k_even (Diagonal algroithm) Notice that this only requires storage


# k_odd     = dt*f(t_n, y_n)
# y_nplus1  = 1/3*k_odd
# k_even    = dt*f(t_n + 1/2*dt, y_n + 1/2*k_odd)
# y_nplus1 += 1/6*k_even
# k_odd     = dt*f(t_n + 1/2*dt, y_n + 1/2*k_even)
# y_nplus1 += 1/6*k_odd
# k_even    = dt*f(t_n + dt, y_n + k_odd)
# y_nplus1 += 1/3*k_even
########################################################################################################################
class base_register_CFunction_MoL_step_forward_in_time:
    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        rhs_string: str = "",
        post_rhs_string: str = "",
        post_post_rhs_string: str = "",
        enable_rfm_precompute: bool = False,
        enable_curviBCs: bool = False,
        fp_type: str = "double",
    ) -> None:
        """
        Register MoL_step_forward_in_time() C function, which is the core driver for time evolution in BHaH codes.

        :param Butcher_dict: A dictionary containing the Butcher tables for various RK-like methods.
        :param MoL_method: The method of lines (MoL) used for time-stepping.
        :param rhs_string: Right-hand side string of the C code.
        :param post_rhs_string: Input string for post-RHS phase in the C code.
        :param post_post_rhs_string: String to be used after the post-RHS phase.
        :param enable_rfm_precompute: Flag to enable reference metric functionality.
        :param enable_curviBCs: Flag to enable curvilinear boundary conditions.
        :param enable_simd: Flag to enable SIMD functionality.
        :param fp_type: Floating point type, e.g., "double".

        Doctest:
        # FIXME
        """
        self.Butcher_dict=Butcher_dict
        self.MoL_method=MoL_method
        self.rhs_string=rhs_string
        self.post_rhs_string=post_rhs_string
        self.post_post_rhs_string=post_post_rhs_string
        self.enable_rfm_precompute=enable_rfm_precompute
        self.enable_curviBCs=enable_curviBCs
        self.fp_type=fp_type
        
        #FIXME
        self.enable_simd=False
        
        self.includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

        self.desc = f'Method of Lines (MoL) for "{self.MoL_method}" method: Step forward one full timestep.\n'
        self.cfunc_type = "void"
        self.name = "MoL_step_forward_in_time"
        self.params = (
            "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
        )

        # Code body
        self.body = f"""
    // C code implementation of -={{ {self.MoL_method} }}=- Method of Lines timestepping.

    // First set the initial time:
    const REAL time_start = commondata->time;
    """
        (
            self.y_n_gridfunctions,
            self.non_y_n_gridfunctions_list,
            _throwaway,
            _throwaway2,
        ) = generate_gridfunction_names(self.Butcher_dict, self.MoL_method)

        self.gf_prefix = "griddata[grid].gridfuncs."

        self.gf_aliases = f"""// Set gridfunction aliases from gridfuncs struct
// y_n gridfunctions
REAL *restrict {self.y_n_gridfunctions} = {self.gf_prefix}{self.y_n_gridfunctions};
// Temporary timelevel & AUXEVOL gridfunctions:\n"""
        for gf in self.non_y_n_gridfunctions_list:
            self.gf_aliases += f"REAL *restrict {gf} = {self.gf_prefix}{gf};\n"

        self.gf_aliases += "params_struct *restrict params = &griddata[grid].params;\n"
        if self.enable_rfm_precompute:
            self.gf_aliases += (
                "const rfm_struct *restrict rfmstruct = &griddata[grid].rfmstruct;\n"
            )
        else:
            self.gf_aliases += "REAL *restrict xx[3]; for(int ww=0;ww<3;ww++) xx[ww] = griddata[grid].xx[ww];\n"

        if self.enable_curviBCs:
            self.gf_aliases += "const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;\n"
        for i in ["0", "1", "2"]:
            self.gf_aliases += f"const int Nxx_plus_2NGHOSTS{i} = griddata[grid].params.Nxx_plus_2NGHOSTS{i};\n"

        # Implement Method of Lines (MoL) Timestepping
        self.Butcher = self.Butcher_dict[self.MoL_method][
            0
        ]  # Get the desired Butcher table from the dictionary
        num_steps = (
            len(self.Butcher) - 1
        )  # Specify the number of required steps to update solution

        dt = sp.Symbol("commondata->dt", real=True)

        if is_diagonal_Butcher(Butcher_dict, MoL_method) and "RK3" in MoL_method:
            # Diagonal RK3 only!!!
            #  In a diagonal RK3 method, only 3 gridfunctions need be defined. Below implements this approach.
            y_n_gfs = sp.Symbol("y_n_gfsL", real=True)
            k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs = sp.Symbol(
                "k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfsL", real=True
            )
            k2_or_y_nplus_a32_k2_gfs = sp.Symbol("k2_or_y_nplus_a32_k2_gfsL", real=True)
            # k_1
            self.body += """
// In a diagonal RK3 method like this one, only 3 gridfunctions need be defined. Below implements this approach.
// Using y_n_gfs as input, k1 and apply boundary conditions\n"""
            self.body += (
                single_RK_substep_input_symbolic(
                    comment_block="""// -={ START k1 substep }=-
// RHS evaluation:
//  1. We will store k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs now as
//     ...  the update for the next rhs evaluation y_n + a21*k1*dt
// Post-RHS evaluation:
//  1. Apply post-RHS to y_n + a21*k1*dt""",
                    substep_time_offset_dt=self.Butcher[0][0],
                    rhs_str=self.rhs_string,
                    rhs_input_expr=y_n_gfs,
                    rhs_output_expr=k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                    RK_lhs_list=[k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs],
                    RK_rhs_list=[
                        self.Butcher[1][1]
                        * k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs
                        * dt
                        + y_n_gfs
                    ],
                    post_rhs_list=[self.post_rhs_string],
                    post_rhs_output_list=[
                        k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs
                    ],
                    enable_simd=enable_simd,
                    gf_aliases=self.gf_aliases,
                    post_post_rhs_string=self.post_post_rhs_string,
                    fp_type=self.fp_type,
                )
                + "// -={ END k1 substep }=-\n\n"
            )

            # k_2
            self.body += (
                single_RK_substep_input_symbolic(
                    comment_block="""// -={ START k2 substep }=-
// RHS evaluation:
//    1. Reassign k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs to be the running total y_{n+1}; a32*k2*dt to the running total
//    2. Store k2_or_y_nplus_a32_k2_gfs now as y_n + a32*k2*dt
// Post-RHS evaluation:
//    1. Apply post-RHS to both y_n + a32*k2 (stored in k2_or_y_nplus_a32_k2_gfs)
//       ... and the y_{n+1} running total, as they have not been applied yet to k2-related gridfunctions""",
                    rhs_str=self.rhs_string,
                    substep_time_offset_dt=self.Butcher[1][0],
                    rhs_input_expr=k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                    rhs_output_expr=k2_or_y_nplus_a32_k2_gfs,
                    RK_lhs_list=[
                        k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                        k2_or_y_nplus_a32_k2_gfs,
                    ],
                    RK_rhs_list=[
                        self.Butcher[3][1]
                        * (k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs - y_n_gfs)
                        / self.Butcher[1][1]
                        + y_n_gfs
                        + self.Butcher[3][2] * k2_or_y_nplus_a32_k2_gfs * dt,
                        self.Butcher[2][2] * k2_or_y_nplus_a32_k2_gfs * dt + y_n_gfs,
                    ],
                    post_rhs_list=[self.post_rhs_string, self.post_rhs_string],
                    post_rhs_output_list=[
                        k2_or_y_nplus_a32_k2_gfs,
                        k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs,
                    ],
                    enable_simd=self.enable_simd,
                    gf_aliases=self.gf_aliases,
                    post_post_rhs_string=self.post_post_rhs_string,
                    fp_type=self.fp_type,
                )
                + "// -={ END k2 substep }=-\n\n"
            )

            # k_3
            self.body += (
                single_RK_substep_input_symbolic(
                    comment_block="""// -={ START k3 substep }=-
// RHS evaluation:
//    1. Add k3 to the running total and save to y_n
// Post-RHS evaluation:
//    1. Apply post-RHS to y_n""",
                    substep_time_offset_dt=self.Butcher[2][0],
                    rhs_str=rhs_string,
                    rhs_input_expr=k2_or_y_nplus_a32_k2_gfs,
                    rhs_output_expr=y_n_gfs,
                    RK_lhs_list=[y_n_gfs],
                    RK_rhs_list=[
                        k1_or_y_nplus_a21_k1_or_y_nplus1_running_total_gfs
                        + self.Butcher[3][3] * y_n_gfs * dt
                    ],
                    post_rhs_list=[post_rhs_string],
                    post_rhs_output_list=[y_n_gfs],
                    enable_simd=enable_simd,
                    gf_aliases=self.gf_aliases,
                    post_post_rhs_string=self.post_post_rhs_string,
                    fp_type=self.fp_type,
                )
                + "// -={ END k3 substep }=-\n\n"
            )
        else:
            y_n = sp.Symbol("y_n_gfsL", real=True)                
            if not is_diagonal_Butcher(Butcher_dict, MoL_method):
                for s in range(num_steps):
                    next_y_input = sp.Symbol("next_y_input_gfsL", real=True)

                    # If we're on the first step (s=0), we use y_n gridfunction as input.
                    #      Otherwise next_y_input is input. Output is just the reverse.
                    if s == 0:  # If on first step:
                        rhs_input = y_n
                    else:  # If on second step or later:
                        rhs_input = next_y_input
                    rhs_output = sp.Symbol("k" + str(s + 1) + "_gfs", real=True)
                    if s == num_steps - 1:  # If on final step:
                        RK_lhs = y_n
                    else:  # If on anything but the final step:
                        RK_lhs = next_y_input
                    RK_rhs = y_n
                    for m in range(s + 1):
                        k_mp1_gfs = sp.Symbol("k" + str(m + 1) + "_gfsL")
                        if self.Butcher[s + 1][m + 1] != 0:
                            if self.Butcher[s + 1][m + 1] != 1:
                                RK_rhs += dt * k_mp1_gfs * self.Butcher[s + 1][m + 1]
                            else:
                                RK_rhs += dt * k_mp1_gfs

                    post_rhs = self.post_rhs_string
                    if s == num_steps - 1:  # If on final step:
                        post_rhs_output = y_n
                    else:  # If on anything but the final step:
                        post_rhs_output = next_y_input

                    self.body += f"""{single_RK_substep_input_symbolic(
                        comment_block=f"// -={{ START k{str(s + 1)} substep }}=-",
                        substep_time_offset_dt=self.Butcher[s][0],
                        rhs_str=self.rhs_string,
                        rhs_input_expr=rhs_input,
                        rhs_output_expr=rhs_output,
                        RK_lhs_list=[RK_lhs],
                        RK_rhs_list=[RK_rhs],
                        post_rhs_list=[post_rhs],
                        post_rhs_output_list=[post_rhs_output],
                        enable_simd=self.enable_simd,
                        gf_aliases=self.gf_aliases,
                        post_post_rhs_string=self.post_post_rhs_string,
                        fp_type=self.fp_type,
                    )}// -={{ END k{str(s + 1)} substep }}=-\n\n"""

            else:
                y_n = sp.Symbol("y_n_gfsL", real=True)
                y_nplus1_running_total = sp.Symbol("y_nplus1_running_total_gfsL", real=True)
                if (
                    MoL_method == "Euler"
                ):  # Euler's method doesn't require any k_i, and gets its own unique algorithm
                    self.body += single_RK_substep_input_symbolic(
                        comment_block="// ***Euler timestepping only requires one RHS evaluation***",
                        substep_time_offset_dt=self.Butcher[0][0],
                        rhs_str=self.rhs_string,
                        rhs_input_expr=y_n,
                        rhs_output_expr=y_nplus1_running_total,
                        RK_lhs_list=[y_n],
                        RK_rhs_list=[y_n + y_nplus1_running_total * dt],
                        post_rhs_list=[self.post_rhs_string],
                        post_rhs_output_list=[y_n],
                        enable_simd=self.enable_simd,
                        gf_aliases=self.gf_aliases,
                        post_post_rhs_string=self.post_post_rhs_string,
                        fp_type=self.fp_type,
                    )
                else:
                    for s in range(num_steps):
                        # If we're on the first step (s=0), we use y_n gridfunction as input.
                        # and k_odd as output.
                        if s == 0:
                            rhs_input = sp.Symbol("y_n_gfsL", real=True)
                            rhs_output = sp.Symbol("k_odd_gfsL", real=True)
                        # For the remaining steps the inputs and ouputs alternate between k_odd and k_even
                        elif s % 2 == 0:
                            rhs_input = sp.Symbol("k_even_gfsL", real=True)
                            rhs_output = sp.Symbol("k_odd_gfsL", real=True)
                        else:
                            rhs_input = sp.Symbol("k_odd_gfsL", real=True)
                            rhs_output = sp.Symbol("k_even_gfsL", real=True)
                        RK_lhs_list: List[sp.Basic] = []
                        RK_rhs_list = []
                        if s != num_steps - 1:  # For anything besides the final step
                            if s == 0:  # The first RK step
                                RK_lhs_list.append(y_nplus1_running_total)
                                RK_rhs_list.append(
                                    rhs_output * dt * self.Butcher[num_steps][s + 1]
                                )

                                RK_lhs_list.append(rhs_output)
                                RK_rhs_list.append(
                                    y_n + rhs_output * dt * self.Butcher[s + 1][s + 1]
                                )
                            else:
                                if self.Butcher[num_steps][s + 1] != 0:
                                    RK_lhs_list.append(y_nplus1_running_total)
                                    if self.Butcher[num_steps][s + 1] != 1:
                                        RK_rhs_list.append(
                                            y_nplus1_running_total
                                            + rhs_output * dt * self.Butcher[num_steps][s + 1]
                                        )
                                    else:
                                        RK_rhs_list.append(
                                            y_nplus1_running_total + rhs_output * dt
                                        )
                                if self.Butcher[s + 1][s + 1] != 0:
                                    RK_lhs_list.append(rhs_output)
                                    if self.Butcher[s + 1][s + 1] != 1:
                                        RK_rhs_list.append(
                                            y_n + rhs_output * dt * self.Butcher[s + 1][s + 1]
                                        )
                                    else:
                                        RK_rhs_list.append(y_n + rhs_output * dt)
                            post_rhs_output = rhs_output
                        if s == num_steps - 1:  # If on the final step
                            if self.Butcher[num_steps][s + 1] != 0:
                                RK_lhs_list.append(y_n)
                                if self.Butcher[num_steps][s + 1] != 1:
                                    RK_rhs_list.append(
                                        y_n
                                        + y_nplus1_running_total
                                        + rhs_output * dt * self.Butcher[num_steps][s + 1]
                                    )
                                else:
                                    RK_rhs_list.append(
                                        y_n + y_nplus1_running_total + rhs_output * dt
                                    )
                            post_rhs_output = y_n
                        self.body += (
                            single_RK_substep_input_symbolic(
                                comment_block=f"// -={{ START k{s + 1} substep }}=-",
                                substep_time_offset_dt=self.Butcher[s][0],
                                rhs_str=self.rhs_string,
                                rhs_input_expr=rhs_input,
                                rhs_output_expr=rhs_output,
                                RK_lhs_list=RK_lhs_list,
                                RK_rhs_list=RK_rhs_list,
                                post_rhs_list=[self.post_rhs_string],
                                post_rhs_output_list=[post_rhs_output],
                                enable_simd=self.enable_simd,
                                gf_aliases=self.gf_aliases,
                                post_post_rhs_string=self.post_post_rhs_string,
                                fp_type=self.fp_type,
                            )
                            + f"// -={{ END k{s + 1} substep }}=-\n\n"
                        )

        self.body += """
// Adding dt to commondata->time many times will induce roundoff error,
//   so here we set time based on the iteration number.
commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

// Finally, increment the timestep n:
commondata->nn++;
"""
        cfc.register_CFunction(
            includes=self.includes,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=self.params,
            include_CodeParameters_h=False,
            body=self.body,
        )

class base_register_CFunction_MoL_free_memory:
    def __init__(
        self,
        Butcher_dict: Dict[str, Tuple[List[List[Union[sp.Basic, int, str]]], int]],
        MoL_method: str,
        which_gfs: str,
    ) -> None:
        """
        Base class to generate function to Free memory for the specified Method of Lines (MoL) gridfunctions, given an MoL_method.

        :param Butcher_dict: Dictionary containing Butcher tableau for MoL methods.
        :param MoL_method: The Method of Lines method.
        :param which_gfs: The gridfunctions to be freed, either 'y_n_gfs' or 'non_y_n_gfs'.

        :raises ValueError: If the 'which_gfs' argument is unrecognized.
        """
        self.Butcher_dict=Butcher_dict
        self.MoL_method=MoL_method
        self.which_gfs=which_gfs
        
        self.includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
        self.desc = f'Method of Lines (MoL) for "{MoL_method}" method: Free memory for "{which_gfs}" gridfunctions\n'
        self.desc += "   - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep\n"
        self.desc += "   - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method\n"
        self.cfunc_type = "void"

        (
            self.y_n_gridfunctions,
            self.non_y_n_gridfunctions_list,
            _diagnostic_gridfunctions_point_to,
            _diagnostic_gridfunctions2_point_to,
        ) = generate_gridfunction_names(self.Butcher_dict, self.MoL_method)

        if self.which_gfs == "y_n_gfs":
            self.gridfunctions_list = [self.y_n_gridfunctions]
        elif self.which_gfs == "non_y_n_gfs":
            self.gridfunctions_list = self.non_y_n_gridfunctions_list
        else:
            raise ValueError(f'ERROR: which_gfs = "{self.which_gfs}" unrecognized.')

        self.name: str = f"MoL_free_memory_{self.which_gfs}"
        self.params = "MoL_gridfunctions_struct *restrict gridfuncs"
        self.body = ""


# Register all the CFunctions and NRPy basic defines
class base_register_CFunctions:
    def __init__(
        self,
        MoL_method: str = "RK4",
        rhs_string: str = "rhs_eval(Nxx, Nxx_plus_2NGHOSTS, dxx, RK_INPUT_GFS, RK_OUTPUT_GFS);",
        post_rhs_string: str = "apply_bcs(Nxx, Nxx_plus_2NGHOSTS, RK_OUTPUT_GFS);",
        post_post_rhs_string: str = "",
        enable_rfm_precompute: bool = False,
        enable_curviBCs: bool = False,
        register_MoL_step_forward_in_time: bool = True,
        fp_type: str = "double",
    ) -> None:
        r"""
        Base class to generate MoL C functions and NRPy basic defines.

        :param MoL_method: The method to be used for MoL. Default is 'RK4'.
        :param rhs_string: RHS function call as string. Default is "rhs_eval(Nxx, Nxx_plus_2NGHOSTS, dxx, RK_INPUT_GFS, RK_OUTPUT_GFS);"
        :param post_rhs_string: Post-RHS function call as string. Default is "apply_bcs(Nxx, Nxx_plus_2NGHOSTS, RK_OUTPUT_GFS);"
        :param post_post_rhs_string: Post-post-RHS function call as string. Default is an empty string.
        :param enable_rfm_precompute: Enable reference metric support. Default is False.
        :param enable_curviBCs: Enable curvilinear boundary conditions. Default is False.
        :param enable_simd: Enable Single Instruction, Multiple Data (SIMD). Default is False.
        :param register_MoL_step_forward_in_time: Whether to register the MoL step forward function. Default is True.
        :param fp_type: Floating point type, e.g., "double".
        """
        self.MoL_method=MoL_method
        self.rhs_string=rhs_string
        self.post_rhs_string=post_rhs_string
        self.post_post_rhs_string=post_post_rhs_string
        self.enable_rfm_precompute=enable_rfm_precompute
        self.enable_curviBCs=enable_curviBCs
        self.register_MoL_step_forward_in_time=register_MoL_step_forward_in_time
        self.fp_type=fp_type
        
        self.Butcher_dict = generate_Butcher_tables()

        # Generating gridfunction names based on the given MoL method
        (
            self.y_n_gridfunctions,
            self.non_y_n_gridfunctions_list,
            _diagnostic_gridfunctions_point_to,
            _diagnostic_gridfunctions2_point_to,
        ) = generate_gridfunction_names(self.Butcher_dict, MoL_method=self.MoL_method)

        # Step 3.b: Create MoL_timestepping struct:
        print(self.non_y_n_gridfunctions_list)
        self.BHaH_MoL_body = f"typedef struct __MoL_gridfunctions_struct__ {{\n" \
        + f"REAL *restrict {self.y_n_gridfunctions};\n" \
        + "".join(f"REAL *restrict {gfs};\n" for gfs in self.non_y_n_gridfunctions_list) \
        + r"""REAL *restrict diagnostic_output_gfs;
REAL *restrict diagnostic_output_gfs2;
} MoL_gridfunctions_struct;
"""



if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")