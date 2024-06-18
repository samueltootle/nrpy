"""
Set up basic functions and loop insertions for precomputed reference metric infrastructure.
This module overloads the defaults for CUDA based codes

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot** com
        Samuel D. Tootle
        sdtootle **at** gmail **dot** com
"""

from typing import List, Dict, Any
import sympy as sp
import sympy.codegen.ast as sp_ast

import nrpy.c_codegen as ccg
from nrpy.helpers.generic import superfast_uniq
from nrpy.infrastructures.BHaH import rfm_precompute


class ReferenceMetricPrecompute(rfm_precompute.ReferenceMetricPrecompute):
    """
    Class for reference metric precomputation for CUDA.

    This class stores contributions to BHaH_defines.h, as well as functions for memory allocation,
    definition, and freeing of rfm precomputation data. It also provides strings for reading rfm
    precompute quantities within loops for both SIMD-ized and non-SIMD loops.
    """

    def __init__(self, CoordSystem: str, fp_type: str = "double", expansion_form: bool = False):
        super().__init__(CoordSystem, fp_type=fp_type)
        self.rfm_struct__define_kernel_dict: Dict[sp.Expr, Any]
        # rfmstruct stores pointers to (so far) 1D arrays. The rfm_struct__malloc string allocates space for the arrays.
        self.rfm_struct__malloc = ""
        self.rfm_struct__freemem = ""

        # readvr_str reads the arrays from memory as needed
        self.readvr_str = ["", "", ""]
        self.readvr_SIMD_outer_str = ["", "", ""]
        self.readvr_SIMD_inner_str = ["", "", ""]
        self.rfm_struct__define_kernel_dict = {}

        which_freevar: int = 0
        fp_ccg_type = ccg.fp_type_to_sympy_type[fp_type]
        sp_type_alias = {sp_ast.real: fp_ccg_type}
        if expansion_form:
            self.BHaH_defines_list = [v.replace("REAL *", "float *") for v in self.BHaH_defines_list]
        # print(self.BHaH_defines_list)
        # user_functions = {}
        # if expansion_form:
        #     user_functions = {
        #         "Pow" : [
        #             (lambda b, e: e == sp.Rational(1, 2), lambda b, e: f"expansion_math::sqrt_expansion({b})"),
        #             (lambda b, e: e == 0.5, lambda b, e: f"expansion_math::sqrt_expansion({b})"),
        #             (lambda b, e: e == -sp.Rational(1, 2), lambda b, e: f"(1.0/expansion_math::sqrt_expansion({b}))"),
        #             (lambda b, e: e == -0.5, lambda b, e: f"(1.0/expansion_math::sqrt_expansion({b}))"),
        #             # (lambda b, e: e == sp.S.One / 3, lambda b, e: f"cbrt({b})"),
        #             # (lambda b, e: e == -sp.S.One / 3, lambda b, e: f"(1.0/cbrt({b}))"),
        #             (lambda b, e: e != -0.5, lambda b, e: f"expansion_math::pow_expansion({b}, {e})"),
        #         ],
        #         "Exp" : [(lambda b,e: True, lambda b, e: f"expansion_math::exp_TAYLOR({b})")],
        #         "Sqrt": [(lambda b,e: True, lambda b, e: f"expansion_math::sqrt_expansion({b})")],
        #     }
        for expr in self.freevars_uniq_vals:
            if "_of_xx" in str(self.freevars_uniq_xx_indep[which_freevar]):
                frees = list(expr.free_symbols)
                frees_uniq = superfast_uniq(frees)
                xx_list: List[sp.Basic] = []
                malloc_size: int = 1
                for i in range(3):
                    if self.rfm.xx[i] in frees_uniq:
                        xx_list.append(self.rfm.xx[i])
                        malloc_size *= self.Nxx_plus_2NGHOSTS[i]

                array_type = "float" if expansion_form else "REAL"
                array_factor = 2 if expansion_form else 1
                self.rfm_struct__malloc += f"""cudaMalloc(&rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}, sizeof({array_type})*{malloc_size}*{array_factor});
                    cudaCheckErrors(malloc, "Malloc failed");
                    """
                self.rfm_struct__freemem += f"""cudaFree(rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]});
                cudaCheckErrors(free, "cudaFree failed");
                """

                output_define_and_readvr = False
                for dirn in range(3):
                    if (
                        (self.rfm.xx[dirn] in frees_uniq)
                        and not (self.rfm.xx[(dirn + 1) % 3] in frees_uniq)
                        and not (self.rfm.xx[(dirn + 2) % 3] in frees_uniq)
                    ):
                        key = self.freevars_uniq_xx_indep[which_freevar]
                        if expansion_form:
                            kernel_body = (
                                f"const int Nxx_plus_2NGHOSTS{dirn} = d_params.Nxx_plus_2NGHOSTS{dirn};\n\n"
                                "// Kernel thread/stride setup\n"
                                "const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;\n"
                                "const int stride0 = blockDim.x * gridDim.x;\n\n"
                                f"for(int i{dirn}=tid0;i{dirn}<Nxx_plus_2NGHOSTS{dirn};i{dirn} += 2 * stride0) {{\n"
                                f"  const expansion_math::float2<float> xx{dirn}_exp(x{dirn}[i{dirn}], x{dirn}[i{dirn} + 1]);\n"
                                f"  const REAL xx{dirn} = expansion_math::recast_sum<double>(xx{dirn}_exp);\n"
                                f"  const REAL res = {sp.ccode(self.freevars_uniq_vals[which_freevar], type_aliases=sp_type_alias)};\n"
                                f"  const expansion_math::float2<float> split_res = expansion_math::split<float>(res);\n"
                                f"  {key}[i{dirn}] = split_res.value;\n"
                                f"  {key}[i{dirn}+1] = split_res.remainder;\n"
                                "}"
                            )
                        else:
                            kernel_body = (
                                f"const int Nxx_plus_2NGHOSTS{dirn} = d_params.Nxx_plus_2NGHOSTS{dirn};\n\n"
                                "// Kernel thread/stride setup\n"
                                "const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;\n"
                                "const int stride0 = blockDim.x * gridDim.x;\n\n"
                                f"for(int i{dirn}=tid0;i{dirn}<Nxx_plus_2NGHOSTS{dirn};i{dirn}+=stride0) {{\n"
                                f"  const REAL xx{dirn} = x{dirn}[i{dirn}];\n"
                                f"  {key}[i{dirn}] = {sp.ccode(self.freevars_uniq_vals[which_freevar], type_aliases=sp_type_alias)};\n"
                                "}"
                            )

                        self.rfm_struct__define_kernel_dict[key] = {
                            "body": kernel_body,
                            "expr": self.freevars_uniq_vals[which_freevar],
                            "coord": f"x{dirn}",
                        }

                        # These have to be passed to kernel as rfm_{freevar}
                        if expansion_form:
                            self.readvr_str[
                                dirn
                            ] += f"const expansion_math::float2<float> {self.freevars_uniq_xx_indep[which_freevar]}(rfm_{self.freevars_uniq_xx_indep[which_freevar]}[i{dirn}], rfm_{self.freevars_uniq_xx_indep[which_freevar]}[i{dirn}+1]);\n"
                        else:
                            self.readvr_str[
                                dirn
                            ] += f"const REAL {self.freevars_uniq_xx_indep[which_freevar]} = rfm_{self.freevars_uniq_xx_indep[which_freevar]}[i{dirn}];\n"
                        output_define_and_readvr = True
                if (
                    (not output_define_and_readvr)
                    and (self.rfm.xx[0] in frees_uniq)
                    and (self.rfm.xx[1] in frees_uniq)
                ):
                    if expansion_form:
                        raise RuntimeError(
                            f"ERROR: Not yet implemented for expansion form"
                        )
                    #     self.rfm_struct__define += f"""
                    # for(int i1=0;i1<Nxx_plus_2NGHOSTS1;i1++) for(int i0=0;i0<Nxx_plus_2NGHOSTS0;i0++) {{
                    # const expansion_math::float2<float> xx0(xx[0][i0], xx[0][i0+1]);
                    # const expansion_math::float2<float> xx1(xx[1][i1], xx[1][i1+1]);
                    # rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}[i0 + Nxx_plus_2NGHOSTS0*i1] = {sp.ccode(self.freevars_uniq_vals[which_freevar], type_aliases=sp_type_alias)};
                    # }}\n\n"""
                    #     self.readvr_str[
                    #         0
                    #     ] += f"const expansion_math::float2<float> {self.freevars_uniq_xx_indep[which_freevar]}(rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}[i0 + Nxx_plus_2NGHOSTS0*i1], rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}[i0 + Nxx_plus_2NGHOSTS0*i1 + 1];\n"
                    else:
                        self.rfm_struct__define += f"""
                    for(int i1=0;i1<Nxx_plus_2NGHOSTS1;i1++) for(int i0=0;i0<Nxx_plus_2NGHOSTS0;i0++) {{
                    const REAL xx0 = xx[0][i0];
                    const REAL xx1 = xx[1][i1];
                    rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}[i0 + Nxx_plus_2NGHOSTS0*i1] = {sp.ccode(self.freevars_uniq_vals[which_freevar], type_aliases=sp_type_alias)};
                    }}\n\n"""
                        self.readvr_str[
                            0
                        ] += f"const REAL {self.freevars_uniq_xx_indep[which_freevar]} = rfmstruct->{self.freevars_uniq_xx_indep[which_freevar]}[i0 + Nxx_plus_2NGHOSTS0*i1];\n"
                    output_define_and_readvr = True

                if not output_define_and_readvr:
                    raise RuntimeError(
                        f"ERROR: Could not figure out the (xx0,xx1,xx2) dependency within the expression for {self.freevars_uniq_xx_indep[which_freevar]}: {self.freevars_uniq_vals[which_freevar]}"
                    )

            which_freevar += 1