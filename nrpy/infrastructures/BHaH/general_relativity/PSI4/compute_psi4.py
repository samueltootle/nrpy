"""
Library of C functions for solving the BSSN equations in curvilinear coordinates, using a reference-metric formalism.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.grid as gri
import nrpy.helpers.parallel_codegen as pcg
import nrpy.infrastructures.BHaH.simple_loop as lp
from nrpy.equations.general_relativity import psi4
import nrpy.params as par
import nrpy.helpers.parallelization.utilities as parallel_utils
from nrpy.helpers.expression_utils import (
    generate_definition_header,
    get_params_commondata_symbols_from_expr_list,
)

def register_CFunction_psi4(
    CoordSystem: str,
    OMP_collapse: int,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Add psi4 to Cfunction dictionary.

    :param CoordSystem: Coordinate system to be used.
    :param OMP_collapse: OpenMP collapse clause integer value.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    # Set up the C function for psi4
    parallelization = par.parval_from_str("parallelization")
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "Compute psi4 at all interior gridpoints"
    name = "psi4"

    arg_dict_cuda = {
        "x0": "const REAL *restrict",
        "x1": "const REAL *restrict",
        "x2": "const REAL *restrict",
        "in_gfs": "const REAL *restrict",
        "diagnostic_output_gfs": "REAL *restrict",
    }

    arg_dict_host = {
        "commondata": "const commondata_struct *restrict",
        "params": "const params_struct *restrict",
        **arg_dict_cuda,
    }

    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const REAL *restrict in_gfs, REAL *restrict diagnostic_output_gfs"

    gri.register_gridfunctions(["psi4_re", "psi4_im"], group="AUX")

    psi4_class = psi4.Psi4(CoordSystem=CoordSystem, enable_rfm_precompute=False)
    body = r"""if(! (params->Cart_originx == 0 && params->Cart_originy == 0 && params->Cart_originz == 0) ) {
  fprintf(stderr, "Error: psi4 output assumes that the grid is centered on the origin.\n");
  fprintf(stderr, "       Good news: check out the C code for proposed modifications.\n");
  exit(1);
}
"""
    for i in range(3):
        body += f"const REAL *restrict x{i} = xx[{i}];\n"

    expr_list = [psi4_class.psi4_re, psi4_class.psi4_im]

    # Find symbols stored in params
    param_symbols, commondata_symbols = get_params_commondata_symbols_from_expr_list(
        expr_list, exclude=[f"xx{j}" for j in range(3)]
    )
    loop_params = parallel_utils.get_loop_parameters(
        parallelization
    )

    params_definitions = generate_definition_header(
        param_symbols,
        var_access=parallel_utils.get_params_access(parallelization),
    )
    kernel_body = f"{loop_params}\n{params_definitions}\n"
    body += kernel_body
    loop_prefix = rf"""
REAL xx0, xx1, xx2;
{{
  xx0 = x0[i0];
  xx1 = x1[i1];
  xx2 = x2[i2];

/* PROPOSED MODIFICATIONS FOR COMPUTING PSI4 ON GRIDS NOT CENTERED ON THE ORIGIN
    REAL xCart_rel_to_globalgrid_center[3];
    REAL xOrig[3] = {{xx[0][i0], xx[1][i1], xx[2][i2]}};
    xx_to_Cart(params, xOrig, xCart_rel_to_globalgrid_center);
    int ignore_Cart_to_i0i1i2[3];  REAL xx_rel_to_globalgridorigin[3];
    Cart_to_xx_and_nearest_i0i1i2_global_grid_center(params, xCart_rel_to_globalgrid_center,xx_rel_to_globalgridorigin,ignore_Cart_to_i0i1i2);
    xx0=xx_rel_to_globalgridorigin[0];
    xx1=xx_rel_to_globalgridorigin[1];
    xx2=xx_rel_to_globalgridorigin[2];
*/
}}

// Metric derivative quantities.
REAL arr_gammaDDdDD[3*3*3*3], arr_GammaUDD[3*3*3], arr_KDDdD[3*3*3];
// Tetrad quantities.
REAL mre4U0,mre4U1,mre4U2,mre4U3,mim4U0,mim4U1,mim4U2,mim4U3,n4U0,n4U1,n4U2,n4U3;
const int idx3 = IDX3(i0, i1, i2);

psi4_tetrad(params,
    in_gfs[IDX4pt(CFGF, idx3)],
    in_gfs[IDX4pt(HDD00GF, idx3)],
    in_gfs[IDX4pt(HDD01GF, idx3)],
    in_gfs[IDX4pt(HDD02GF, idx3)],
    in_gfs[IDX4pt(HDD11GF, idx3)],
    in_gfs[IDX4pt(HDD12GF, idx3)],
    in_gfs[IDX4pt(HDD22GF, idx3)],
    &mre4U0,&mre4U1,&mre4U2,&mre4U3,&mim4U0,&mim4U1,&mim4U2,&mim4U3,&n4U0,&n4U1,&n4U2,&n4U3,
    xx0, xx1, xx2);

psi4_metric_deriv_quantities(params, in_gfs, xx0, xx1, xx2, i0, i1, i2, arr_gammaDDdDD, arr_GammaUDD, arr_KDDdD);
// Next, unpack gammaDDdDD, GammaUDD, KDDdD from their arrays:
MAYBE_UNUSED REAL {psi4_class.metric_deriv_var_list_str};
{psi4_class.metric_deriv_unpack_arrays}
"""
    body += lp.simple_loop(
        loop_body=loop_prefix
        + ccg.c_codegen(
            expr_list,
            [
                gri.BHaHGridFunction.access_gf(
                    "psi4_re",
                    gf_array_name="diagnostic_output_gfs",
                ),
                gri.BHaHGridFunction.access_gf(
                    "psi4_im",
                    gf_array_name="diagnostic_output_gfs",
                ),
            ],
            enable_fd_codegen=True,
        ),
        loop_region="interior",
        enable_intrinsics=False,
        CoordSystem=CoordSystem,
        enable_rfm_precompute=False,
        read_xxs=False,
        OMP_collapse=OMP_collapse,
    )

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        CoordSystem_for_wrapper_func=CoordSystem,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())