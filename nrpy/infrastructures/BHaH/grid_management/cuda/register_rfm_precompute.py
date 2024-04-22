"""
CUDA implementation to register CFunctions for precomputed reference metric infrastructure.

Authors: Samuel D. Tootle
        sdtootle **at** gmail **dot** com
        Zachariah B. Etienne
        zachetie **at** gmail **dot** com
"""

from typing import List

import nrpy.c_function as cfc
from nrpy.helpers.generic import superfast_uniq
from nrpy.infrastructures.BHaH.BHaH_defines_h import register_BHaH_defines
from nrpy.infrastructures.BHaH.grid_management.base_register_rfm_precompute import base_register_CFunctions_rfm_precompute
from nrpy.infrastructures.BHaH.grid_management.cuda.rfm_precompute import ReferenceMetricPrecompute
# import nrpy.helpers.gpu_kernel as gputils

class register_CFunctions_rfm_precompute(base_register_CFunctions_rfm_precompute):
    """
    Cuda implementation to register C functions for reference metric precomputed lookup arrays.

    :param list_of_CoordSystems: List of coordinate systems to register the C functions.
    :param fp_type: Floating point type, e.g., "double".
    """
    def __init__(
        self,
        list_of_CoordSystems: List[str], 
        fp_type: str = "double"
    ) -> None:
        super().__init__(list_of_CoordSystems, fp_type=fp_type)
        self.include_CodeParameters_h = False
        
        for CoordSystem in list_of_CoordSystems:
            rfm_precompute = ReferenceMetricPrecompute(CoordSystem, fp_type=self.fp_type)

            for func in [
                ("malloc", rfm_precompute.rfm_struct__malloc),
                ("defines", rfm_precompute.rfm_struct__define),
                ("free", rfm_precompute.rfm_struct__freemem),
            ]:

                desc = f"rfm_precompute_{func[0]}: reference metric precomputed lookup arrays: {func[0]}"
                cfunc_type = "void"
                name = "rfm_precompute_" + func[0]
                params = "const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct"
                if func[0] == "defines":
                    params += ", REAL *restrict xx[3]"

                body = " "
                body += func[1]
                
                self.function_dict[name] = {
                    'desc' : desc,
                    'cfunc_type' : cfunc_type,
                    'params' : params,
                    'body' : body,
                    'CoordSystem' : CoordSystem,
                }
        self.register_CFunction()