"""
Simple loop generation for use within the BHaH infrastructure for GPUs.

Author: Samuel D. Tootle
Email: sdtootle **at** gmail **dot** com
"""

from typing import Any
import nrpy.infrastructures.BHaH.loop_utilities.base_simple_loop as base_sl
from nrpy.infrastructures.BHaH.grid_management.cuda import rfm_precompute


class simple_loop(base_sl.base_simple_loop):
    """
    Generate a simple loop in C (for use inside of a function).

    :param loop_body: Loop body
    :param enable_simd: Enable SIMD support
    :param loop_region: Loop over all points on a numerical grid or just the interior
    :param read_xxs: Read the xx[3][:] 1D coordinate arrays if interior dependency exists
    :param CoordSystem: Coordinate system, e.g., "Cartesian"
    :param enable_rfm_precompute: Enable pre-computation of reference metric
    :param enable_OpenMP: Enable loop parallelization using OpenMP
    :param OMP_custom_pragma: Enable loop parallelization using OpenMP with custom pragma
    :param OMP_collapse: Specifies the number of nested loops to collapse
    :param fp_type: Floating point type, e.g., "double".
    :raises ValueError: If `loop_region` is unsupported or if `read_xxs` and `enable_rfm_precompute` are both enabled.

    Doctests:
    >>> from nrpy.helpers.generic import clang_format
    >>> print(clang_format(simple_loop('// <INTERIOR>', loop_region="all points").full_loop_body))
    <BLANKLINE>
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;
    <BLANKLINE>
    const int stride0 = blockDim.x * gridDim.x;
    const int stride1 = blockDim.y * gridDim.y;
    const int stride2 = blockDim.z * gridDim.z;
    <BLANKLINE>
    for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
      for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
        for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
          // <INTERIOR>
        } // END LOOP: for (int i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0)
      } // END LOOP: for (int i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1)
    } // END LOOP: for (int i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2)
    <BLANKLINE>
    >>> print(clang_format(simple_loop('// <INTERIOR>', loop_region="interior",
    ...       CoordSystem="SinhSymTP", enable_rfm_precompute=True).full_loop_body))
    Setting up reference metric for CoordSystem = SinhSymTP.
    <BLANKLINE>
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid2 = blockIdx.z * blockDim.z + threadIdx.z;
    <BLANKLINE>
    const int stride0 = blockDim.x * gridDim.x;
    const int stride1 = blockDim.y * gridDim.y;
    const int stride2 = blockDim.z * gridDim.z;
    <BLANKLINE>
    for (int i2 = tid2 + NGHOSTS; i2 < NGHOSTS + Nxx2; i2 += stride2) {
      for (int i1 = tid1 + NGHOSTS; i1 < NGHOSTS + Nxx1; i1 += stride1) {
        const REAL f1_of_xx1 = rfmstruct->f1_of_xx1[i1];
        const REAL f1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1[i1];
        const REAL f1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11[i1];
        const REAL f4_of_xx1 = rfmstruct->f4_of_xx1[i1];
        const REAL f4_of_xx1__D1 = rfmstruct->f4_of_xx1__D1[i1];
        const REAL f4_of_xx1__DD11 = rfmstruct->f4_of_xx1__DD11[i1];
    <BLANKLINE>
        for (int i0 = tid0 + NGHOSTS; i0 < NGHOSTS + Nxx0; i0 += stride0) {
          const REAL f0_of_xx0 = rfmstruct->f0_of_xx0[i0];
          const REAL f0_of_xx0__D0 = rfmstruct->f0_of_xx0__D0[i0];
          const REAL f0_of_xx0__DD00 = rfmstruct->f0_of_xx0__DD00[i0];
          const REAL f0_of_xx0__DDD000 = rfmstruct->f0_of_xx0__DDD000[i0];
          const REAL f2_of_xx0 = rfmstruct->f2_of_xx0[i0];
          const REAL f2_of_xx0__D0 = rfmstruct->f2_of_xx0__D0[i0];
          const REAL f2_of_xx0__DD00 = rfmstruct->f2_of_xx0__DD00[i0];
          // <INTERIOR>
        } // END LOOP: for (int i0 = tid0+NGHOSTS; i0 < NGHOSTS+Nxx0; i0 += stride0)
      } // END LOOP: for (int i1 = tid1+NGHOSTS; i1 < NGHOSTS+Nxx1; i1 += stride1)
    } // END LOOP: for (int i2 = tid2+NGHOSTS; i2 < NGHOSTS+Nxx2; i2 += stride2)
    <BLANKLINE>
    """

    def __init__(
        self,
        loop_body: str,
        loop_region: str = "",
        read_xxs: bool = False,
        CoordSystem: str = "Cartesian",
        enable_rfm_precompute: bool = False,
        fp_type: str = "double",
        expansion_form: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            loop_body,
            read_xxs=read_xxs,
            CoordSystem=CoordSystem,
            enable_rfm_precompute=enable_rfm_precompute,
            fp_type=fp_type,
            loop_region=loop_region,
            cuda=True,
        )

        self.rfmp = rfm_precompute.ReferenceMetricPrecompute(
            CoordSystem, fp_type=fp_type, expansion_form=expansion_form
        )

        if self.read_xxs:
            if expansion_form:
              self.read_rfm_xx_arrays = [
                  "[[maybe_unused]] const expansion_math::float2<float> xx0(x0[i0], x0[i0+1]);",
                  "[[maybe_unused]] const expansion_math::float2<float> xx1(x1[i1], x1[i1+1]);",
                  "[[maybe_unused]] const expansion_math::float2<float> xx2(x2[i2], x2[i2+1]);",
              ]
            else:
              self.read_rfm_xx_arrays = [
                  "[[maybe_unused]] const REAL xx0 = x0[i0];",
                  "[[maybe_unused]] const REAL xx1 = x1[i1];",
                  "[[maybe_unused]] const REAL xx2 = x2[i2];",
              ]
        elif self.enable_rfm_precompute:
            self.read_rfm_xx_arrays = [
                self.rfmp.readvr_str[0],
                self.rfmp.readvr_str[1],
                self.rfmp.readvr_str[2],
            ]
        self.initialize_based_on__read_rfm_xx_arrays()

        self.increment = ["stride2", "stride1", f"{int(expansion_form) + 1} * stride0"]
        self.gen_loop_body()
        kernel_setup = f"""
  const int Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;
    
  const int tid0  = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid1  = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid2  = blockIdx.z * blockDim.z + threadIdx.z;
  
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;
"""
        if expansion_form:
          inv_constant_str = """[[maybe_unused]] expansion_math::float2<float> invdxx0 = expansion_math::split<float>(d_params.invdxx0);
  [[maybe_unused]] expansion_math::float2<float> invdxx1 = expansion_math::split<float>(d_params.invdxx1);
  [[maybe_unused]] expansion_math::float2<float> invdxx2 = expansion_math::split<float>(d_params.invdxx2);
"""
        else:
          inv_constant_str = """[[maybe_unused]] const REAL invdxx0 = d_params.invdxx0;
  [[maybe_unused]] const REAL invdxx1 = d_params.invdxx1;
  [[maybe_unused]] const REAL invdxx2 = d_params.invdxx2;
"""
        self.full_loop_body = f"""{kernel_setup}
  
  {inv_constant_str}
  
  {self.full_loop_body}
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