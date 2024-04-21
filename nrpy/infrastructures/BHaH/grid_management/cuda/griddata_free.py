"""
Class to register griddata_free method for CUDA based applications.

Author: Samuel D. Tootle
        sdtootle **at** gmail **dot** com        
"""

import nrpy.infrastructures.BHaH.grid_management.base_griddata_free as base_free
class register_CFunction_griddata_free(base_free.base_register_CFunction_griddata_free):
    """
    Register the C function griddata_free() to free all memory within the griddata struct.
    Overload based on CUDA parallelization

    :param enable_rfm_precompute: A flag to enable/disable rfm_precompute_free within the C function body.
    :param enable_CurviBCs: A flag to enable/disable freeing CurviBCs within the C function body.
    """    
    def __init__(
        self,
        enable_rfm_precompute: bool,
        enable_CurviBCs: bool,
    ) -> None:
        super().__init__()
        self.params = "const commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_host, const bool enable_free_non_y_n_gfs"
        self.body = r"""for(int grid=0;grid<commondata->NUMGRIDS;grid++) {
"""
        if enable_rfm_precompute:
            self.body += "  rfm_precompute_free(commondata, &griddata[grid].params, &griddata[grid].rfmstruct);\n"
        
        if enable_CurviBCs:
            self.body += r"""
  cudaFree(griddata[grid].bcstruct.inner_bc_array);
  for(int ng=0;ng<NGHOSTS*3;ng++) { 
    cudaFree(griddata[grid].bcstruct.pure_outer_bc_array[ng]); 
  }
"""
        self.body += r"""

  MoL_free_memory_y_n_gfs(&griddata[grid].gridfuncs);
  cudaFreeHost(griddata_host[grid].gridfuncs.y_n_gfs);
  cudaFreeHost(griddata_host[grid].gridfuncs.diagnostic_output_gfs);
  if(enable_free_non_y_n_gfs) {
    for(int i=0;i<3;i++) { 
        cudaFree(griddata[grid].xx[i]);
        free(griddata_host[grid].xx[i]);
    }
  }

} // END for(int grid=0;grid<commondata->NUMGRIDS;grid++)
"""
        self.body += """
  cudaFree(d_gridfunctions_f_infinity);
  cudaFree(d_gridfunctions_wavespeed);
  free(griddata);
  free(griddata_host);
"""
        self.register_CFunction()


if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
