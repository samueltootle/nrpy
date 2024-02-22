#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "trusted_data_dump/trusted_data_dump_prototypes.h"
/*
 * Set initial data.
 */
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  ID_persist_struct ID_persist;

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
    initial_data_reader__convert_ADM_Cartesian_to_BSSN(commondata, params, griddata[grid].xx, &griddata[grid].bcstruct, &griddata[grid].gridfuncs,
                                                       &ID_persist, BrillLindquist);
    dump_gf_array(grid, params, griddata[grid].gridfuncs.y_n_gfs, "post-initial", "evolv", NUM_EVOL_GFS);
    apply_bcs_outerextrap_and_inner(commondata, params, &griddata[grid].bcstruct, griddata[grid].gridfuncs.y_n_gfs);
    dump_gf_array(grid, params, griddata[grid].gridfuncs.y_n_gfs, "post-initial-bcs", "evolv", NUM_EVOL_GFS);
  }
}
