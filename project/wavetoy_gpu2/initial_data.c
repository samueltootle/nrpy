#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/*
 * Set initial data to params.time==0 corresponds to the initial data.
 */
void initial_data(const commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
#pragma omp parallel for
    for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
      const REAL xx2 = xx[2][i2];
      for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
        const REAL xx1 = xx[1][i1];
        for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
          const REAL xx0 = xx[0][i0]; // exact_solution_single_Cartesian_point() takes Cartesian coordinates as input.
          // To avoid confusion in other reference metrics, we make this explicit here.
          const REAL xCart0 = xx0;
          const REAL xCart1 = xx1;
          const REAL xCart2 = xx2;
          exact_solution_single_Cartesian_point(commondata, params, xCart0, xCart1, xCart2, &in_gfs[IDX4(UUGF, i0, i1, i2)],
                                                &in_gfs[IDX4(VVGF, i0, i1, i2)]);
        } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
      }   // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
    }     // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
  }
}
