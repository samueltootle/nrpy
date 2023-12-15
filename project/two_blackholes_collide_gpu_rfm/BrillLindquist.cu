#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * BrillLindquist initial data
 */
__device__
void BrillLindquist(const commondata_struct *restrict commondata, const REAL xCart[3],
                    const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data) {
// #include "set_CodeParameters.h"
  // [[maybe_unused]] const REAL BH1_mass = commondata->BH1_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_mass
  // [[maybe_unused]] const REAL BH1_posn_x = commondata->BH1_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_x
  // [[maybe_unused]] const REAL BH1_posn_y = commondata->BH1_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_y
  // [[maybe_unused]] const REAL BH1_posn_z = commondata->BH1_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH1_posn_z
  // [[maybe_unused]] const REAL BH2_mass = commondata->BH2_mass;                     // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_mass
  // [[maybe_unused]] const REAL BH2_posn_x = commondata->BH2_posn_x;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_x
  // [[maybe_unused]] const REAL BH2_posn_y = commondata->BH2_posn_y;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_y
  // [[maybe_unused]] const REAL BH2_posn_z = commondata->BH2_posn_z;                 // nrpy.equations.general_relativity.InitialData_Cartesian::BH2_posn_z
//   const REAL x = xCart[0], y = xCart[1], z = xCart[2];
//   const REAL tmp1 = -y;
//   const REAL tmp3 =
//       (1.0 / 2.0) * BH1_mass /
//           sqrt(((-BH1_posn_x + x) * (-BH1_posn_x + x)) + ((-BH1_posn_y - tmp1) * (-BH1_posn_y - tmp1)) + ((-BH1_posn_z + z) * (-BH1_posn_z + z))) +
//       (1.0 / 2.0) * BH2_mass /
//           sqrt(((-BH2_posn_x + x) * (-BH2_posn_x + x)) + ((-BH2_posn_y - tmp1) * (-BH2_posn_y - tmp1)) + ((-BH2_posn_z + z) * (-BH2_posn_z + z))) +
//       1;
//   const REAL tmp4 = ((tmp3) * (tmp3) * (tmp3) * (tmp3));
//   initial_data->BSphorCartU0 = 0;
//   initial_data->BSphorCartU1 = 0;
//   initial_data->BSphorCartU2 = 0;
//   initial_data->KSphorCartDD00 = 0;
//   initial_data->KSphorCartDD01 = 0;
//   initial_data->KSphorCartDD02 = 0;
//   initial_data->KSphorCartDD11 = 0;
//   initial_data->KSphorCartDD12 = 0;
//   initial_data->KSphorCartDD22 = 0;
//   initial_data->alpha = pow(pow(tmp3, 12), -1.0 / 6.0);
//   initial_data->betaSphorCartU0 = 0;
//   initial_data->betaSphorCartU1 = 0;
//   initial_data->betaSphorCartU2 = 0;
//   initial_data->gammaSphorCartDD00 = tmp4;
//   initial_data->gammaSphorCartDD01 = 0;
//   initial_data->gammaSphorCartDD02 = 0;
//   initial_data->gammaSphorCartDD11 = tmp4;
//   initial_data->gammaSphorCartDD12 = 0;
//   initial_data->gammaSphorCartDD22 = tmp4;
}
