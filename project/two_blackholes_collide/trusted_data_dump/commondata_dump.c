#include<stdio.h>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "trusted_data_dump_prototypes.h"
void dump_common_data(const commondata_struct *restrict commondata, const char* suffix) {
  char fname[100] = "trusted_data_dump/dumps/commondata_";
  sprintf(fname, "trusted_data_dump/dumps/commondata_%s.txt", suffix);
  // printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "w");
  fprintf(fp, "%+1.15f\n", commondata->BH1_mass);
  fprintf(fp, "%+1.15f\n", commondata->BH1_posn_x);
  fprintf(fp, "%+1.15f\n", commondata->BH1_posn_y);
  fprintf(fp, "%+1.15f\n", commondata->BH1_posn_z);
  fprintf(fp, "%+1.15f\n", commondata->BH2_mass);
  fprintf(fp, "%+1.15f\n", commondata->BH2_posn_x);
  fprintf(fp, "%+1.15f\n", commondata->BH2_posn_y);
  fprintf(fp, "%+1.15f\n", commondata->BH2_posn_z);
  fprintf(fp, "%+1.15f\n", commondata->CFL_FACTOR);
  fprintf(fp, "%+1.15f\n", commondata->convergence_factor);
  fprintf(fp, "%+1.15f\n", commondata->diagnostics_output_every);
  fprintf(fp, "%+1.15f\n", commondata->dt);
  fprintf(fp, "%+1.15f\n", commondata->eta);
  fprintf(fp, "%+1.15f\n", commondata->t_0);
  fprintf(fp, "%+1.15f\n", commondata->t_final);
  fprintf(fp, "%+1.15f\n", commondata->time);
  fprintf(fp, "%+d\n", commondata->NUMGRIDS);
  fprintf(fp, "%+d\n", commondata->nn);
  fprintf(fp, "%+d\n", commondata->nn_0);
  fprintf(fp, "%s\n", commondata->outer_bc_type);
  fclose(fp);
}
#endif