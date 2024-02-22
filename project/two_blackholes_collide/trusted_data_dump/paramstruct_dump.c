#include<stdio.h>
#include "../BHaH_defines.h"
#include "trusted_data_dump_prototypes.h"
void dump_param_struct(const int grid, const params_struct *restrict params, const char* suffix){
  char fname[100];
  sprintf(fname, "trusted_data_dump/dumps/params_grid_%d_%s.txt", grid, suffix);
  printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "w");
  fprintf(fp, "%+1.15f\n", params->Cart_originx);
  fprintf(fp, "%+1.15f\n", params->Cart_originy);
  fprintf(fp, "%+1.15f\n", params->Cart_originz);
  fprintf(fp, "%+1.15f\n", params->RMAX);
  fprintf(fp, "%+1.15f\n", params->dxx0);
  fprintf(fp, "%+1.15f\n", params->dxx1);
  fprintf(fp, "%+1.15f\n", params->dxx2);
  fprintf(fp, "%+1.15f\n", params->f0_of_xx0);
  fprintf(fp, "%+1.15f\n", params->f0_of_xx0__D0);
  fprintf(fp, "%+1.15f\n", params->f0_of_xx0__DD00);
  fprintf(fp, "%+1.15f\n", params->f0_of_xx0__DDD000);
  fprintf(fp, "%+1.15f\n", params->f1_of_xx1);
  fprintf(fp, "%+1.15f\n", params->f1_of_xx1__D1);
  fprintf(fp, "%+1.15f\n", params->f1_of_xx1__DD11);
  fprintf(fp, "%+1.15f\n", params->f1_of_xx1__DDD111);
  fprintf(fp, "%+1.15f\n", params->f2_of_xx0);
  fprintf(fp, "%+1.15f\n", params->f2_of_xx0__D0);
  fprintf(fp, "%+1.15f\n", params->f2_of_xx0__DD00);
  fprintf(fp, "%+1.15f\n", params->f3_of_xx2);
  fprintf(fp, "%+1.15f\n", params->f3_of_xx2__D2);
  fprintf(fp, "%+1.15f\n", params->f3_of_xx2__DD22);
  fprintf(fp, "%+1.15f\n", params->f4_of_xx1);
  fprintf(fp, "%+1.15f\n", params->f4_of_xx1__D1);
  fprintf(fp, "%+1.15f\n", params->f4_of_xx1__DD11);
  fprintf(fp, "%+1.15f\n", params->f4_of_xx1__DDD111);
  fprintf(fp, "%+1.15f\n", params->grid_physical_size);
  fprintf(fp, "%+1.15f\n", params->invdxx0);
  fprintf(fp, "%+1.15f\n", params->invdxx1);
  fprintf(fp, "%+1.15f\n", params->invdxx2);
  fprintf(fp, "%+1.15f\n", params->xxmax0);
  fprintf(fp, "%+1.15f\n", params->xxmax1);
  fprintf(fp, "%+1.15f\n", params->xxmax2);
  fprintf(fp, "%+1.15f\n", params->xxmin0);
  fprintf(fp, "%+1.15f\n", params->xxmin1);
  fprintf(fp, "%+1.15f\n", params->xxmin2);
  fprintf(fp, "%+d\n", params->CoordSystem_hash);
  fprintf(fp, "%+d\n", params->Nxx0);
  fprintf(fp, "%+d\n", params->Nxx1);
  fprintf(fp, "%+d\n", params->Nxx2);
  fprintf(fp, "%+d\n", params->Nxx_plus_2NGHOSTS0);
  fprintf(fp, "%+d\n", params->Nxx_plus_2NGHOSTS1);
  fprintf(fp, "%+d\n", params->Nxx_plus_2NGHOSTS2);
  fprintf(fp, "%s\n", params-> CoordSystemName);
  fclose(fp);
}