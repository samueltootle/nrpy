#include<stdio.h>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "trusted_data_dump_prototypes.h"
// PREFIX: e.g. auxvars, evolvars
// SUFFIX: e.g. post_initial, post_rk1
void dump_coord_direction(const int grid, const REAL *restrict xx, const char* dir, const int Nxx) {
  
  char fname[100];
  sprintf(fname, "trusted_data_dump/dumps/coord_dump_%s_grid_%d.txt", dir, grid);
  // printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "w");

  for(int i = 0; i < Nxx; ++i) {
    fprintf(fp, "%+1.15f\n", xx[i]);
  }
  fclose(fp);
}
#endif