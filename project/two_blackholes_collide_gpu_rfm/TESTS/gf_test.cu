#include<stdio.h>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "TEST_prototypes.h"
// PREFIX: e.g. auxvars, evolvars
// SUFFIX: e.g. post_initial, post_rk1
void dump_gf_array(const int grid,const params_struct *restrict params, const REAL *restrict gfs, 
  const char* prefix, const char* suffix, const int numgfs) {
  
  char fname[100];
  sprintf(fname, "TEST/dumps/gfs_%s_grid_%d_%s.txt", prefix, grid, suffix);
  // printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "w");
  
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int N = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  for(int gf = 0; gf < numgfs; ++gf) {
    for(int i = 0; i < N; ++i) {
      int idx = IDX4pt(gf, i);
      fprintf(fp, "%+1.15f\n", gfs[idx]);
    }
  }
  fclose(fp);
}
#endif