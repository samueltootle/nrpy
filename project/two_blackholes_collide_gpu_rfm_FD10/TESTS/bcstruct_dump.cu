#include<stdio.h>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "TEST_prototypes.h"
void TEST_bcstruct(const bc_struct *restrict bcstruct, const char* suffix) {
  char fname[100];
  sprintf(fname, "TESTS/dumps/bcstruct_%s.txt", suffix);
  // printf("TEST bcstruct: %s\n", fname);
  FILE* fp = fopen(fname, "r");
  if (!fp) {
    perror("Trusted data file opening failed\n");
  }

  fprintf(fp, "%+d\n", bcstruct->bc_info.num_inner_boundary_points);
  for(int i = 0; i < NGHOSTS; ++i) {
    for(int j = 0; j < 3; ++j) {
      fprintf(fp, "%+d\n", bcstruct->bc_info.num_pure_outer_boundary_points[i][j]);
    }
  }
  for(int i = 0; i < NGHOSTS; ++i) {
    for(int j = 0; j < 6; ++j) {
      for(int k = 0; k < 6; ++k) {
        fprintf(fp, "%+d\n", bcstruct->bc_info.bc_loop_bounds[i][j][k]);
      }
    }
  }
  
  for(int i = 0; i < bcstruct->bc_info.num_inner_boundary_points; ++i) {
    fprintf(fp, "%+d\n", bcstruct->inner_bc_array[i].dstpt);
    fprintf(fp, "%+d\n", bcstruct->inner_bc_array[i].srcpt);
    for(int j = 0; j < 10; ++j) {
      fprintf(fp, "%+d\n", bcstruct->inner_bc_array[i].parity[j]);
    }
  }

  // for(int i = 0; i < NGHOSTS *3; ++i) {

  // }
  // fprintf(fp, "%+1.15f\n", commondata->BH1_mass);

  // fprintf(fp, "%+d\n", commondata->NUMGRIDS);
  fclose(fp);
}
#endif