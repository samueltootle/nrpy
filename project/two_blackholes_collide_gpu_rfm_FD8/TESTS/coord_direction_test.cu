#include<stdio.h>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "TEST_prototypes.h"

__global__
void compute_rel_diff_gpu(const REAL *restrict ref, const REAL *restrict gpu, REAL *restrict result, const int N) {
    for(int i = 0; i < N; ++i) {
        result[i] = std::fabs(1. - gpu[i] / ref[i]);
    }
}

__host__
void compute_rel_diff(const REAL *restrict ref, const REAL *restrict gpu, REAL *restrict result, const int N) {
  REAL* ref_gpu;
  cudaMalloc(&ref_gpu, N * sizeof(REAL));
  cudaMemcpy(ref_gpu, ref, N * sizeof(REAL), cudaMemcpyHostToDevice);
  REAL* reldiff_gpu;
  cudaMalloc(&reldiff_gpu, N * sizeof(REAL));

  compute_rel_diff_gpu<<<1,1>>>(ref_gpu, gpu, reldiff_gpu, N);
  cudaMemcpy(result, reldiff_gpu, N * sizeof(REAL), cudaMemcpyDeviceToHost);
}

__host__
void TEST_coord_direction(const int grid, const REAL *restrict xx, const char* dir, const int Nxx) {
  char fname[100];
  sprintf(fname, "TESTS/dumps/coord_dump_%s_grid_%d.txt", dir, grid);
//   printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "r");
  if (!fp) {
    perror("Trusted data file opening failed\n");
  }

  REAL* ref = (REAL*)malloc(Nxx * sizeof(REAL));

  auto read_data =[fp](auto& val) {
    char buf[20];
    char* end{};
    fgets(buf, sizeof buf, fp);
    val = std::strtod(buf, &end);
  };
  for(int i = 0; i < Nxx; ++i) {
    read_data(ref[i]);
    // printf("%1.15f\n", ref[i]);
  }
  fclose(fp);

  REAL* reldiff = (REAL*)malloc(Nxx * sizeof(REAL));
  compute_rel_diff(ref, xx, reldiff, Nxx);

  char fname2[100];
  sprintf(fname2, "TESTS/reldiff/coord_dump_%s_grid_%d.txt", dir, grid);
  fp = fopen(fname2, "w");
    if (!fp) {
    perror("Output file opening failed\n");
  }
  for(int i = 0; i < Nxx; ++i) {
    fprintf(fp, "%+1.15f\n", reldiff[i]);
  }
}
#endif