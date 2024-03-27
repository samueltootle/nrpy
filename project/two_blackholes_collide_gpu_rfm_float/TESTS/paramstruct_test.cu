#include<stdio.h>
#include<string>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "TEST_prototypes.h"
void TEST_param_struct(const int grid, const params_struct *restrict params, const char* suffix){
  char fname[100];
  sprintf(fname, "TESTS/dumps/params_grid_%d_%s.txt", grid, suffix);
  printf("TEST: %s\n", fname);
  FILE* fp = fopen(fname, "r");
  if (!fp) {
    perror("Trusted data file opening failed\n");
  }
  
  auto read_data =[fp](auto& val) {
    char buf[20];
    char* end{};
    fgets(buf, sizeof buf, fp);
    val = std::strtod(buf, &end);
  };
  params_struct ref;
  read_data(ref.Cart_originx);
  read_data(ref.Cart_originy);
  read_data(ref.Cart_originz);
  read_data(ref.RMAX);
  read_data(ref.dxx0);
  read_data(ref.dxx1);
  read_data(ref.dxx2);
  read_data(ref.grid_physical_size);
  read_data(ref.invdxx0);
  read_data(ref.invdxx1);
  read_data(ref.invdxx2);
  read_data(ref.xxmax0);
  read_data(ref.xxmax1);
  read_data(ref.xxmax2);
  read_data(ref.xxmin0);
  read_data(ref.xxmin1);
  read_data(ref.xxmin2);

  {
    char buf[16];
    size_t pos{};
    fgets(buf, sizeof buf, fp);
    ref.CoordSystem_hash = std::stoi(buf, &pos);
  }
  auto read_int =[fp](auto& val) {
    char buf[5];
    size_t pos{};
    fgets(buf, sizeof buf, fp);
    val = std::stoi(buf, &pos);
  };
  read_int(ref.Nxx0);
  read_int(ref.Nxx1);
  read_int(ref.Nxx2);
  read_int(ref.Nxx_plus_2NGHOSTS0);
  read_int(ref.Nxx_plus_2NGHOSTS1);
  read_int(ref.Nxx_plus_2NGHOSTS2);
  fclose(fp);
  
  char fname2[100];
  sprintf(fname2, "TESTS/reldiff/params_grid_%d_%s.txt", grid, suffix);
  fp = fopen(fname2, "w");
    if (!fp) {
    perror("Output file opening failed\n");
  }

  auto fprint_reldiff = [fp](auto const & a, auto const & b) {
    double reldiff;
    if(std::fabs(a) < 1e-15 && std::fabs(b) < 1e-15)
      reldiff = 0;
    else
      reldiff = std::fabs(1. - (double)a / (double)b);
    fprintf(fp, "%+1.15f\n", reldiff);
  };
  fprint_reldiff(ref.Cart_originx, params->Cart_originx);
  fprint_reldiff(ref.Cart_originy, params->Cart_originy);
  fprint_reldiff(ref.Cart_originz, params->Cart_originz);
  fprint_reldiff(ref.RMAX, params->RMAX);
  fprint_reldiff(ref.dxx0, params->dxx0);
  fprint_reldiff(ref.dxx1, params->dxx1);
  fprint_reldiff(ref.dxx2, params->dxx2);
  fprint_reldiff(ref.grid_physical_size, params->grid_physical_size);
  fprint_reldiff(ref.invdxx0, params->invdxx0);
  fprint_reldiff(ref.invdxx1, params->invdxx1);
  fprint_reldiff(ref.invdxx2, params->invdxx2);
  fprint_reldiff(ref.xxmax0, params->xxmax0);
  fprint_reldiff(ref.xxmax1, params->xxmax1);
  fprint_reldiff(ref.xxmax2, params->xxmax2);
  fprint_reldiff(ref.xxmin0, params->xxmin0);
  fprint_reldiff(ref.xxmin1, params->xxmin1);
  fprint_reldiff(ref.xxmin2, params->xxmin2);
  fprint_reldiff(ref.Nxx0, params->Nxx0);
  fprint_reldiff(ref.Nxx1, params->Nxx1);
  fprint_reldiff(ref.Nxx2, params->Nxx2);
  fprint_reldiff(ref.Nxx_plus_2NGHOSTS0, params->Nxx_plus_2NGHOSTS0);
  fprint_reldiff(ref.Nxx_plus_2NGHOSTS1, params->Nxx_plus_2NGHOSTS1);
  fprint_reldiff(ref.Nxx_plus_2NGHOSTS2, params->Nxx_plus_2NGHOSTS2);
  fclose(fp);
}
#endif