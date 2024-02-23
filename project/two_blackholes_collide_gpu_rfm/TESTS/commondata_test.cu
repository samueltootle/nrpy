#include<stdio.h>
#include<string>
#include "../BHaH_defines.h"
#ifdef GPU_TESTS
#include "TEST_prototypes.h"
void TEST_commondata(const commondata_struct *restrict commondata, const char* suffix) {
  char fname[100];
  sprintf(fname, "TESTS/dumps/commondata_%s.txt", suffix);
  // printf("TEST: %s\n", fname);
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
  commondata_struct ref;
  read_data(ref.BH1_mass);
  read_data(ref.BH1_posn_x);
  read_data(ref.BH1_posn_y);
  read_data(ref.BH1_posn_z);
  read_data(ref.BH2_mass);
  read_data(ref.BH2_posn_x);
  read_data(ref.BH2_posn_y);
  read_data(ref.BH2_posn_z);
  read_data(ref.CFL_FACTOR);
  read_data(ref.convergence_factor);
  read_data(ref.diagnostics_output_every);
  read_data(ref.dt);
  read_data(ref.eta);
  read_data(ref.t_0);
  read_data(ref.t_final);
  read_data(ref.time);

  auto read_int =[fp](auto& val) {
    char buf[4];
    size_t pos{};
    fgets(buf, sizeof buf, fp);
    val = std::stoi(buf, &pos);
  };  
  read_int(ref.NUMGRIDS);
  read_int(ref.nn);
  read_int(ref.nn_0);
  fclose(fp);

  char fname2[100] = "TESTS/reldiff/commondata_";
  sprintf(fname2, "TESTS/reldiff/commondata_%s.txt", suffix);
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
  fprint_reldiff(ref.BH1_mass, commondata->BH1_mass);
  fprint_reldiff(ref.BH1_posn_x, commondata->BH1_posn_x);
  fprint_reldiff(ref.BH1_posn_y, commondata->BH1_posn_y);
  fprint_reldiff(ref.BH1_posn_z, commondata->BH1_posn_z);
  fprint_reldiff(ref.BH2_mass, commondata->BH2_mass);
  fprint_reldiff(ref.BH2_posn_x, commondata->BH2_posn_x);
  fprint_reldiff(ref.BH2_posn_y, commondata->BH2_posn_y);
  fprint_reldiff(ref.BH2_posn_z, commondata->BH2_posn_z);
  fprint_reldiff(ref.CFL_FACTOR, commondata->CFL_FACTOR);
  fprint_reldiff(ref.convergence_factor, commondata->convergence_factor);
  fprint_reldiff(ref.diagnostics_output_every, commondata->diagnostics_output_every);
  printf("nn: %+1.15f - %+1.15f\n", ref.dt, commondata->dt);
  fprint_reldiff(ref.dt, commondata->dt);
  fprint_reldiff(ref.eta, commondata->eta);
  fprint_reldiff(ref.t_0, commondata->t_0);
  fprint_reldiff(ref.t_final, commondata->t_final);
  fprint_reldiff(ref.time, commondata->time);
  fprint_reldiff(ref.NUMGRIDS, commondata->NUMGRIDS);
  fprint_reldiff(ref.nn, commondata->nn);
  printf("nn: %d - %d\n", ref.nn, commondata->nn);
  fprint_reldiff(ref.nn_0, commondata->nn_0);
  printf("nn: %d - %d\n", ref.nn_0, commondata->nn_0);

  fclose(fp);
}
#endif