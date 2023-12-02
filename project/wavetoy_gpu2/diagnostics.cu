#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"
#include "BHaH_gpu_function_prototypes.h"
/*
 * Diagnostics.
 */
__global__
void diagnostics_gpu(const commondata_struct commondata, 
                     const params_struct *restrict params, 
                     const REAL *restrict xx0,
                     const REAL *restrict xx1,
                     const REAL *restrict xx2,
                     const REAL *restrict y_n_gfs,
                     REAL* exact_soln_at_center_UUGF, 
                     REAL* exact_soln_at_center_VVGF,
                     REAL* num_soln_at_center_UUGF, 
                     REAL* num_soln_at_center_VVGF) {
  int const & Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  
  const int i0_center = params->Nxx_plus_2NGHOSTS0 / 2;
  const int i1_center = params->Nxx_plus_2NGHOSTS1 / 2;
  const int i2_center = params->Nxx_plus_2NGHOSTS2 / 2;
  
  const int center_of_grid_idx = IDX3(i0_center, i1_center, i2_center);

  *num_soln_at_center_UUGF = y_n_gfs[IDX4pt(UUGF, center_of_grid_idx)];
  *num_soln_at_center_VVGF = y_n_gfs[IDX4pt(VVGF, center_of_grid_idx)];

  exact_solution_single_Cartesian_point_gpu(&commondata, 
                                         params, 
                                         xx0[i0_center], 
                                         xx1[i1_center], 
                                         xx2[i2_center], 
                                         exact_soln_at_center_UUGF,
                                         exact_soln_at_center_VVGF);
}

void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  const REAL currtime = commondata->time, currdt = commondata->dt, outevery = commondata->diagnostics_output_every;

  REAL* d_exact_soln_at_center_UUGF, *d_exact_soln_at_center_VVGF;
  REAL* d_num_soln_at_center_UUGF, *d_num_soln_at_center_VVGF;
  cudaMalloc(&d_exact_soln_at_center_UUGF, sizeof(REAL));
  cudaMalloc(&d_exact_soln_at_center_VVGF, sizeof(REAL));
  cudaMalloc(&d_num_soln_at_center_UUGF, sizeof(REAL));
  cudaMalloc(&d_num_soln_at_center_VVGF, sizeof(REAL));
  // Explanation of the if() below:
  // Step 1: round(currtime / outevery) rounds to the nearest integer multiple of currtime.
  // Step 2: Multiplying by outevery yields the exact time we should output again, t_out.
  // Step 3: If fabs(t_out - currtime) < 0.5 * currdt, then currtime is as close to t_out as possible!
  if (fabs(round(currtime / outevery) * outevery - currtime) < 0.5 * currdt) {
    for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
      REAL const convergence_factor = commondata->convergence_factor;
      int const nn = commondata->nn;
      REAL time = commondata->time;

      // 0D output
      {
        char filename[256];
        sprintf(filename, "out0d-conv_factor%.2f.txt", convergence_factor);
        FILE *outfile;
        if (nn == 0)
          outfile = fopen(filename, "w");
        else
          outfile = fopen(filename, "a");
        if (outfile == NULL) {
          fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
          exit(1);
        }        

        REAL num_soln_at_center_UUGF, num_soln_at_center_VVGF;
        REAL exact_soln_at_center_UUGF, exact_soln_at_center_VVGF;
        diagnostics_gpu<<<1,1>>>(*commondata, 
                     griddata[grid].params, 
                     griddata[grid].xx[0],
                     griddata[grid].xx[1],
                     griddata[grid].xx[2],
                     griddata[grid].gridfuncs.y_n_gfs,
                     d_exact_soln_at_center_UUGF, 
                     d_exact_soln_at_center_VVGF,
                     d_num_soln_at_center_UUGF, 
                     d_num_soln_at_center_VVGF);
        cudaMemcpy(&num_soln_at_center_UUGF, d_num_soln_at_center_UUGF, sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(&num_soln_at_center_VVGF, d_num_soln_at_center_VVGF, sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(&exact_soln_at_center_UUGF, d_exact_soln_at_center_UUGF, sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(&exact_soln_at_center_VVGF, d_exact_soln_at_center_VVGF, sizeof(REAL), cudaMemcpyDeviceToHost);
        
        fprintf(outfile, "%e %e %e %1.15e %1.15e\n", time, fabs(fabs(num_soln_at_center_UUGF - exact_soln_at_center_UUGF) / exact_soln_at_center_UUGF),
                fabs(fabs(num_soln_at_center_VVGF - exact_soln_at_center_VVGF) / (1e-16 + exact_soln_at_center_VVGF)), num_soln_at_center_UUGF,
                exact_soln_at_center_UUGF);

        fclose(outfile);
      }
    }
  }
  progress_indicator(commondata, griddata);
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
}
