#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "unistd.h"

#define FREAD(ptr, size, nmemb, stream)                                                                                                              \
  { const int numitems = fread((ptr), (size), (nmemb), (stream)); }

/*
 * Read a checkpoint file
 */
int read_checkpoint(commondata_struct *restrict commondata, griddata_struct *restrict griddata, griddata_struct *restrict griddata_GPU) {

  char filename[256];
  snprintf(filename, 256, "checkpoint-conv_factor%.2f.dat", commondata->convergence_factor);
  // If the checkpoint doesn't exist then return 0.
  if (access(filename, F_OK) != 0)
    return 0;

  FILE *cp_file = fopen(filename, "r");
  FREAD(commondata, sizeof(commondata_struct), 1, cp_file);
  fprintf(stderr, "cd struct size = %ld time=%e\n", sizeof(commondata_struct), commondata->time);
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {

    FREAD(&griddata[grid].params, sizeof(params_struct), 1, cp_file);

    // Copy params to griddata that is used with the device
    memcpy(&griddata_GPU[grid].params, &griddata[grid].params, sizeof(params_struct));

    int count;
    FREAD(&count, sizeof(int), 1, cp_file);

    int *out_data_indices = (int *)malloc(sizeof(int) * count);
    REAL *compact_out_data = (REAL *)malloc(sizeof(REAL) * NUM_EVOL_GFS * count);

    const int Nxx_plus_2NGHOSTS0 = griddata[grid].params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata[grid].params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata[grid].params.Nxx_plus_2NGHOSTS2;
    const int ntot = griddata[grid].params.Nxx_plus_2NGHOSTS0 * griddata[grid].params.Nxx_plus_2NGHOSTS1 * griddata[grid].params.Nxx_plus_2NGHOSTS2;
    fprintf(stderr, "Reading checkpoint: grid = %d | pts = %d / %d | %d\n", grid, count, ntot, Nxx_plus_2NGHOSTS2);
    FREAD(out_data_indices, sizeof(int), count, cp_file);
    FREAD(compact_out_data, sizeof(REAL), count * NUM_EVOL_GFS, cp_file);

    // Malloc for both GFs
    CUDA__free_host_gfs(&griddata[grid].gridfuncs);
    CUDA__malloc_host_gfs(commondata, &griddata[grid].params, &griddata[grid].gridfuncs);

    MoL_free_memory_y_n_gfs(&griddata_GPU[grid].gridfuncs);
    MoL_malloc_y_n_gfs(commondata, &griddata_GPU[grid].params, &griddata_GPU[grid].gridfuncs);
#pragma omp parallel for
    for (int i = 0; i < count; i++) {
      for (int gf = 0; gf < NUM_EVOL_GFS; gf++) {
        griddata[grid].gridfuncs.y_n_gfs[IDX4pt(gf, out_data_indices[i])] = compact_out_data[i * NUM_EVOL_GFS + gf];
      }
    }
    free(out_data_indices);
    free(compact_out_data);

    // Set gridfunctions aliases for HOST data
    REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;

    // Set gridfunctions aliases for GPU data
    REAL *restrict y_n_gfs_GPU = griddata_GPU[grid].gridfuncs.y_n_gfs;
    for (int gf = 0; gf < NUM_EVOL_GFS; ++gf) {
      cpyHosttoDevice__gf(commondata, &griddata[grid].params, y_n_gfs, y_n_gfs_GPU, gf, gf);
    }
  }

  fclose(cp_file);
  fprintf(stderr, "FINISHED WITH READING\n");

  // Next set t_0 and n_0
  commondata->t_0 = commondata->time;
  commondata->nn_0 = commondata->nn;

  // local stream syncs?
  cudaDeviceSynchronize();

  return 1;
}
