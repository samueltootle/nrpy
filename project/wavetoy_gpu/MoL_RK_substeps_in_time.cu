#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "BHaH_gpu_defines.h"

__global__
void rk_substep(commondata_struct *restrict commondata, 
                params_struct *restrict params,
                MoL_gridfunctions_struct *restrict gridfuncs,
                REAL rk_weight,
                REAL dt_step_factor) {
    // Set gridfunction aliases from gridfuncs struct
    // y_n gridfunctions
    REAL *restrict y_n_gfs = gridfuncs->y_n_gfs;
    
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = gridfuncs->y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = gridfuncs->k_odd_gfs;
    REAL *restrict k_even_gfs = gridfuncs->k_even_gfs;
    REAL *restrict auxevol_gfs = gridfuncs->auxevol_gfs;
    REAL const dt = commondata->dt;

    const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int N = Nxx_plus_2NGHOSTS0 \ 
                * Nxx_plus_2NGHOSTS1 \
                * Nxx_plus_2NGHOSTS2 \
                * NUM_EVOL_GFS;
    for(int i=index;i<N;i+=stride) {
        const REAL k_odd_gfsL = k_odd_gfs[i];
        const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
        const REAL y_n_gfsL = y_n_gfs[i];
        y_nplus1_running_total_gfs[i] =     \
            rk_weight * dt * k_odd_gfsL     \
            + y_nplus1_running_total_gfsL;
        k_odd_gfs[i] = dt_step_factor * dt * k_odd_gfsL + y_n_gfsL;
    }
}