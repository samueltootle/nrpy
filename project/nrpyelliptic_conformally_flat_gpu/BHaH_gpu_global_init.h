// BHaH core header file, automatically generated from cuda.output_BHaH_defines_h,
//    DO NOT EDIT THIS FILE BY HAND.

// Initialize streams
for (int i = 0; i < nstreams; ++i) {
  cudaStreamCreate(&streams[i]);
}
// Copy parity array to device __constant__ memory
cudaMemcpyToSymbol(d_evol_gf_parity, evol_gf_parity, 24 * sizeof(int8_t));
cudaCheckErrors(copy, "Copy to d_evol_gf_parity failed");

// Copy gridfunctions_wavespeed array to device memory
cudaMemcpyToSymbol(d_gridfunctions_wavespeed, gridfunctions_wavespeed, NUM_EVOL_GFS * sizeof(REAL));
cudaCheckErrors(copy, "Copy to d_gridfunctions_wavespeed failed");

// Copy gridfunctions_f_infinity array to device memory
cudaMemcpyToSymbol(d_gridfunctions_f_infinity, gridfunctions_f_infinity, NUM_EVOL_GFS * sizeof(REAL));
cudaCheckErrors(copy, "Copy to d_gridfunctions_f_infinity failed");
