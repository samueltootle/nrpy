#include "../BHaH_defines.h"
/**
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0_host.
 * Host Kernel to precompute metric quantity f0_of_xx0.
 */
static void rfm_precompute_defines__f0_of_xx0_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct, const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMPL = params->AMPL;
  const REAL SINHW = params->SINHW;
  const REAL r_slope = params->r_slope;
  for (int i0 = 0; i0 < params->Nxx_plus_2NGHOSTS0; i0++) {
    const REAL xx0 = x0[i0];
    rfmstruct->f0_of_xx0[i0] =
        r_slope * xx0 + powf(xx0, 2) * (AMPL - r_slope) * (expf(xx0 / SINHW) - expf(-xx0 / SINHW)) / (expf(1.0F / SINHW) - expf(-1 / SINHW));
  }
} // END FUNCTION rfm_precompute_defines__f0_of_xx0_host
/**
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__D0_host.
 * Host Kernel to precompute metric quantity f0_of_xx0__D0.
 */
static void rfm_precompute_defines__f0_of_xx0__D0_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                                                       const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMPL = params->AMPL;
  const REAL SINHW = params->SINHW;
  const REAL r_slope = params->r_slope;
  for (int i0 = 0; i0 < params->Nxx_plus_2NGHOSTS0; i0++) {
    const REAL xx0 = x0[i0];
    rfmstruct->f0_of_xx0__D0[i0] =
        r_slope +
        powf(xx0, 2) * (AMPL - r_slope) * (expf(xx0 / SINHW) / SINHW + expf(-xx0 / SINHW) / SINHW) / (expf(1.0F / SINHW) - expf(-1 / SINHW)) +
        2 * xx0 * (AMPL - r_slope) * (expf(xx0 / SINHW) - expf(-xx0 / SINHW)) / (expf(1.0F / SINHW) - expf(-1 / SINHW));
  }
} // END FUNCTION rfm_precompute_defines__f0_of_xx0__D0_host
/**
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__DD00_host.
 * Host Kernel to precompute metric quantity f0_of_xx0__DD00.
 */
static void rfm_precompute_defines__f0_of_xx0__DD00_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                                                         const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMPL = params->AMPL;
  const REAL SINHW = params->SINHW;
  const REAL r_slope = params->r_slope;
  for (int i0 = 0; i0 < params->Nxx_plus_2NGHOSTS0; i0++) {
    const REAL xx0 = x0[i0];
    rfmstruct->f0_of_xx0__DD00[i0] =
        powf(xx0, 2) * (AMPL - r_slope) * (expf(xx0 / SINHW) / powf(SINHW, 2) - expf(-xx0 / SINHW) / powf(SINHW, 2)) /
            (expf(1.0F / SINHW) - expf(-1 / SINHW)) +
        4 * xx0 * (AMPL - r_slope) * (expf(xx0 / SINHW) / SINHW + expf(-xx0 / SINHW) / SINHW) / (expf(1.0F / SINHW) - expf(-1 / SINHW)) +
        2 * (AMPL - r_slope) * (expf(xx0 / SINHW) - expf(-xx0 / SINHW)) / (expf(1.0F / SINHW) - expf(-1 / SINHW));
  }
} // END FUNCTION rfm_precompute_defines__f0_of_xx0__DD00_host
/**
 * GPU Kernel: rfm_precompute_defines__f0_of_xx0__DDD000_host.
 * Host Kernel to precompute metric quantity f0_of_xx0__DDD000.
 */
static void rfm_precompute_defines__f0_of_xx0__DDD000_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                                                           const REAL *restrict x0) {
  // Temporary parameters
  const REAL AMPL = params->AMPL;
  const REAL SINHW = params->SINHW;
  const REAL r_slope = params->r_slope;
  for (int i0 = 0; i0 < params->Nxx_plus_2NGHOSTS0; i0++) {
    const REAL xx0 = x0[i0];
    rfmstruct->f0_of_xx0__DDD000[i0] =
        powf(xx0, 2) * (AMPL - r_slope) * (expf(xx0 / SINHW) / powf(SINHW, 3) + expf(-xx0 / SINHW) / powf(SINHW, 3)) /
            (expf(1.0F / SINHW) - expf(-1 / SINHW)) +
        6 * xx0 * (AMPL - r_slope) * (expf(xx0 / SINHW) / powf(SINHW, 2) - expf(-xx0 / SINHW) / powf(SINHW, 2)) /
            (expf(1.0F / SINHW) - expf(-1 / SINHW)) +
        6 * (AMPL - r_slope) * (expf(xx0 / SINHW) / SINHW + expf(-xx0 / SINHW) / SINHW) / (expf(1.0F / SINHW) - expf(-1 / SINHW));
  }
} // END FUNCTION rfm_precompute_defines__f0_of_xx0__DDD000_host
/**
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1_host.
 * Host Kernel to precompute metric quantity f1_of_xx1.
 */
static void rfm_precompute_defines__f1_of_xx1_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct, const REAL *restrict x1) {
  // Temporary parameters
  for (int i1 = 0; i1 < params->Nxx_plus_2NGHOSTS1; i1++) {
    const REAL xx1 = x1[i1];
    rfmstruct->f1_of_xx1[i1] = sinf(xx1);
  }
} // END FUNCTION rfm_precompute_defines__f1_of_xx1_host
/**
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1__D1_host.
 * Host Kernel to precompute metric quantity f1_of_xx1__D1.
 */
static void rfm_precompute_defines__f1_of_xx1__D1_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                                                       const REAL *restrict x1) {
  // Temporary parameters
  for (int i1 = 0; i1 < params->Nxx_plus_2NGHOSTS1; i1++) {
    const REAL xx1 = x1[i1];
    rfmstruct->f1_of_xx1__D1[i1] = cosf(xx1);
  }
} // END FUNCTION rfm_precompute_defines__f1_of_xx1__D1_host
/**
 * GPU Kernel: rfm_precompute_defines__f1_of_xx1__DD11_host.
 * Host Kernel to precompute metric quantity f1_of_xx1__DD11.
 */
static void rfm_precompute_defines__f1_of_xx1__DD11_host(const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                                                         const REAL *restrict x1) {
  // Temporary parameters
  for (int i1 = 0; i1 < params->Nxx_plus_2NGHOSTS1; i1++) {
    const REAL xx1 = x1[i1];
    rfmstruct->f1_of_xx1__DD11[i1] = -sinf(xx1);
  }
} // END FUNCTION rfm_precompute_defines__f1_of_xx1__DD11_host

/**
 * rfm_precompute_defines: reference metric precomputed lookup arrays: defines
 */
void rfm_precompute_defines__rfm__SinhSphericalv2n2(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                    rfm_struct *restrict rfmstruct, REAL *restrict xx[3]) {
  MAYBE_UNUSED const REAL *restrict x0 = xx[0];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  MAYBE_UNUSED const REAL *restrict x1 = xx[1];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  MAYBE_UNUSED const REAL *restrict x2 = xx[2];
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  {
    rfm_precompute_defines__f0_of_xx0_host(params, rfmstruct, x0);
  }
  {
    rfm_precompute_defines__f0_of_xx0__D0_host(params, rfmstruct, x0);
  }
  {
    rfm_precompute_defines__f0_of_xx0__DD00_host(params, rfmstruct, x0);
  }
  {
    rfm_precompute_defines__f0_of_xx0__DDD000_host(params, rfmstruct, x0);
  }
  {
    rfm_precompute_defines__f1_of_xx1_host(params, rfmstruct, x1);
  }
  {
    rfm_precompute_defines__f1_of_xx1__D1_host(params, rfmstruct, x1);
  }
  {
    rfm_precompute_defines__f1_of_xx1__DD11_host(params, rfmstruct, x1);
  }
} // END FUNCTION rfm_precompute_defines__rfm__SinhSphericalv2n2
