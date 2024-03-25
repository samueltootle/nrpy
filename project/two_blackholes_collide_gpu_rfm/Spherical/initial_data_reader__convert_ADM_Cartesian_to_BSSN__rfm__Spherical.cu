#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"

__device__ ID_pfunc id_ptr = BrillLindquist;

// ADM variables in the Cartesian basis:
typedef struct __ADM_Cart_basis_struct__ {
  REAL alpha, betaU0, betaU1, betaU2, BU0, BU1, BU2;
  REAL gammaDD00, gammaDD01, gammaDD02, gammaDD11, gammaDD12, gammaDD22;
  REAL KDD00, KDD01, KDD02, KDD11, KDD12, KDD22;
} ADM_Cart_basis_struct;

// BSSN variables in the Cartesian basis:
typedef struct __BSSN_Cart_basis_struct__ {
  REAL alpha, betaU0, betaU1, betaU2, BU0, BU1, BU2;
  REAL cf, trK;
  REAL gammabarDD00, gammabarDD01, gammabarDD02, gammabarDD11, gammabarDD12, gammabarDD22;
  REAL AbarDD00, AbarDD01, AbarDD02, AbarDD11, AbarDD12, AbarDD22;
} BSSN_Cart_basis_struct;

// Rescaled BSSN variables in the rfm basis:
typedef struct __rescaled_BSSN_rfm_basis_struct__ {
  REAL alpha, vetU0, vetU1, vetU2, betU0, betU1, betU2;
  REAL cf, trK;
  REAL hDD00, hDD01, hDD02, hDD11, hDD12, hDD22;
  REAL aDD00, aDD01, aDD02, aDD11, aDD12, aDD22;
} rescaled_BSSN_rfm_basis_struct;
/*
 * Convert ADM variables from the spherical or Cartesian basis to the Cartesian basis
 */
__device__
void ADM_SphorCart_to_Cart(const commondata_struct *restrict commondata, const REAL xCart[3],
                                  const initial_data_struct *restrict initial_data, ADM_Cart_basis_struct *restrict ADM_Cart_basis) {

  // Unpack initial_data for ADM vectors/tensors
  const REAL betaSphorCartU0 = initial_data->betaSphorCartU0;
  const REAL betaSphorCartU1 = initial_data->betaSphorCartU1;
  const REAL betaSphorCartU2 = initial_data->betaSphorCartU2;

  const REAL BSphorCartU0 = initial_data->BSphorCartU0;
  const REAL BSphorCartU1 = initial_data->BSphorCartU1;
  const REAL BSphorCartU2 = initial_data->BSphorCartU2;

  const REAL gammaSphorCartDD00 = initial_data->gammaSphorCartDD00;
  const REAL gammaSphorCartDD01 = initial_data->gammaSphorCartDD01;
  const REAL gammaSphorCartDD02 = initial_data->gammaSphorCartDD02;
  const REAL gammaSphorCartDD11 = initial_data->gammaSphorCartDD11;
  const REAL gammaSphorCartDD12 = initial_data->gammaSphorCartDD12;
  const REAL gammaSphorCartDD22 = initial_data->gammaSphorCartDD22;

  const REAL KSphorCartDD00 = initial_data->KSphorCartDD00;
  const REAL KSphorCartDD01 = initial_data->KSphorCartDD01;
  const REAL KSphorCartDD02 = initial_data->KSphorCartDD02;
  const REAL KSphorCartDD11 = initial_data->KSphorCartDD11;
  const REAL KSphorCartDD12 = initial_data->KSphorCartDD12;
  const REAL KSphorCartDD22 = initial_data->KSphorCartDD22;

  ADM_Cart_basis->BU0 = BSphorCartU0;
  ADM_Cart_basis->BU1 = BSphorCartU1;
  ADM_Cart_basis->BU2 = BSphorCartU2;
  ADM_Cart_basis->KDD00 = KSphorCartDD00;
  ADM_Cart_basis->KDD01 = KSphorCartDD01;
  ADM_Cart_basis->KDD02 = KSphorCartDD02;
  ADM_Cart_basis->KDD11 = KSphorCartDD11;
  ADM_Cart_basis->KDD12 = KSphorCartDD12;
  ADM_Cart_basis->KDD22 = KSphorCartDD22;
  ADM_Cart_basis->alpha = initial_data->alpha;
  ADM_Cart_basis->betaU0 = betaSphorCartU0;
  ADM_Cart_basis->betaU1 = betaSphorCartU1;
  ADM_Cart_basis->betaU2 = betaSphorCartU2;
  ADM_Cart_basis->gammaDD00 = gammaSphorCartDD00;
  ADM_Cart_basis->gammaDD01 = gammaSphorCartDD01;
  ADM_Cart_basis->gammaDD02 = gammaSphorCartDD02;
  ADM_Cart_basis->gammaDD11 = gammaSphorCartDD11;
  ADM_Cart_basis->gammaDD12 = gammaSphorCartDD12;
  ADM_Cart_basis->gammaDD22 = gammaSphorCartDD22;
}
/*
 * Convert ADM variables in the Cartesian basis to BSSN variables in the Cartesian basis
 */
__device__ void ADM_Cart_to_BSSN_Cart(const commondata_struct *restrict commondata, const REAL xCart[3],
                                  const ADM_Cart_basis_struct *restrict ADM_Cart_basis, BSSN_Cart_basis_struct *restrict BSSN_Cart_basis) {

  // *In the Cartesian basis*, convert ADM quantities gammaDD & KDD
  //   into BSSN gammabarDD, AbarDD, cf, and trK.
  BSSN_Cart_basis->alpha = ADM_Cart_basis->alpha;
  BSSN_Cart_basis->betaU0 = ADM_Cart_basis->betaU0;
  BSSN_Cart_basis->betaU1 = ADM_Cart_basis->betaU1;
  BSSN_Cart_basis->betaU2 = ADM_Cart_basis->betaU2;
  BSSN_Cart_basis->BU0 = ADM_Cart_basis->BU0;
  BSSN_Cart_basis->BU1 = ADM_Cart_basis->BU1;
  BSSN_Cart_basis->BU2 = ADM_Cart_basis->BU2;
  const REAL tmp1 = ADM_Cart_basis->gammaDD00 * ((ADM_Cart_basis->gammaDD12) * (ADM_Cart_basis->gammaDD12));
  const REAL tmp3 = ((ADM_Cart_basis->gammaDD01) * (ADM_Cart_basis->gammaDD01)) * ADM_Cart_basis->gammaDD22;
  const REAL tmp5 = ((ADM_Cart_basis->gammaDD02) * (ADM_Cart_basis->gammaDD02)) * ADM_Cart_basis->gammaDD11;
  const REAL tmp6 = ADM_Cart_basis->gammaDD00 * ADM_Cart_basis->gammaDD11 * ADM_Cart_basis->gammaDD22 +
                    2 * ADM_Cart_basis->gammaDD01 * ADM_Cart_basis->gammaDD02 * ADM_Cart_basis->gammaDD12 - tmp1 - tmp3 - tmp5;
  const REAL tmp7 = (1.0 / (tmp6));
  const REAL tmp8 = cbrt(tmp7);
  const REAL tmp9 = 2 * tmp7;
  const REAL tmp10 =
      ADM_Cart_basis->KDD00 * tmp7 *
          (ADM_Cart_basis->gammaDD11 * ADM_Cart_basis->gammaDD22 - ((ADM_Cart_basis->gammaDD12) * (ADM_Cart_basis->gammaDD12))) +
      ADM_Cart_basis->KDD01 * tmp9 *
          (-ADM_Cart_basis->gammaDD01 * ADM_Cart_basis->gammaDD22 + ADM_Cart_basis->gammaDD02 * ADM_Cart_basis->gammaDD12) +
      ADM_Cart_basis->KDD02 * tmp9 * (ADM_Cart_basis->gammaDD01 * ADM_Cart_basis->gammaDD12 - ADM_Cart_basis->gammaDD02 * ADM_Cart_basis->gammaDD11) +
      ADM_Cart_basis->KDD11 * tmp7 *
          (ADM_Cart_basis->gammaDD00 * ADM_Cart_basis->gammaDD22 - ((ADM_Cart_basis->gammaDD02) * (ADM_Cart_basis->gammaDD02))) +
      ADM_Cart_basis->KDD12 * tmp9 *
          (-ADM_Cart_basis->gammaDD00 * ADM_Cart_basis->gammaDD12 + ADM_Cart_basis->gammaDD01 * ADM_Cart_basis->gammaDD02) +
      ADM_Cart_basis->KDD22 * tmp7 *
          (ADM_Cart_basis->gammaDD00 * ADM_Cart_basis->gammaDD11 - ((ADM_Cart_basis->gammaDD01) * (ADM_Cart_basis->gammaDD01)));
  const REAL tmp11 = (1.0 / 3.0) * tmp10;
  BSSN_Cart_basis->AbarDD00 = tmp8 * (ADM_Cart_basis->KDD00 - ADM_Cart_basis->gammaDD00 * tmp11);
  BSSN_Cart_basis->AbarDD01 = tmp8 * (ADM_Cart_basis->KDD01 - ADM_Cart_basis->gammaDD01 * tmp11);
  BSSN_Cart_basis->AbarDD02 = tmp8 * (ADM_Cart_basis->KDD02 - ADM_Cart_basis->gammaDD02 * tmp11);
  BSSN_Cart_basis->AbarDD11 = tmp8 * (ADM_Cart_basis->KDD11 - ADM_Cart_basis->gammaDD11 * tmp11);
  BSSN_Cart_basis->AbarDD12 = tmp8 * (ADM_Cart_basis->KDD12 - ADM_Cart_basis->gammaDD12 * tmp11);
  BSSN_Cart_basis->AbarDD22 = tmp8 * (ADM_Cart_basis->KDD22 - ADM_Cart_basis->gammaDD22 * tmp11);
  BSSN_Cart_basis->cf = pow(tmp6 / (ADM_Cart_basis->gammaDD00 * ADM_Cart_basis->gammaDD11 * ADM_Cart_basis->gammaDD22 * tmp7 +
                                    2 * ADM_Cart_basis->gammaDD01 * ADM_Cart_basis->gammaDD02 * ADM_Cart_basis->gammaDD12 * tmp7 - tmp1 * tmp7 -
                                    tmp3 * tmp7 - tmp5 * tmp7),
                            -1.0 / 6.0);
  BSSN_Cart_basis->gammabarDD00 = ADM_Cart_basis->gammaDD00 * tmp8;
  BSSN_Cart_basis->gammabarDD01 = ADM_Cart_basis->gammaDD01 * tmp8;
  BSSN_Cart_basis->gammabarDD02 = ADM_Cart_basis->gammaDD02 * tmp8;
  BSSN_Cart_basis->gammabarDD11 = ADM_Cart_basis->gammaDD11 * tmp8;
  BSSN_Cart_basis->gammabarDD12 = ADM_Cart_basis->gammaDD12 * tmp8;
  BSSN_Cart_basis->gammabarDD22 = ADM_Cart_basis->gammaDD22 * tmp8;
  BSSN_Cart_basis->trK = tmp10;
}
/*
 * Cartesian -> Spherical basis transformation of BSSN vectors/tensors *except* lambda^i.
 * After the basis transform, all BSSN quantities are rescaled.
 */
__device__ 
void BSSN_Cart_to_rescaled_BSSN_rfm(const commondata_struct *restrict commondata, const REAL xCart[3],
                                           const BSSN_Cart_basis_struct *restrict BSSN_Cart_basis,
                                           rescaled_BSSN_rfm_basis_struct *restrict rescaled_BSSN_rfm_basis) {
// #include "../set_CodeParameters.h"

  REAL xx0, xx1, xx2 __attribute__((unused)); // xx2 might be unused in the case of axisymmetric initial data.
  {
    int unused_Cart_to_i0i1i2[3];
    REAL xx[3];
    Cart_to_xx_and_nearest_i0i1i2(commondata, xCart, xx, unused_Cart_to_i0i1i2);
    xx0 = xx[0];
    xx1 = xx[1];
    xx2 = xx[2];
  }
  const REAL tmp0 = cos(xx1);
  const REAL tmp2 = sin(xx1);
  const REAL tmp4 = cos(xx2);
  const REAL tmp7 = sin(xx2);
  const REAL tmp16 = (1.0 / (xx0));
  const REAL tmp32 = ((xx0) * (xx0));
  const REAL tmp1 = ((tmp0) * (tmp0));
  const REAL tmp3 = tmp0 * tmp2;
  const REAL tmp5 = 2 * tmp4;
  const REAL tmp8 = 2 * tmp7;
  const REAL tmp9 = ((tmp2) * (tmp2));
  const REAL tmp10 = ((tmp4) * (tmp4));
  const REAL tmp12 = ((tmp7) * (tmp7));
  const REAL tmp17 = tmp2 * xx0;
  const REAL tmp19 = tmp4 * xx0;
  const REAL tmp21 = tmp7 * xx0;
  const REAL tmp30 = (1.0 / (tmp2));
  const REAL tmp33 = (1.0 / (tmp32));
  const REAL tmp49 = ((tmp2) * (tmp2) * (tmp2)) * tmp32;
  const REAL tmp11 = tmp10 * tmp9;
  const REAL tmp13 = tmp12 * tmp9;
  const REAL tmp14 = BSSN_Cart_basis->AbarDD01 * tmp5;
  const REAL tmp18 = tmp0 * tmp17;
  const REAL tmp20 = tmp1 * tmp19;
  const REAL tmp22 = tmp1 * tmp21;
  const REAL tmp23 = tmp19 * tmp9;
  const REAL tmp24 = tmp21 * tmp9;
  const REAL tmp27 = tmp21 * tmp3;
  const REAL tmp34 = tmp32 * tmp9;
  const REAL tmp36 = tmp0 * tmp2 * tmp32;
  const REAL tmp38 = tmp1 * tmp32;
  const REAL tmp48 = tmp33 / tmp9;
  const REAL tmp62 = BSSN_Cart_basis->gammabarDD01 * tmp5;
  const REAL tmp28 = tmp13 * xx0;
  const REAL tmp39 = tmp12 * tmp38;
  const REAL tmp41 = tmp34 * tmp4;
  const REAL tmp42 = tmp12 * tmp36;
  const REAL tmp43 = tmp36 * tmp4 * tmp7;
  const REAL tmp47 = tmp34 * tmp7;
  const REAL tmp51 = (1.0 / (tmp10 * tmp2 * tmp38 + tmp10 * tmp49 + tmp12 * tmp49 + tmp2 * tmp39));
  const REAL tmp54 = tmp0 * tmp10 * tmp2 * tmp32 + tmp42;
  const REAL tmp58 = -tmp11 * xx0 - tmp28;
  const REAL tmp56 = tmp19 * tmp3 * tmp51;
  const REAL tmp59 = tmp51 * (tmp20 + tmp23);
  const REAL tmp60 = tmp51 * (-tmp22 - tmp24);
  rescaled_BSSN_rfm_basis->aDD00 = BSSN_Cart_basis->AbarDD00 * tmp11 + BSSN_Cart_basis->AbarDD02 * tmp3 * tmp5 + BSSN_Cart_basis->AbarDD11 * tmp13 +
                                   BSSN_Cart_basis->AbarDD12 * tmp3 * tmp8 + BSSN_Cart_basis->AbarDD22 * tmp1 + tmp14 * tmp7 * tmp9;
  rescaled_BSSN_rfm_basis->aDD01 =
      tmp16 * (BSSN_Cart_basis->AbarDD00 * tmp10 * tmp18 + BSSN_Cart_basis->AbarDD02 * tmp20 - BSSN_Cart_basis->AbarDD02 * tmp23 +
               BSSN_Cart_basis->AbarDD11 * tmp12 * tmp18 + BSSN_Cart_basis->AbarDD12 * tmp22 - BSSN_Cart_basis->AbarDD12 * tmp24 -
               BSSN_Cart_basis->AbarDD22 * tmp18 + tmp14 * tmp27);
  rescaled_BSSN_rfm_basis->aDD02 = tmp16 * tmp30 *
                                   (-BSSN_Cart_basis->AbarDD00 * tmp24 * tmp4 + BSSN_Cart_basis->AbarDD01 * tmp10 * tmp9 * xx0 -
                                    BSSN_Cart_basis->AbarDD01 * tmp28 - BSSN_Cart_basis->AbarDD02 * tmp27 +
                                    BSSN_Cart_basis->AbarDD11 * tmp4 * tmp7 * tmp9 * xx0 + BSSN_Cart_basis->AbarDD12 * tmp0 * tmp2 * tmp4 * xx0);
  rescaled_BSSN_rfm_basis->aDD11 =
      tmp33 * (BSSN_Cart_basis->AbarDD00 * tmp10 * tmp38 - BSSN_Cart_basis->AbarDD02 * tmp36 * tmp5 + BSSN_Cart_basis->AbarDD11 * tmp39 -
               BSSN_Cart_basis->AbarDD12 * tmp36 * tmp8 + BSSN_Cart_basis->AbarDD22 * tmp34 + tmp14 * tmp38 * tmp7);
  rescaled_BSSN_rfm_basis->aDD12 = tmp30 * tmp33 *
                                   (-BSSN_Cart_basis->AbarDD00 * tmp43 + BSSN_Cart_basis->AbarDD01 * tmp0 * tmp10 * tmp2 * tmp32 -
                                    BSSN_Cart_basis->AbarDD01 * tmp42 + BSSN_Cart_basis->AbarDD02 * tmp32 * tmp7 * tmp9 +
                                    BSSN_Cart_basis->AbarDD11 * tmp0 * tmp2 * tmp32 * tmp4 * tmp7 - BSSN_Cart_basis->AbarDD12 * tmp41);
  rescaled_BSSN_rfm_basis->aDD22 = tmp48 * (BSSN_Cart_basis->AbarDD00 * tmp12 * tmp34 + BSSN_Cart_basis->AbarDD11 * tmp10 * tmp34 - tmp14 * tmp47);
  rescaled_BSSN_rfm_basis->alpha = BSSN_Cart_basis->alpha;
  rescaled_BSSN_rfm_basis->betU0 = BSSN_Cart_basis->BU0 * tmp41 * tmp51 + BSSN_Cart_basis->BU1 * tmp47 * tmp51 + BSSN_Cart_basis->BU2 * tmp51 * tmp54;
  rescaled_BSSN_rfm_basis->betU1 = xx0 * (BSSN_Cart_basis->BU0 * tmp56 + BSSN_Cart_basis->BU1 * tmp27 * tmp51 + BSSN_Cart_basis->BU2 * tmp51 * tmp58);
  rescaled_BSSN_rfm_basis->betU2 = tmp17 * (BSSN_Cart_basis->BU0 * tmp60 + BSSN_Cart_basis->BU1 * tmp59);
  rescaled_BSSN_rfm_basis->cf = BSSN_Cart_basis->cf;
  rescaled_BSSN_rfm_basis->hDD00 = BSSN_Cart_basis->gammabarDD00 * tmp11 + BSSN_Cart_basis->gammabarDD02 * tmp3 * tmp5 +
                                   BSSN_Cart_basis->gammabarDD11 * tmp13 + BSSN_Cart_basis->gammabarDD12 * tmp3 * tmp8 +
                                   BSSN_Cart_basis->gammabarDD22 * tmp1 + tmp62 * tmp7 * tmp9 - 1;
  rescaled_BSSN_rfm_basis->hDD01 =
      tmp16 * (BSSN_Cart_basis->gammabarDD00 * tmp10 * tmp18 + BSSN_Cart_basis->gammabarDD02 * tmp20 - BSSN_Cart_basis->gammabarDD02 * tmp23 +
               BSSN_Cart_basis->gammabarDD11 * tmp12 * tmp18 + BSSN_Cart_basis->gammabarDD12 * tmp22 - BSSN_Cart_basis->gammabarDD12 * tmp24 -
               BSSN_Cart_basis->gammabarDD22 * tmp18 + tmp27 * tmp62);
  rescaled_BSSN_rfm_basis->hDD02 =
      tmp16 * tmp30 *
      (-BSSN_Cart_basis->gammabarDD00 * tmp24 * tmp4 + BSSN_Cart_basis->gammabarDD01 * tmp10 * tmp9 * xx0 - BSSN_Cart_basis->gammabarDD01 * tmp28 -
       BSSN_Cart_basis->gammabarDD02 * tmp27 + BSSN_Cart_basis->gammabarDD11 * tmp4 * tmp7 * tmp9 * xx0 +
       BSSN_Cart_basis->gammabarDD12 * tmp0 * tmp2 * tmp4 * xx0);
  rescaled_BSSN_rfm_basis->hDD11 =
      tmp33 * (BSSN_Cart_basis->gammabarDD00 * tmp10 * tmp38 - BSSN_Cart_basis->gammabarDD02 * tmp36 * tmp5 + BSSN_Cart_basis->gammabarDD11 * tmp39 -
               BSSN_Cart_basis->gammabarDD12 * tmp36 * tmp8 + BSSN_Cart_basis->gammabarDD22 * tmp34 - tmp32 + tmp38 * tmp62 * tmp7);
  rescaled_BSSN_rfm_basis->hDD12 = tmp30 * tmp33 *
                                   (-BSSN_Cart_basis->gammabarDD00 * tmp43 + BSSN_Cart_basis->gammabarDD01 * tmp0 * tmp10 * tmp2 * tmp32 -
                                    BSSN_Cart_basis->gammabarDD01 * tmp42 + BSSN_Cart_basis->gammabarDD02 * tmp32 * tmp7 * tmp9 +
                                    BSSN_Cart_basis->gammabarDD11 * tmp0 * tmp2 * tmp32 * tmp4 * tmp7 - BSSN_Cart_basis->gammabarDD12 * tmp41);
  rescaled_BSSN_rfm_basis->hDD22 =
      tmp48 * (BSSN_Cart_basis->gammabarDD00 * tmp12 * tmp34 + BSSN_Cart_basis->gammabarDD11 * tmp10 * tmp34 - tmp34 - tmp47 * tmp62);
  rescaled_BSSN_rfm_basis->trK = BSSN_Cart_basis->trK;
  rescaled_BSSN_rfm_basis->vetU0 =
      BSSN_Cart_basis->betaU0 * tmp41 * tmp51 + BSSN_Cart_basis->betaU1 * tmp47 * tmp51 + BSSN_Cart_basis->betaU2 * tmp51 * tmp54;
  rescaled_BSSN_rfm_basis->vetU1 =
      xx0 * (BSSN_Cart_basis->betaU0 * tmp56 + BSSN_Cart_basis->betaU1 * tmp27 * tmp51 + BSSN_Cart_basis->betaU2 * tmp51 * tmp58);
  rescaled_BSSN_rfm_basis->vetU2 = tmp17 * (BSSN_Cart_basis->betaU0 * tmp60 + BSSN_Cart_basis->betaU1 * tmp59);
}
/*
 * Compute lambdaU in Spherical coordinates
 */
__global__
void initial_data_lambdaU_grid_interior_gpu(REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2, REAL *restrict in_gfs) {

  int const & Nxx0 = d_params.Nxx0;
  int const & Nxx1 = d_params.Nxx1;
  int const & Nxx2 = d_params.Nxx2;

  int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

  REAL const & invdxx0 = d_params.invdxx0;
  REAL const & invdxx1 = d_params.invdxx1;
  REAL const & invdxx2 = d_params.invdxx2;

  // Global data index - expecting a 1D dataset
  // Thread indices
  const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
  const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
  // Thread strides
  const int stride0 = blockDim.x * gridDim.x;
  const int stride1 = blockDim.y * gridDim.y;
  const int stride2 = blockDim.z * gridDim.z;

  // REAL * xx[3] = {_xx0, _xx1, _xx2};

  for(size_t i2 = tid2+NGHOSTS; i2 < Nxx2+NGHOSTS; i2 += stride2) {
    for(size_t i1 = tid1+NGHOSTS; i1 < Nxx1+NGHOSTS; i1 += stride1) {
      for(size_t i0 = tid0+NGHOSTS; i0 < Nxx0+NGHOSTS; i0 += stride0) {
        __attribute_maybe_unused__ const REAL xx2 = _xx2[i2];
        __attribute_maybe_unused__ const REAL xx1 = _xx1[i1];
        __attribute_maybe_unused__ const REAL xx0 = _xx0[i0]; /*
                                     * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
                                     * Read gridfunction(s) from main memory and compute FD stencils as needed.
                                     */
        const REAL hDD00_i2m2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 2)];
        const REAL hDD00_i2m1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 - 1)];
        const REAL hDD00_i1m2 = in_gfs[IDX4(HDD00GF, i0, i1 - 2, i2)];
        const REAL hDD00_i1m1 = in_gfs[IDX4(HDD00GF, i0, i1 - 1, i2)];
        const REAL hDD00_i0m2 = in_gfs[IDX4(HDD00GF, i0 - 2, i1, i2)];
        const REAL hDD00_i0m1 = in_gfs[IDX4(HDD00GF, i0 - 1, i1, i2)];
        const REAL hDD00 = in_gfs[IDX4(HDD00GF, i0, i1, i2)];
        const REAL hDD00_i0p1 = in_gfs[IDX4(HDD00GF, i0 + 1, i1, i2)];
        const REAL hDD00_i0p2 = in_gfs[IDX4(HDD00GF, i0 + 2, i1, i2)];
        const REAL hDD00_i1p1 = in_gfs[IDX4(HDD00GF, i0, i1 + 1, i2)];
        const REAL hDD00_i1p2 = in_gfs[IDX4(HDD00GF, i0, i1 + 2, i2)];
        const REAL hDD00_i2p1 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 1)];
        const REAL hDD00_i2p2 = in_gfs[IDX4(HDD00GF, i0, i1, i2 + 2)];
        const REAL hDD01_i2m2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 2)];
        const REAL hDD01_i2m1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 - 1)];
        const REAL hDD01_i1m2 = in_gfs[IDX4(HDD01GF, i0, i1 - 2, i2)];
        const REAL hDD01_i1m1 = in_gfs[IDX4(HDD01GF, i0, i1 - 1, i2)];
        const REAL hDD01_i0m2 = in_gfs[IDX4(HDD01GF, i0 - 2, i1, i2)];
        const REAL hDD01_i0m1 = in_gfs[IDX4(HDD01GF, i0 - 1, i1, i2)];
        const REAL hDD01 = in_gfs[IDX4(HDD01GF, i0, i1, i2)];
        const REAL hDD01_i0p1 = in_gfs[IDX4(HDD01GF, i0 + 1, i1, i2)];
        const REAL hDD01_i0p2 = in_gfs[IDX4(HDD01GF, i0 + 2, i1, i2)];
        const REAL hDD01_i1p1 = in_gfs[IDX4(HDD01GF, i0, i1 + 1, i2)];
        const REAL hDD01_i1p2 = in_gfs[IDX4(HDD01GF, i0, i1 + 2, i2)];
        const REAL hDD01_i2p1 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 1)];
        const REAL hDD01_i2p2 = in_gfs[IDX4(HDD01GF, i0, i1, i2 + 2)];
        const REAL hDD02_i2m2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 2)];
        const REAL hDD02_i2m1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 - 1)];
        const REAL hDD02_i1m2 = in_gfs[IDX4(HDD02GF, i0, i1 - 2, i2)];
        const REAL hDD02_i1m1 = in_gfs[IDX4(HDD02GF, i0, i1 - 1, i2)];
        const REAL hDD02_i0m2 = in_gfs[IDX4(HDD02GF, i0 - 2, i1, i2)];
        const REAL hDD02_i0m1 = in_gfs[IDX4(HDD02GF, i0 - 1, i1, i2)];
        const REAL hDD02 = in_gfs[IDX4(HDD02GF, i0, i1, i2)];
        const REAL hDD02_i0p1 = in_gfs[IDX4(HDD02GF, i0 + 1, i1, i2)];
        const REAL hDD02_i0p2 = in_gfs[IDX4(HDD02GF, i0 + 2, i1, i2)];
        const REAL hDD02_i1p1 = in_gfs[IDX4(HDD02GF, i0, i1 + 1, i2)];
        const REAL hDD02_i1p2 = in_gfs[IDX4(HDD02GF, i0, i1 + 2, i2)];
        const REAL hDD02_i2p1 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 1)];
        const REAL hDD02_i2p2 = in_gfs[IDX4(HDD02GF, i0, i1, i2 + 2)];
        const REAL hDD11_i2m2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 2)];
        const REAL hDD11_i2m1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 - 1)];
        const REAL hDD11_i1m2 = in_gfs[IDX4(HDD11GF, i0, i1 - 2, i2)];
        const REAL hDD11_i1m1 = in_gfs[IDX4(HDD11GF, i0, i1 - 1, i2)];
        const REAL hDD11_i0m2 = in_gfs[IDX4(HDD11GF, i0 - 2, i1, i2)];
        const REAL hDD11_i0m1 = in_gfs[IDX4(HDD11GF, i0 - 1, i1, i2)];
        const REAL hDD11 = in_gfs[IDX4(HDD11GF, i0, i1, i2)];
        const REAL hDD11_i0p1 = in_gfs[IDX4(HDD11GF, i0 + 1, i1, i2)];
        const REAL hDD11_i0p2 = in_gfs[IDX4(HDD11GF, i0 + 2, i1, i2)];
        const REAL hDD11_i1p1 = in_gfs[IDX4(HDD11GF, i0, i1 + 1, i2)];
        const REAL hDD11_i1p2 = in_gfs[IDX4(HDD11GF, i0, i1 + 2, i2)];
        const REAL hDD11_i2p1 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 1)];
        const REAL hDD11_i2p2 = in_gfs[IDX4(HDD11GF, i0, i1, i2 + 2)];
        const REAL hDD12_i2m2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 2)];
        const REAL hDD12_i2m1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 - 1)];
        const REAL hDD12_i1m2 = in_gfs[IDX4(HDD12GF, i0, i1 - 2, i2)];
        const REAL hDD12_i1m1 = in_gfs[IDX4(HDD12GF, i0, i1 - 1, i2)];
        const REAL hDD12_i0m2 = in_gfs[IDX4(HDD12GF, i0 - 2, i1, i2)];
        const REAL hDD12_i0m1 = in_gfs[IDX4(HDD12GF, i0 - 1, i1, i2)];
        const REAL hDD12 = in_gfs[IDX4(HDD12GF, i0, i1, i2)];
        const REAL hDD12_i0p1 = in_gfs[IDX4(HDD12GF, i0 + 1, i1, i2)];
        const REAL hDD12_i0p2 = in_gfs[IDX4(HDD12GF, i0 + 2, i1, i2)];
        const REAL hDD12_i1p1 = in_gfs[IDX4(HDD12GF, i0, i1 + 1, i2)];
        const REAL hDD12_i1p2 = in_gfs[IDX4(HDD12GF, i0, i1 + 2, i2)];
        const REAL hDD12_i2p1 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 1)];
        const REAL hDD12_i2p2 = in_gfs[IDX4(HDD12GF, i0, i1, i2 + 2)];
        const REAL hDD22_i2m2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 2)];
        const REAL hDD22_i2m1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 - 1)];
        const REAL hDD22_i1m2 = in_gfs[IDX4(HDD22GF, i0, i1 - 2, i2)];
        const REAL hDD22_i1m1 = in_gfs[IDX4(HDD22GF, i0, i1 - 1, i2)];
        const REAL hDD22_i0m2 = in_gfs[IDX4(HDD22GF, i0 - 2, i1, i2)];
        const REAL hDD22_i0m1 = in_gfs[IDX4(HDD22GF, i0 - 1, i1, i2)];
        const REAL hDD22 = in_gfs[IDX4(HDD22GF, i0, i1, i2)];
        const REAL hDD22_i0p1 = in_gfs[IDX4(HDD22GF, i0 + 1, i1, i2)];
        const REAL hDD22_i0p2 = in_gfs[IDX4(HDD22GF, i0 + 2, i1, i2)];
        const REAL hDD22_i1p1 = in_gfs[IDX4(HDD22GF, i0, i1 + 1, i2)];
        const REAL hDD22_i1p2 = in_gfs[IDX4(HDD22GF, i0, i1 + 2, i2)];
        const REAL hDD22_i2p1 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 1)];
        const REAL hDD22_i2p2 = in_gfs[IDX4(HDD22GF, i0, i1, i2 + 2)];
        const REAL hDD_dD000 = invdxx0 * (FDPart1_Rational_1_12 * (hDD00_i0m2 - hDD00_i0p2) + FDPart1_Rational_2_3 * (-hDD00_i0m1 + hDD00_i0p1));
        const REAL hDD_dD001 = invdxx1 * (FDPart1_Rational_1_12 * (hDD00_i1m2 - hDD00_i1p2) + FDPart1_Rational_2_3 * (-hDD00_i1m1 + hDD00_i1p1));
        const REAL hDD_dD002 = invdxx2 * (FDPart1_Rational_1_12 * (hDD00_i2m2 - hDD00_i2p2) + FDPart1_Rational_2_3 * (-hDD00_i2m1 + hDD00_i2p1));
        const REAL hDD_dD010 = invdxx0 * (FDPart1_Rational_1_12 * (hDD01_i0m2 - hDD01_i0p2) + FDPart1_Rational_2_3 * (-hDD01_i0m1 + hDD01_i0p1));
        const REAL hDD_dD011 = invdxx1 * (FDPart1_Rational_1_12 * (hDD01_i1m2 - hDD01_i1p2) + FDPart1_Rational_2_3 * (-hDD01_i1m1 + hDD01_i1p1));
        const REAL hDD_dD012 = invdxx2 * (FDPart1_Rational_1_12 * (hDD01_i2m2 - hDD01_i2p2) + FDPart1_Rational_2_3 * (-hDD01_i2m1 + hDD01_i2p1));
        const REAL hDD_dD020 = invdxx0 * (FDPart1_Rational_1_12 * (hDD02_i0m2 - hDD02_i0p2) + FDPart1_Rational_2_3 * (-hDD02_i0m1 + hDD02_i0p1));
        const REAL hDD_dD021 = invdxx1 * (FDPart1_Rational_1_12 * (hDD02_i1m2 - hDD02_i1p2) + FDPart1_Rational_2_3 * (-hDD02_i1m1 + hDD02_i1p1));
        const REAL hDD_dD022 = invdxx2 * (FDPart1_Rational_1_12 * (hDD02_i2m2 - hDD02_i2p2) + FDPart1_Rational_2_3 * (-hDD02_i2m1 + hDD02_i2p1));
        const REAL hDD_dD110 = invdxx0 * (FDPart1_Rational_1_12 * (hDD11_i0m2 - hDD11_i0p2) + FDPart1_Rational_2_3 * (-hDD11_i0m1 + hDD11_i0p1));
        const REAL hDD_dD111 = invdxx1 * (FDPart1_Rational_1_12 * (hDD11_i1m2 - hDD11_i1p2) + FDPart1_Rational_2_3 * (-hDD11_i1m1 + hDD11_i1p1));
        const REAL hDD_dD112 = invdxx2 * (FDPart1_Rational_1_12 * (hDD11_i2m2 - hDD11_i2p2) + FDPart1_Rational_2_3 * (-hDD11_i2m1 + hDD11_i2p1));
        const REAL hDD_dD120 = invdxx0 * (FDPart1_Rational_1_12 * (hDD12_i0m2 - hDD12_i0p2) + FDPart1_Rational_2_3 * (-hDD12_i0m1 + hDD12_i0p1));
        const REAL hDD_dD121 = invdxx1 * (FDPart1_Rational_1_12 * (hDD12_i1m2 - hDD12_i1p2) + FDPart1_Rational_2_3 * (-hDD12_i1m1 + hDD12_i1p1));
        const REAL hDD_dD122 = invdxx2 * (FDPart1_Rational_1_12 * (hDD12_i2m2 - hDD12_i2p2) + FDPart1_Rational_2_3 * (-hDD12_i2m1 + hDD12_i2p1));
        const REAL hDD_dD220 = invdxx0 * (FDPart1_Rational_1_12 * (hDD22_i0m2 - hDD22_i0p2) + FDPart1_Rational_2_3 * (-hDD22_i0m1 + hDD22_i0p1));
        const REAL hDD_dD221 = invdxx1 * (FDPart1_Rational_1_12 * (hDD22_i1m2 - hDD22_i1p2) + FDPart1_Rational_2_3 * (-hDD22_i1m1 + hDD22_i1p1));
        const REAL hDD_dD222 = invdxx2 * (FDPart1_Rational_1_12 * (hDD22_i2m2 - hDD22_i2p2) + FDPart1_Rational_2_3 * (-hDD22_i2m1 + hDD22_i2p1));

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = sin(xx1);
        const REAL FDPart3tmp2 = 2 * xx0;
        const REAL FDPart3tmp5 = ((xx0) * (xx0) * (xx0));
        const REAL FDPart3tmp6 = ((xx0) * (xx0));
        const REAL FDPart3tmp10 = ((xx0) * (xx0) * (xx0) * (xx0));
        const REAL FDPart3tmp12 = hDD00 + 1;
        const REAL FDPart3tmp27 = hDD_dD012 * xx0;
        const REAL FDPart3tmp28 = cos(xx1);
        const REAL FDPart3tmp63 = -1 / xx0;  // This could go in precompute...
        const REAL FDPart3tmp7 = FDPart3tmp6 * hDD11 + FDPart3tmp6;
        const REAL FDPart3tmp11 = ((FDPart3tmp0) * (FDPart3tmp0));
        const REAL FDPart3tmp17 = FDPart3tmp6 * ((hDD01) * (hDD01));
        const REAL FDPart3tmp23 = FDPart3tmp2 * hDD_dD010 + 2 * hDD01 - hDD_dD001;
        const REAL FDPart3tmp26 = FDPart3tmp2 * hDD11 + FDPart3tmp2 + FDPart3tmp6 * hDD_dD110;
        const REAL FDPart3tmp29 = FDPart3tmp28 * hDD02 * xx0;
        const REAL FDPart3tmp33 = FDPart3tmp0 * FDPart3tmp6;
        const REAL FDPart3tmp45 = FDPart3tmp6 * hDD_dD111;
        const REAL FDPart3tmp46 = FDPart3tmp6 * hDD_dD112;
        const REAL FDPart3tmp4 = FDPart3tmp0 * FDPart3tmp2 * hDD_dD020 + 2 * FDPart3tmp0 * hDD02 - hDD_dD002;
        const REAL FDPart3tmp9 = FDPart3tmp0 * FDPart3tmp5 * hDD01 * hDD12 - FDPart3tmp0 * FDPart3tmp7 * hDD02 * xx0;
        const REAL FDPart3tmp13 = FDPart3tmp10 * FDPart3tmp11 * ((hDD12) * (hDD12));
        const REAL FDPart3tmp14 = FDPart3tmp11 * FDPart3tmp6;
        const REAL FDPart3tmp31 = FDPart3tmp0 * hDD_dD021 * xx0;
        const REAL FDPart3tmp35 = FDPart3tmp0 * FDPart3tmp2 * hDD12;
        const REAL FDPart3tmp48 = 2 * FDPart3tmp0 * FDPart3tmp6 * hDD_dD121 + 2 * FDPart3tmp28 * FDPart3tmp6 * hDD12 - FDPart3tmp46;
        const REAL FDPart3tmp49 = -FDPart3tmp26 + 2 * hDD_dD011 * xx0;
        const REAL FDPart3tmp52 = 2 * FDPart3tmp0 * FDPart3tmp28 * FDPart3tmp6;
        const REAL FDPart3tmp55 = -FDPart3tmp12 * FDPart3tmp33 * hDD12 + FDPart3tmp33 * hDD01 * hDD02;
        const REAL FDPart3tmp15 = FDPart3tmp14 * ((hDD02) * (hDD02));
        const REAL FDPart3tmp16 = FDPart3tmp14 * hDD22 + FDPart3tmp14;
        const REAL FDPart3tmp36 = FDPart3tmp33 * hDD_dD120 + FDPart3tmp35;
        const REAL FDPart3tmp41 = 2 * FDPart3tmp11 * xx0;
        const REAL FDPart3tmp53 = FDPart3tmp14 * hDD_dD221 + FDPart3tmp52 * hDD22 + FDPart3tmp52;
        const REAL FDPart3tmp57 = FDPart3tmp14 * hDD_dD222;
        const REAL FDPart3tmp18 = (1.0 / (2 * FDPart3tmp10 * FDPart3tmp11 * hDD01 * hDD02 * hDD12 - FDPart3tmp12 * FDPart3tmp13 +
                                          FDPart3tmp12 * FDPart3tmp16 * FDPart3tmp7 - FDPart3tmp15 * FDPart3tmp7 - FDPart3tmp16 * FDPart3tmp17));
        const REAL FDPart3tmp24 = FDPart3tmp11 * FDPart3tmp5 * hDD02 * hDD12 - FDPart3tmp16 * hDD01 * xx0;
        const REAL FDPart3tmp37 = -FDPart3tmp27 + FDPart3tmp29 + FDPart3tmp31 + FDPart3tmp36;
        const REAL FDPart3tmp42 = FDPart3tmp14 * hDD_dD220 + FDPart3tmp41 * hDD22 + FDPart3tmp41;
        const REAL FDPart3tmp43 = FDPart3tmp27 - FDPart3tmp29 - FDPart3tmp31 + FDPart3tmp36;
        const REAL FDPart3tmp54 = FDPart3tmp27 + FDPart3tmp29 + FDPart3tmp31 - FDPart3tmp33 * hDD_dD120 - FDPart3tmp35;
        const REAL FDPart3tmp58 = 2 * FDPart3tmp0 * FDPart3tmp6 * hDD_dD122 - FDPart3tmp53;
        const REAL FDPart3tmp19 = (0.5) * FDPart3tmp18;
        const REAL FDPart3tmp21 = FDPart3tmp18 * (-FDPart3tmp13 + FDPart3tmp16 * FDPart3tmp7);
        const REAL FDPart3tmp38 = 2 * FDPart3tmp18;
        const REAL FDPart3tmp50 = FDPart3tmp18 * (FDPart3tmp12 * FDPart3tmp16 - FDPart3tmp15);
        const REAL FDPart3tmp59 = 2 * FDPart3tmp0 * hDD_dD022 * xx0 - FDPart3tmp42;
        const REAL FDPart3tmp60 = FDPart3tmp18 * (FDPart3tmp12 * FDPart3tmp7 - FDPart3tmp17);
        const REAL FDPart3tmp20 = FDPart3tmp19 * FDPart3tmp9;
        const REAL FDPart3tmp22 = (0.5) * FDPart3tmp21;
        const REAL FDPart3tmp25 = FDPart3tmp19 * FDPart3tmp24;
        const REAL FDPart3tmp39 = FDPart3tmp24 * FDPart3tmp38;
        const REAL FDPart3tmp44 = FDPart3tmp38 * FDPart3tmp9;
        const REAL FDPart3tmp56 = FDPart3tmp38 * FDPart3tmp55;
        const REAL FDPart3tmp61 = FDPart3tmp19 * FDPart3tmp55;
        const REAL FDPart3tmp62 = (0.5) * FDPart3tmp50;
        const REAL FDPart3tmp64 = (0.5) * FDPart3tmp60;
        in_gfs[IDX4(LAMBDAU0GF, i0, i1, i2)] =
            FDPart3tmp21 * (FDPart3tmp20 * FDPart3tmp4 + FDPart3tmp22 * hDD_dD000 + FDPart3tmp23 * FDPart3tmp25) +
            FDPart3tmp39 * (FDPart3tmp20 * FDPart3tmp37 + FDPart3tmp22 * hDD_dD001 + FDPart3tmp25 * FDPart3tmp26) +
            FDPart3tmp44 * (FDPart3tmp20 * FDPart3tmp42 + FDPart3tmp22 * hDD_dD002 + FDPart3tmp25 * FDPart3tmp43) +
            FDPart3tmp50 * (FDPart3tmp20 * FDPart3tmp48 + FDPart3tmp22 * FDPart3tmp49 + FDPart3tmp25 * FDPart3tmp45 + xx0) +
            FDPart3tmp56 * (FDPart3tmp20 * FDPart3tmp53 + FDPart3tmp22 * FDPart3tmp54 + FDPart3tmp25 * FDPart3tmp46) +
            FDPart3tmp60 * (FDPart3tmp11 * xx0 + FDPart3tmp20 * FDPart3tmp57 + FDPart3tmp22 * FDPart3tmp59 + FDPart3tmp25 * FDPart3tmp58);
        in_gfs[IDX4(LAMBDAU1GF, i0, i1, i2)] =
            xx0 *
            (FDPart3tmp21 * (FDPart3tmp23 * FDPart3tmp62 + FDPart3tmp25 * hDD_dD000 + FDPart3tmp4 * FDPart3tmp61) +
             FDPart3tmp39 * (FDPart3tmp25 * hDD_dD001 + FDPart3tmp26 * FDPart3tmp62 + FDPart3tmp37 * FDPart3tmp61 + FDPart3tmp63) +
             FDPart3tmp44 * (FDPart3tmp25 * hDD_dD002 + FDPart3tmp42 * FDPart3tmp61 + FDPart3tmp43 * FDPart3tmp62) +
             FDPart3tmp50 * (FDPart3tmp25 * FDPart3tmp49 + FDPart3tmp45 * FDPart3tmp62 + FDPart3tmp48 * FDPart3tmp61) +
             FDPart3tmp56 * (FDPart3tmp25 * FDPart3tmp54 + FDPart3tmp46 * FDPart3tmp62 + FDPart3tmp53 * FDPart3tmp61) +
             FDPart3tmp60 * (FDPart3tmp0 * FDPart3tmp28 + FDPart3tmp25 * FDPart3tmp59 + FDPart3tmp57 * FDPart3tmp61 + FDPart3tmp58 * FDPart3tmp62));
        in_gfs[IDX4(LAMBDAU2GF, i0, i1, i2)] =
            FDPart3tmp0 * xx0 *
            (FDPart3tmp21 * (FDPart3tmp20 * hDD_dD000 + FDPart3tmp23 * FDPart3tmp61 + FDPart3tmp4 * FDPart3tmp64) +
             FDPart3tmp39 * (FDPart3tmp20 * hDD_dD001 + FDPart3tmp26 * FDPart3tmp61 + FDPart3tmp37 * FDPart3tmp64) +
             FDPart3tmp44 * (FDPart3tmp20 * hDD_dD002 + FDPart3tmp42 * FDPart3tmp64 + FDPart3tmp43 * FDPart3tmp61 + FDPart3tmp63) +
             FDPart3tmp50 * (FDPart3tmp20 * FDPart3tmp49 + FDPart3tmp45 * FDPart3tmp61 + FDPart3tmp48 * FDPart3tmp64) +
             FDPart3tmp56 * (FDPart3tmp20 * FDPart3tmp54 + FDPart3tmp46 * FDPart3tmp61 + FDPart3tmp53 * FDPart3tmp64 - FDPart3tmp28 / FDPart3tmp0) +
             FDPart3tmp60 * (FDPart3tmp20 * FDPart3tmp59 + FDPart3tmp57 * FDPart3tmp64 + FDPart3tmp58 * FDPart3tmp61));

      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    }   // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  }     // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}
#include <curand_kernel.h>
__device__ void perturb_ID(initial_data_struct *restrict initial_data, unsigned long seed, int tid) {

  curandState state;

  curand_init(seed, tid, 0, &state);

  REAL perturbation = (1. + 1e-12 * curand_uniform(&state));

  initial_data->BSphorCartU0   *= perturbation;
  initial_data->BSphorCartU0   *= perturbation;
  initial_data->BSphorCartU0   *= perturbation;
  initial_data->KSphorCartDD00 *= perturbation;
  initial_data->KSphorCartDD01 *= perturbation;
  initial_data->KSphorCartDD02 *= perturbation;
  initial_data->KSphorCartDD11 *= perturbation;
  initial_data->KSphorCartDD12 *= perturbation;
  initial_data->KSphorCartDD22 *= perturbation;
  initial_data->alpha          *= perturbation;
  initial_data->betaSphorCartU0    *= perturbation;
  initial_data->betaSphorCartU1    *= perturbation;
  initial_data->betaSphorCartU2    *= perturbation;
  initial_data->gammaSphorCartDD00 *= perturbation;
  initial_data->gammaSphorCartDD01 *= perturbation;
  initial_data->gammaSphorCartDD02 *= perturbation;
  initial_data->gammaSphorCartDD11 *= perturbation;
  initial_data->gammaSphorCartDD12 *= perturbation;
  initial_data->gammaSphorCartDD22 *= perturbation;
}
__global__
void initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical_gpu(const commondata_struct *restrict commondata,
  REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2, REAL *restrict y_n_gfs, 
  const ID_persist_struct *restrict ID_persist, ID_pfunc ID_function) {
    int const & Nxx_plus_2NGHOSTS0 = d_params.Nxx_plus_2NGHOSTS0;
    int const & Nxx_plus_2NGHOSTS1 = d_params.Nxx_plus_2NGHOSTS1;
    int const & Nxx_plus_2NGHOSTS2 = d_params.Nxx_plus_2NGHOSTS2;

    // Global data index - expecting a 1D dataset
    // Thread indices
    const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
    const int tid1 = threadIdx.y + blockIdx.y*blockDim.y;
    const int tid2 = threadIdx.z + blockIdx.z*blockDim.z;
    // Thread strides
    const int stride0 = blockDim.x * gridDim.x;
    const int stride1 = blockDim.y * gridDim.y;
    const int stride2 = blockDim.z * gridDim.z;

    REAL * xx[3] = {_xx0, _xx1, _xx2};

    for(size_t i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
      for(size_t i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
        for(size_t i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
          // xCart is the global Cartesian coordinate, which accounts for any grid offsets from the origin.
          REAL xCart[3];
          xx_to_Cart(xx, i0, i1, i2, xCart);
          initial_data_struct initial_data;
          ID_function(commondata, xCart, ID_persist, &initial_data);
          // TESTING ONLY
          // perturb_ID(&initial_data, 0, tid0);

          ADM_Cart_basis_struct ADM_Cart_basis;
          ADM_SphorCart_to_Cart(commondata, xCart, &initial_data, &ADM_Cart_basis);

          BSSN_Cart_basis_struct BSSN_Cart_basis;
          ADM_Cart_to_BSSN_Cart(commondata, xCart, &ADM_Cart_basis, &BSSN_Cart_basis);

          rescaled_BSSN_rfm_basis_struct rescaled_BSSN_rfm_basis;
          BSSN_Cart_to_rescaled_BSSN_rfm(commondata, xCart, &BSSN_Cart_basis, &rescaled_BSSN_rfm_basis);
          
          const int idx3 = IDX3(i0, i1, i2);
          y_n_gfs[IDX4pt(ADD00GF, idx3)] = rescaled_BSSN_rfm_basis.aDD00;
          y_n_gfs[IDX4pt(ADD01GF, idx3)] = rescaled_BSSN_rfm_basis.aDD01;
          y_n_gfs[IDX4pt(ADD02GF, idx3)] = rescaled_BSSN_rfm_basis.aDD02;
          y_n_gfs[IDX4pt(ADD11GF, idx3)] = rescaled_BSSN_rfm_basis.aDD11;
          y_n_gfs[IDX4pt(ADD12GF, idx3)] = rescaled_BSSN_rfm_basis.aDD12;
          y_n_gfs[IDX4pt(ADD22GF, idx3)] = rescaled_BSSN_rfm_basis.aDD22;
          y_n_gfs[IDX4pt(ALPHAGF, idx3)] = rescaled_BSSN_rfm_basis.alpha;
          y_n_gfs[IDX4pt(BETU0GF, idx3)] = rescaled_BSSN_rfm_basis.betU0;
          y_n_gfs[IDX4pt(BETU1GF, idx3)] = rescaled_BSSN_rfm_basis.betU1;
          y_n_gfs[IDX4pt(BETU2GF, idx3)] = rescaled_BSSN_rfm_basis.betU2;
          y_n_gfs[IDX4pt(CFGF, idx3)] = rescaled_BSSN_rfm_basis.cf;
          y_n_gfs[IDX4pt(HDD00GF, idx3)] = rescaled_BSSN_rfm_basis.hDD00;
          y_n_gfs[IDX4pt(HDD01GF, idx3)] = rescaled_BSSN_rfm_basis.hDD01;
          y_n_gfs[IDX4pt(HDD02GF, idx3)] = rescaled_BSSN_rfm_basis.hDD02;
          y_n_gfs[IDX4pt(HDD11GF, idx3)] = rescaled_BSSN_rfm_basis.hDD11;
          y_n_gfs[IDX4pt(HDD12GF, idx3)] = rescaled_BSSN_rfm_basis.hDD12;
          y_n_gfs[IDX4pt(HDD22GF, idx3)] = rescaled_BSSN_rfm_basis.hDD22;
          y_n_gfs[IDX4pt(TRKGF, idx3)] = rescaled_BSSN_rfm_basis.trK;
          y_n_gfs[IDX4pt(VETU0GF, idx3)] = rescaled_BSSN_rfm_basis.vetU0;
          y_n_gfs[IDX4pt(VETU1GF, idx3)] = rescaled_BSSN_rfm_basis.vetU1;
          y_n_gfs[IDX4pt(VETU2GF, idx3)] = rescaled_BSSN_rfm_basis.vetU2;

          // Initialize lambdaU to zero
          y_n_gfs[IDX4pt(LAMBDAU0GF, idx3)] = 0.0;
          y_n_gfs[IDX4pt(LAMBDAU1GF, idx3)] = 0.0;
          y_n_gfs[IDX4pt(LAMBDAU2GF, idx3)] = 0.0;
        }
      }
    }
}

/*
 * Read ADM data in the Cartesian basis, and output rescaled BSSN data in the Spherical basis
 */
void initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical(
    const commondata_struct *restrict commondata, const params_struct *restrict params, REAL * xx[3], bc_struct *restrict bcstruct,
    MoL_gridfunctions_struct *restrict gridfuncs, ID_persist_struct *restrict ID_persist, 
    void ID_function(const commondata_struct *restrict commondata, const REAL xCart[3],
                     const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data)) {
  int const & Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Common data is very rarely used on the GPU, but enough of it is needed that we temporarily store
  // the entire struct there to minimize memory useage
  commondata_struct* commondata_gpu;
  cudaMalloc(&commondata_gpu, sizeof(commondata_struct));
  cudaCheckErrors(cudaMalloc, "Memory failure");
  cudaMemcpy(commondata_gpu, commondata, sizeof(commondata_struct), cudaMemcpyHostToDevice);
  cudaCheckErrors(cudaMemcpy, "Memory failure");

  ID_persist_struct* ID_persist_gpu;
  cudaMalloc(&ID_persist_gpu, sizeof(ID_persist_struct));
  cudaCheckErrors(cudaMalloc, "Memory failure");
  cudaMemcpy(ID_persist_gpu, ID_persist, sizeof(ID_persist_struct), cudaMemcpyHostToDevice);
  cudaCheckErrors(cudaMemcpy, "Memory failure");

  ID_pfunc host_function_ptr;
  cudaMemcpyFromSymbol(&host_function_ptr, id_ptr, sizeof(ID_pfunc));

  int threads_in_x_dir = MIN(GPU_THREADX_MAX, params->Nxx0 / 32);
  int threads_in_y_dir = MIN(GPU_THREADX_MAX / threads_in_x_dir, params->Nxx1);
  int threads_in_z_dir = 1;

  // Setup our thread layout
  dim3 block_threads(threads_in_x_dir, threads_in_y_dir, threads_in_z_dir);

  // Setup our grid layout such that our tiles will iterate through the entire
  // numerical space
  dim3 grid_blocks(
    (Nxx_plus_2NGHOSTS0 + threads_in_x_dir - 1) / threads_in_x_dir,
    (Nxx_plus_2NGHOSTS1 + threads_in_y_dir - 1) / threads_in_y_dir,
    (Nxx_plus_2NGHOSTS2 + threads_in_z_dir - 1) / threads_in_z_dir
  );
  // printf("threads: %d - %d - %d\n", block_threads.x, block_threads.y, block_threads.z);
  // printf("blocks: %d - %d - %d\n", grid_blocks.x, grid_blocks.y, grid_blocks.z);

  initial_data_reader__convert_ADM_Cartesian_to_BSSN__rfm__Spherical_gpu<<<grid_blocks,block_threads>>>(
    commondata_gpu, xx[0], xx[1], xx[2], gridfuncs->y_n_gfs, ID_persist_gpu, host_function_ptr
  );

  // Now we've set all but lambda^i, which will be computed via a finite-difference of hDD.
  //    However, hDD is not correctly set in inner boundary points so we apply inner bcs first.

  // Apply inner bcs to get correct values of all tensor quantities across symmetry boundaries;
  //    BSSN_Cart_to_rescaled_BSSN_rfm() converts each xCart->xx, which guarantees a mapping
  //    to the grid interior. It therefore does not account for parity conditions across
  //    symmetry boundaries being correct.
  apply_bcs_inner_only(commondata, params, bcstruct, gridfuncs->y_n_gfs);

  // initial_data_lambdaU_grid_interior(commondata, params, xx, gridfuncs->y_n_gfs);
  initial_data_lambdaU_grid_interior_gpu<<<grid_blocks, block_threads>>>(xx[0], xx[1], xx[2], gridfuncs->y_n_gfs);
  cudaFree(commondata_gpu); 
  cudaFree(ID_persist_gpu);
}
