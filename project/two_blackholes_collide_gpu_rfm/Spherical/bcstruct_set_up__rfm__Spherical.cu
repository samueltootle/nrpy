#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
#include "../BHaH_gpu_defines.h"
#include "../BHaH_gpu_function_prototypes.h"
/*
 * EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt():
 * A coordinate system's "eigencoordinate" is the simplest member
 * of its family; all spherical-like coordinate systems have
 * Spherical as their eigencoordinate. The same is true for
 * cylindrical-like (Cylindrical is eigencoordinate),
 * Cartesian-like (Cartesian is the eigencoordinate), and
 * SymTP-like (SymTP is the eigencoordinate) coordinates.
 *
 * For a given gridpoint (i0,i1,i2) and corresponding coordinate
 * (x0,x1,x2), this function performs the dual mapping
 * (x0,x1,x2) -> (Cartx,Carty,Cartz) -> (x0,x1,x2)'
 * Note that (x0,x1,x2) IS NOT ALWAYS equal to (x0,x1,x2)';
 * For example consider in Spherical coordinates
 * (x0,x1,x2)=(r,theta,phi)=(-0.1,pi/4,pi/4).
 * This point will map to (x0,x1,x2)', in which x0>0,
 * because the inversion r=sqrt(Cartx^2+Carty^2+Cartz^2)
 * is always positive. In this case, (x0,x1,x2) is considered
 * an *inner* boundary point, and on a cell-centered grid
 * is guaranteed to map to a grid point in the grid interior;
 * filling in this point requires copying data, and possibly
 * multiplying by a +/- 1 if the data is from a gridfunction
 * storing tensors/vectors.
 *
 */
__device__
void EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(
  REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2,
    const int i0, const int i1, const int i2, 
      REAL x0x1x2_inbounds[3], int i0i1i2_inbounds[3]) {

  REAL const & dxx0 = d_params.dxx0;
  REAL const & dxx1 = d_params.dxx1;
  REAL const & dxx2 = d_params.dxx2;

  // This is a 3-step algorithm:
  // Step 1: (x0,x1,x2) -> (Cartx,Carty,Cartz)
  //         Find the Cartesian coordinate that (x0,x1,x2)
  //         maps to, assuming (x0,x1,x2) is the eigen-
  //         coordinate. Note that we assume (x0,x1,x2)
  //         has the same grid boundaries in both the
  //         original coordinate and the eigencoordinate.
  // Step 2: (Cartx,Carty,Cartz) -> (x0,x1,x2)'
  //         Find the interior eigencoordinate point
  //         (x0,x1,x2)' to which (Cartx,Carty,Cartz)
  //         maps, as well as the corresponding
  //         gridpoint integer index (i0,i1,i2). For
  //         cell-centered grids, (x0,x1,x2) will always
  //         overlap exactly (to roundoff error) a point
  //         on the numerical grid.
  // Step 3: Sanity check
  //         Convert x0(i0_inbounds),x1(i1_inbounds),x2(i2_inbounds) -> (Cartx,Carty,Cartz),
  //         and check that
  //         (Cartx,Carty,Cartz) == (Cartx(x0(i0)),Cartx(x1(i1)),Cartx(x2(i2)))
  //         If not, error out!

  // Step 1: Convert the (curvilinear) coordinate (x0,x1,x2) to Cartesian coordinates
  REAL xCart[3]; // where (x,y,z) is output
  {
    // xx_to_Cart for EigenCoordinate Spherical (orig coord = Spherical):
    REAL xx0 = _xx0[i0];
    REAL xx1 = _xx1[i1];
    REAL xx2 = _xx2[i2];
    /*
     *  Original SymPy expressions:
     *  "[xCart[0] = xx0*sin(xx1)*cos(xx2)]"
     *  "[xCart[1] = xx0*sin(xx1)*sin(xx2)]"
     *  "[xCart[2] = xx0*cos(xx1)]"
     */
    {
      const REAL tmp0 = xx0 * sin(xx1);
      xCart[0] = tmp0 * cos(xx2);
      xCart[1] = tmp0 * sin(xx2);
      xCart[2] = xx0 * cos(xx1);
    }
  }

  REAL Cartx = xCart[0];
  REAL Carty = xCart[1];
  REAL Cartz = xCart[2];

  // Step 2: Find the (i0_inbounds,i1_inbounds,i2_inbounds) corresponding to the above Cartesian coordinate.
  //   If (i0_inbounds,i1_inbounds,i2_inbounds) is in a ghost zone, then it must equal (i0,i1,i2), and
  //      the point is an outer boundary point.
  //   Otherwise (i0_inbounds,i1_inbounds,i2_inbounds) is in the grid interior, and data at (i0,i1,i2)
  //      must be replaced with data at (i0_inbounds,i1_inbounds,i2_inbounds), but multiplied by the
  //      appropriate parity condition (+/- 1).
  REAL Cart_to_xx0_inbounds, Cart_to_xx1_inbounds, Cart_to_xx2_inbounds;
  // Cart_to_xx for EigenCoordinate Spherical (orig coord = Spherical);
  /*
   *  Original SymPy expressions:
   *  "[Cart_to_xx0_inbounds = sqrt(Cartx**2 + Carty**2 + Cartz**2)]"
   *  "[Cart_to_xx1_inbounds = acos(Cartz/sqrt(Cartx**2 + Carty**2 + Cartz**2))]"
   *  "[Cart_to_xx2_inbounds = atan2(Carty, Cartx)]"
   */
  {
    const REAL tmp0 = sqrt(((Cartx) * (Cartx)) + ((Carty) * (Carty)) + ((Cartz) * (Cartz)));
    Cart_to_xx0_inbounds = tmp0;
    Cart_to_xx1_inbounds = acos(Cartz / tmp0);
    Cart_to_xx2_inbounds = atan2(Carty, Cartx);
  }

  // Next compute xxmin[i]. By definition,
  //    xx[i][j] = xxmin[i] + ((REAL)(j-NGHOSTS) + (1.0/2.0))*dxxi;
  // -> xxmin[i] = xx[i][0] - ((REAL)(0-NGHOSTS) + (1.0/2.0))*dxxi
  const REAL xxmin[3] = {_xx0[0] - ((REAL)(0 - NGHOSTS) + (1.0 / 2.0)) * dxx0, 
                         _xx1[0] - ((REAL)(0 - NGHOSTS) + (1.0 / 2.0)) * dxx1,
                         _xx2[0] - ((REAL)(0 - NGHOSTS) + (1.0 / 2.0)) * dxx2};

  // Finally compute i{0,1,2}_inbounds (add 0.5 to account for rounding down)
  const int i0_inbounds = (int)((Cart_to_xx0_inbounds - xxmin[0] - (1.0 / 2.0) * dxx0 + ((REAL)NGHOSTS) * dxx0) / dxx0 + 0.5);
  const int i1_inbounds = (int)((Cart_to_xx1_inbounds - xxmin[1] - (1.0 / 2.0) * dxx1 + ((REAL)NGHOSTS) * dxx1) / dxx1 + 0.5);
  const int i2_inbounds = (int)((Cart_to_xx2_inbounds - xxmin[2] - (1.0 / 2.0) * dxx2 + ((REAL)NGHOSTS) * dxx2) / dxx2 + 0.5);

  // Step 3: Convert x0(i0_inbounds),x1(i1_inbounds),x2(i2_inbounds) -> (Cartx,Carty,Cartz),
  //         and check that
  //         (Cartx,Carty,Cartz) == (Cartx(x0(i0)),Cartx(x1(i1)),Cartx(x2(i2)))
  //         If not, error out!

  // Step 3.a: Compute {x,y,z}Cart_from_xx, as a
  //           function of i0,i1,i2
  REAL xCart_from_xx, yCart_from_xx, zCart_from_xx;
  {
    // xx_to_Cart for Coordinate Spherical):
    REAL xx0 = _xx0[i0];
    REAL xx1 = _xx1[i1];
    REAL xx2 = _xx2[i2];
    /*
     *  Original SymPy expressions:
     *  "[xCart_from_xx = xx0*sin(xx1)*cos(xx2)]"
     *  "[yCart_from_xx = xx0*sin(xx1)*sin(xx2)]"
     *  "[zCart_from_xx = xx0*cos(xx1)]"
     */
    const REAL tmp0 = xx0 * sin(xx1);
    xCart_from_xx = tmp0 * cos(xx2);
    yCart_from_xx = tmp0 * sin(xx2);
    zCart_from_xx = xx0 * cos(xx1);
  }

  // Step 3.b: Compute {x,y,z}Cart_from_xx_inbounds, as a
  //           function of i0_inbounds,i1_inbounds,i2_inbounds
  [[maybe_unused]] REAL xCart_from_xx_inbounds, yCart_from_xx_inbounds, zCart_from_xx_inbounds;
  {
    // xx_to_Cart_inbounds for Coordinate Spherical):
    REAL xx0 = _xx0[i0_inbounds];
    REAL xx1 = _xx1[i1_inbounds];
    REAL xx2 = _xx2[i2_inbounds];

    /*
     *  Original SymPy expressions:
     *  "[xCart_from_xx_inbounds = xx0*sin(xx1)*cos(xx2)]"
     *  "[yCart_from_xx_inbounds = xx0*sin(xx1)*sin(xx2)]"
     *  "[zCart_from_xx_inbounds = xx0*cos(xx1)]"
     */
    const REAL tmp0 = xx0 * sin(xx1);
    xCart_from_xx_inbounds = tmp0 * cos(xx2);
    yCart_from_xx_inbounds = tmp0 * sin(xx2);
    zCart_from_xx_inbounds = xx0 * cos(xx1);
  }

  // Step 3.c: Compare xCart_from_xx to xCart_from_xx_inbounds;
  //           they should be identical!!!
#define EPS_REL 1e-8
  const REAL norm_factor = sqrt(xCart_from_xx * xCart_from_xx + yCart_from_xx * yCart_from_xx + zCart_from_xx * zCart_from_xx) + 1e-15;
  // if (fabs((double)(xCart_from_xx - xCart_from_xx_inbounds)) > EPS_REL * norm_factor ||
  //     fabs((double)(yCart_from_xx - yCart_from_xx_inbounds)) > EPS_REL * norm_factor ||
  //     fabs((double)(zCart_from_xx - zCart_from_xx_inbounds)) > EPS_REL * norm_factor) {
  //   fprintf(stderr,
  //           "Error in Spherical coordinate system: Inner boundary point does not map to grid interior point: ( %.15e %.15e %.15e ) != ( %.15e %.15e "
  //           "%.15e ) | xx: %e %e %e -> %e %e %e | %d %d %d\n",
  //           (double)xCart_from_xx, (double)yCart_from_xx, (double)zCart_from_xx, (double)xCart_from_xx_inbounds, (double)yCart_from_xx_inbounds,
  //           (double)zCart_from_xx_inbounds, _xx0[i0], _xx1[i1], _xx2[i2], _xx0[i0_inbounds], _xx1[i1_inbounds], _xx2[i2_inbounds],
  //           Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2);
  //   exit(1);
  // }

  // Step 4: Set output arrays.
  x0x1x2_inbounds[0] = _xx0[i0_inbounds];
  x0x1x2_inbounds[1] = _xx1[i1_inbounds];
  x0x1x2_inbounds[2] = _xx2[i2_inbounds];
  i0i1i2_inbounds[0] = i0_inbounds;
  i0i1i2_inbounds[1] = i1_inbounds;
  i0i1i2_inbounds[2] = i2_inbounds;
}
/*
 * set_parity_for_inner_boundary_single_pt():
 * Given (x0,x1,x2)=(xx0,xx1,xx2) and
 * (x0,x1,x2)'=(x0x1x2_inbounds[0],x0x1x2_inbounds[1],x0x1x2_inbounds[2])
 * (see description of
 * EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt()
 * above for more details), here we compute the parity conditions
 * for all 10 tensor types supported by NRPy+.
 */
__device__
void set_parity_for_inner_boundary_single_pt(const REAL xx0, const REAL xx1, const REAL xx2, const REAL x0x1x2_inbounds[3], const int idx,
                                                    innerpt_bc_struct *restrict innerpt_bc_arr) {

  const REAL xx0_inbounds = x0x1x2_inbounds[0];
  const REAL xx1_inbounds = x0x1x2_inbounds[1];
  const REAL xx2_inbounds = x0x1x2_inbounds[2];

  REAL REAL_parity_array[10];
  {
    // Evaluate dot products needed for setting parity
    //     conditions at a given point (xx0,xx1,xx2),
    //     using C code generated by NRPy+
    /*
    NRPy+ Curvilinear Boundary Conditions: Unit vector dot products for all
         ten parity conditions, in given coordinate system.
         Needed for automatically determining sign of tensor across coordinate boundary.
    Documented in: Tutorial-Start_to_Finish-Curvilinear_BCs.ipynb
    */
    /*
     *  Original SymPy expressions:
     *  "[REAL_parity_array[0] = 1]"
     *  "[REAL_parity_array[1] = sin(xx1)*sin(xx1_inbounds)*sin(xx2)*sin(xx2_inbounds) + sin(xx1)*sin(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds) +
     * cos(xx1)*cos(xx1_inbounds)]"
     *  "[REAL_parity_array[2] = sin(xx1)*sin(xx1_inbounds) + sin(xx2)*sin(xx2_inbounds)*cos(xx1)*cos(xx1_inbounds) +
     * cos(xx1)*cos(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds)]"
     *  "[REAL_parity_array[3] = sin(xx2)*sin(xx2_inbounds) + cos(xx2)*cos(xx2_inbounds)]"
     *  "[REAL_parity_array[4] = (sin(xx1)*sin(xx1_inbounds)*sin(xx2)*sin(xx2_inbounds) + sin(xx1)*sin(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds) +
     * cos(xx1)*cos(xx1_inbounds))**2]"
     *  "[REAL_parity_array[5] = (sin(xx1)*sin(xx1_inbounds) + sin(xx2)*sin(xx2_inbounds)*cos(xx1)*cos(xx1_inbounds) +
     * cos(xx1)*cos(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds))*(sin(xx1)*sin(xx1_inbounds)*sin(xx2)*sin(xx2_inbounds) +
     * sin(xx1)*sin(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds) + cos(xx1)*cos(xx1_inbounds))]"
     *  "[REAL_parity_array[6] = (sin(xx2)*sin(xx2_inbounds) + cos(xx2)*cos(xx2_inbounds))*(sin(xx1)*sin(xx1_inbounds)*sin(xx2)*sin(xx2_inbounds) +
     * sin(xx1)*sin(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds) + cos(xx1)*cos(xx1_inbounds))]"
     *  "[REAL_parity_array[7] = (sin(xx1)*sin(xx1_inbounds) + sin(xx2)*sin(xx2_inbounds)*cos(xx1)*cos(xx1_inbounds) +
     * cos(xx1)*cos(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds))**2]"
     *  "[REAL_parity_array[8] = (sin(xx2)*sin(xx2_inbounds) + cos(xx2)*cos(xx2_inbounds))*(sin(xx1)*sin(xx1_inbounds) +
     * sin(xx2)*sin(xx2_inbounds)*cos(xx1)*cos(xx1_inbounds) + cos(xx1)*cos(xx1_inbounds)*cos(xx2)*cos(xx2_inbounds))]"
     *  "[REAL_parity_array[9] = (sin(xx2)*sin(xx2_inbounds) + cos(xx2)*cos(xx2_inbounds))**2]"
     */
    {
      const REAL tmp0 = cos(xx1) * cos(xx1_inbounds);
      const REAL tmp1 = sin(xx1) * sin(xx1_inbounds);
      const REAL tmp2 = sin(xx2) * sin(xx2_inbounds);
      const REAL tmp3 = cos(xx2) * cos(xx2_inbounds);
      const REAL tmp4 = tmp0 + tmp1 * tmp2 + tmp1 * tmp3;
      const REAL tmp5 = tmp0 * tmp2 + tmp0 * tmp3 + tmp1;
      const REAL tmp6 = tmp2 + tmp3;
      REAL_parity_array[0] = 1;
      REAL_parity_array[1] = tmp4;
      REAL_parity_array[2] = tmp5;
      REAL_parity_array[3] = tmp6;
      REAL_parity_array[4] = ((tmp4) * (tmp4));
      REAL_parity_array[5] = tmp4 * tmp5;
      REAL_parity_array[6] = tmp4 * tmp6;
      REAL_parity_array[7] = ((tmp5) * (tmp5));
      REAL_parity_array[8] = tmp5 * tmp6;
      REAL_parity_array[9] = ((tmp6) * (tmp6));
    }
  }

  // Next perform sanity check on parity array output: should be +1 or -1 to within 8 significant digits:
  for (int whichparity = 0; whichparity < 10; whichparity++) {
    if (fabs(REAL_parity_array[whichparity]) < 1 - 1e-8 || fabs(REAL_parity_array[whichparity]) > 1 + 1e-8) {
      printf("Error at point (%e %e %e), which maps to (%e %e %e).\n", xx0, xx1, xx2, xx0_inbounds, xx1_inbounds, xx2_inbounds);
      printf("Parity evaluated to %e , which is not within 8 significant digits of +1 or -1.\n", REAL_parity_array[whichparity]);
    }
    for (int parity = 0; parity < 10; parity++) {
      innerpt_bc_arr[idx].parity[parity] = 1;
      if (REAL_parity_array[parity] < 0)
        innerpt_bc_arr[idx].parity[parity] = -1;
    }
  } // END for(int whichparity=0;whichparity<10;whichparity++)
}

__global__
void count_ib_points(uint * n_ib, REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2) {
    
    // shared data between all warps
    // Assumes one block = 32 warps = 32 * 32 threads
    // As of today, the standard maximum threads per
    // block is 1024 = 32 * 32
    __shared__ uint shared_data[2][32];

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

    // thread index
    uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    // reduce excessive looping by computing both at the same time
    uint local_ib_points = 0;

    // // warp mask - says all threads are involved in shuffle
    // // 0xFFFFFFFFU in binary is 32 1's.
    unsigned mask = 0xFFFFFFFFU;

    // lane = which thread am I in the warp
    uint lane = tid % warpSize;
    // warpID = which warp am I in the block
    uint warpID = tid / warpSize;
    
    // Loop over bounds on both sides of x/y/z, at the same time
    int i0i1i2[3];
    for(size_t i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
      for(size_t i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
        for(size_t i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
          // Initialize
          REAL x0x1x2_inbounds[3] = {0,0,0};
          int i0i1i2_inbounds[3] = {0,0,0};
          // Assign lower ghost zone boundary points
          i0i1i2[0]=i0; i0i1i2[1]=i1; i0i1i2[2]=i2;
          bool is_in_interior = IS_IN_GRID_INTERIOR(i0i1i2, Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2, NGHOSTS);
          if(!is_in_interior) {
            EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(
              _xx0, _xx1, _xx2, i0, i1, i2, x0x1x2_inbounds, i0i1i2_inbounds);
            bool pure_boundary_point = \
              (i0 == i0i1i2_inbounds[0]) && \
              (i1 == i0i1i2_inbounds[1]) && \
              (i2 == i0i1i2_inbounds[2]);
            if(!pure_boundary_point) {
              local_ib_points++;
            }
          }
        }
      }
    }

    // Shuffle down kernel
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
        uint shfl_ib = __shfl_down_sync(mask, local_ib_points, offset);
        local_ib_points += shfl_ib;
    }
    // Shuffle results in lane 0 have the shuffle result
    if(lane == 0) {
        shared_data[0][warpID] = local_ib_points;
    }
    
    // Make sure all warps in the block are syncronized
    __syncthreads();
    // Since there is only 32 partial reductions, we only
    // have one warp worth of work
    if(warpID == 0) {
        // Check to make sure we had 32 blocks of data
        if(tid < blockDim.x / warpSize) {
            local_ib_points = shared_data[0][lane];
        } else {
            local_ib_points = 0;
        }        
        
        // Shuffle down kernel
        for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
            uint shfl_ib = __shfl_down_sync(mask, local_ib_points, offset);
            local_ib_points += shfl_ib;
        }
        if(tid == 0) {
            atomicAdd(n_ib, local_ib_points);
        }
    }
}

__host__
[[nodiscard]] uint compute_num_inner(REAL * xx[3], const params_struct * params) {
  
  uint num_inner=0;
  uint* num_inner_gpu;
  cudaMalloc(&num_inner_gpu, sizeof(uint));
  cudaCheckErrors(countMalloc, "memory failure")
  cudaMemcpy(num_inner_gpu, &num_inner, sizeof(uint), cudaMemcpyHostToDevice);
  cudaCheckErrors(cpy, "memory failure")

  // We're only interested in the ghost zones
  // size_t total_ghosts = (2. * NGHOSTS);
  int const & Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const & Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const & Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  size_t N = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  size_t block_threads = MIN(MAX(N,32), 1024)/2;
  size_t grids = (N + block_threads - 1)/block_threads;
  
  count_ib_points<<<grids, block_threads>>>(num_inner_gpu, xx[0], xx[1], xx[2]);
  cudaCheckErrors(count_ib_points, "kernel failure")
  cudaMemcpy(&num_inner, num_inner_gpu, sizeof(uint), cudaMemcpyDeviceToHost);
  cudaCheckErrors(cudaMemcpy, "copy failure")
  cudaFree(num_inner_gpu);
  return num_inner;
}

__global__
void set_pure_outer_bc_array_gpu(int const which_gz, uint * idx2d, 
  outerpt_bc_struct *  pure_outer_bc_array,
    REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2,
      int i0min, int i0max, int i1min, int i1max, int i2min, int i2max,
          const int face) {

    const int FACEX0 = (face == 0) - (face == 1); // +1 if face==0 ; -1 if face==1. Otherwise 0.
    const int FACEX1 = (face == 2) - (face == 3); // +1 if face==2 ; -1 if face==3. Otherwise 0.
    const int FACEX2 = (face == 4) - (face == 5); // +1 if face==4 ; -1 if face==5. Otherwise 0.

    // Global data index - expecting a 1D dataset
    // Thread indices
    const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
    const int tid1 = 0; //threadIdx.y + blockIdx.y*blockDim.y;
    const int tid2 = 0; //threadIdx.z + blockIdx.z*blockDim.z;
    // Thread strides
    const int stride0 = blockDim.x * gridDim.x;
    const int stride1 = 1; //blockDim.y * gridDim.y;
    const int stride2 = 1; //blockDim.z * gridDim.z;

    for(size_t i2 = tid2+i2min; i2 < i2max; i2 += stride2) {
      for(size_t i1 = tid1+i1min; i1 < i1max; i1 += stride1) {
        for(size_t i0 = tid0+i0min; i0 < i0max; i0 += stride0) {
          // Initialize
          REAL x0x1x2_inbounds[3] = {0,0,0};
          int i0i1i2_inbounds[3] = {0,0,0};
                          
          EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(
            _xx0, _xx1, _xx2, i0, i1, i2, x0x1x2_inbounds, i0i1i2_inbounds);
          bool pure_boundary_point = \
            (i0 == i0i1i2_inbounds[0]) && \
            (i1 == i0i1i2_inbounds[1]) && \
            (i2 == i0i1i2_inbounds[2]);
          if(pure_boundary_point) {
            int const idx = *idx2d;
            pure_outer_bc_array[idx].i0 = i0;
            pure_outer_bc_array[idx].i1 = i1;
            pure_outer_bc_array[idx].i2 = i2;
            pure_outer_bc_array[idx].FACEX0 = FACEX0;
            pure_outer_bc_array[idx].FACEX1 = FACEX1;
            pure_outer_bc_array[idx].FACEX2 = FACEX2;
            *idx2d += 1;
          }
        }
      }
    }
}

__host__
void set_pure_outer_bc_array(REAL * xx[3], bc_struct *restrict bcstruct) {
  uint* idx2d_gpu;
  cudaMalloc(&idx2d_gpu, sizeof(uint));
  cudaCheckErrors(countMalloc, "memory failure")
  for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
    for (int dirn = 0; dirn < 3; dirn++) {
      int idx2d = 0;
      cudaMemcpy(idx2d_gpu, &idx2d, sizeof(uint), cudaMemcpyHostToDevice);
      cudaCheckErrors(cpy, "memory failure")

      {
        const int face = dirn * 2;
        set_pure_outer_bc_array_gpu<<<1,1>>>(which_gz, idx2d_gpu, bcstruct->pure_outer_bc_array[dirn + (3 * which_gz)],
            xx[0], xx[1], xx[2], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][0], bcstruct->bc_info.bc_loop_bounds[which_gz][face][1], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][2], bcstruct->bc_info.bc_loop_bounds[which_gz][face][3], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][4], bcstruct->bc_info.bc_loop_bounds[which_gz][face][5], 
            face);
        cudaCheckErrors(set_pure_outer_bc_array_gpu, "kernel failure")
      }
      // UPPER FACE: dirn=0 -> x0max; dirn=1 -> x1max; dirn=2 -> x2max
      {
        const int face = dirn * 2 + 1;
        set_pure_outer_bc_array_gpu<<<1,1>>>(which_gz, idx2d_gpu, bcstruct->pure_outer_bc_array[dirn + (3 * which_gz)],
            xx[0], xx[1], xx[2], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][0], bcstruct->bc_info.bc_loop_bounds[which_gz][face][1], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][2], bcstruct->bc_info.bc_loop_bounds[which_gz][face][3], 
            bcstruct->bc_info.bc_loop_bounds[which_gz][face][4], bcstruct->bc_info.bc_loop_bounds[which_gz][face][5], 
            face);
        cudaCheckErrors(set_pure_outer_bc_array_gpu, "kernel failure")
      }
      cudaMemcpy(&idx2d, idx2d_gpu, sizeof(uint), cudaMemcpyDeviceToHost);
      cudaCheckErrors(cudaMemcpy, "copy failure")
      bcstruct->bc_info.num_pure_outer_boundary_points[which_gz][dirn] = idx2d;
    }
  }
  cudaFree(idx2d_gpu);
}

__global__
void set_inner_bc_array(innerpt_bc_struct *restrict inner_bc_array, REAL *restrict _xx0, REAL *restrict _xx1, REAL *restrict _xx2){

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
    
    uint which_inner = 0;
    int i0i1i2[3];
    for(size_t i2 = tid2; i2 < Nxx_plus_2NGHOSTS2; i2 += stride2) {
      for(size_t i1 = tid1; i1 < Nxx_plus_2NGHOSTS1; i1 += stride1) {
        for(size_t i0 = tid0; i0 < Nxx_plus_2NGHOSTS0; i0 += stride0) {
          // Assign lower ghost zone boundary points
          i0i1i2[0]=i0; i0i1i2[1]=i1; i0i1i2[2]=i2;
          bool is_in_interior = IS_IN_GRID_INTERIOR(i0i1i2, Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1, Nxx_plus_2NGHOSTS2, NGHOSTS);
          if(!is_in_interior) {
            // Initialize
            REAL x0x1x2_inbounds[3] = {0,0,0};
            int i0i1i2_inbounds[3] = {0,0,0};
            EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(
              _xx0, _xx1, _xx2, i0, i1, i2, x0x1x2_inbounds, i0i1i2_inbounds);
            bool pure_boundary_point = \
              (i0 == i0i1i2_inbounds[0]) && \
              (i1 == i0i1i2_inbounds[1]) && \
              (i2 == i0i1i2_inbounds[2]);
            if(!pure_boundary_point) {
              inner_bc_array[which_inner].dstpt = IDX3(i0, i1, i2);
              inner_bc_array[which_inner].srcpt = IDX3(i0i1i2_inbounds[0], i0i1i2_inbounds[1], i0i1i2_inbounds[2]);
              set_parity_for_inner_boundary_single_pt(_xx0[i0], _xx1[i1], _xx2[i2], x0x1x2_inbounds, which_inner, inner_bc_array);
              which_inner++;
            }
          }
        }
      }
    }
}

/*
 * At each coordinate point (x0,x1,x2) situated at grid index (i0,i1,i2):
 * Step 1: Set up inner boundary structs bcstruct->inner_bc_array[].
 * Recall that at each inner boundary point we must set innerpt_bc_struct:
 * typedef struct __innerpt_bc_struct__ {
 * int dstpt;  // dstpt is the 3D grid index IDX3(i0,i1,i2) of the inner boundary point (i0,i1,i2)
 * int srcpt;  // srcpt is the 3D grid index (a la IDX3) to which the inner boundary point maps
 * int8_t parity[10];  // parity[10] is a calculation of dot products for the 10 independent parity types
 * } innerpt_bc_struct;
 * At each ghostzone (i.e., each point within NGHOSTS points from grid boundary):
 * Call EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt().
 * This function converts the curvilinear coordinate (x0,x1,x2) to the corresponding
 * Cartesian coordinate (x,y,z), then finds the grid point
 * (i0_inbounds,i1_inbounds,i2_inbounds) in the grid interior or outer boundary
 * corresponding to this Cartesian coordinate (x,y,z).
 * If (i0,i1,i2) *is not* the same as (i0_inbounds,i1_inbounds,i2_inbounds),
 * then we are at an inner boundary point. We must set
 * Set bcstruct->inner_bc_array for this point, which requires we specify
 * both (i0_inbounds,i1_inbounds,i2_inbounds) [just found!] and parity
 * conditions for this gridpoint. The latter is found & specified within the
 * function set_parity_for_inner_boundary_single_pt().
 * If (i0,i1,i2) *is* the same as (i0_inbounds,i1_inbounds,i2_inbounds),
 * then we are at an outer boundary point. Take care of outer BCs in Step 2.
 * Step 2: Set up outer boundary structs bcstruct->outer_bc_array[which_gz][face][idx2d]:
 * Recall that at each inner boundary point we must set outerpt_bc_struct:
 * typedef struct __outerpt_bc_struct__ {
 * short i0,i1,i2;  // the outer boundary point grid index (i0,i1,i2), on the 3D grid
 * int8_t FACEX0,FACEX1,FACEX2;  // 1-byte integers that store
 * //                               FACEX0,FACEX1,FACEX2 = +1, 0, 0 if on the i0=i0min face,
 * //                               FACEX0,FACEX1,FACEX2 = -1, 0, 0 if on the i0=i0max face,
 * //                               FACEX0,FACEX1,FACEX2 =  0,+1, 0 if on the i1=i2min face,
 * //                               FACEX0,FACEX1,FACEX2 =  0,-1, 0 if on the i1=i1max face,
 * //                               FACEX0,FACEX1,FACEX2 =  0, 0,+1 if on the i2=i2min face, or
 * //                               FACEX0,FACEX1,FACEX2 =  0, 0,-1 if on the i2=i2max face,
 * } outerpt_bc_struct;
 * Outer boundary points are filled from the inside out, two faces at a time.
 * E.g., consider a Cartesian coordinate grid that has 14 points in each direction,
 * including the ghostzones, with NGHOSTS=2.
 * We first fill in the lower x0 face with (i0=1,i1={2,11},i2={2,11}). We fill these
 * points in first, since they will in general (at least in the case of extrapolation
 * outer BCs) depend on e.g., i0=2 and i0=3 points.
 * Simultaneously we can fill in the upper x0 face with (i0=12,i1={2,11},i2={2,11}),
 * since these points depend only on e.g., i0=11 and i0=10 (again assuming extrap. BCs).
 * Next we can fill in the lower x1 face: (i0={1,12},i1=2,i2={2,11}). Notice these
 * depend on i0 min and max faces being filled. The remaining pattern goes like this:
 * Upper x1 face: (i0={1,12},i1=12,i2={2,11})
 * Lower x2 face: (i0={1,12},i1={1,12},i2=1)
 * Upper x2 face: (i0={1,12},i1={1,12},i2=12)
 * Lower x0 face: (i0=0,i1={1,12},i2={1,12})
 * Upper x0 face: (i0=13,i1={1,12},i2={1,12})
 * Lower x1 face: (i0={0,13},i1=0,i2={2,11})
 * Upper x1 face: (i0={0,13},i1=13,i2={2,11})
 * Lower x2 face: (i0={0,13},i1={0,13},i2=0)
 * Upper x2 face: (i0={0,13},i1={0,13},i2=13)
 * Note that we allocate a outerpt_bc_struct at *all* boundary points,
 * regardless of whether the point is an outer or inner point. However
 * the struct is set only at outer boundary points. This is slightly
 * wasteful, but only in memory, not in CPU.
 */
void bcstruct_set_up__rfm__Spherical(const commondata_struct *restrict commondata, const params_struct * params, REAL * xx[3], bc_struct *restrict bcstruct) {

  cudaDeviceSynchronize();
  
  ////////////////////////////////////////
  // STEP 1: SET UP INNER BOUNDARY STRUCTS
  {
    // Get number of inner boundary points
    uint num_inner = compute_num_inner(xx, params);
    // Allocate storage for mapping
    bcstruct->bc_info.num_inner_boundary_points = num_inner;
    cudaMalloc(&bcstruct->inner_bc_array, sizeof(innerpt_bc_struct) * num_inner);
    cudaCheckErrors(cudaMalloc, "memory failure")
    // Fill inner_bc_array mapping
    set_inner_bc_array<<<1,1>>>(bcstruct->inner_bc_array,xx[0], xx[1], xx[2]);
    cudaCheckErrors(set_inner_bc_array, "kernel failure")
  }
  ////////////////////////////////////////
  // STEP 2: SET UP OUTER BOUNDARY STRUCTS
  // First set up loop bounds for outer boundary condition updates,
  //   store to bc_info->bc_loop_bounds[which_gz][face][]. Also
  //   allocate memory for outer_bc_array[which_gz][face][]:
  int const& Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  int const& Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  int const& Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  
  int imin[3] = {NGHOSTS, NGHOSTS, NGHOSTS};
  int imax[3] = {Nxx_plus_2NGHOSTS0 - NGHOSTS, Nxx_plus_2NGHOSTS1 - NGHOSTS, Nxx_plus_2NGHOSTS2 - NGHOSTS};
for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
    const int x0min_face_range[6] = {imin[0] - 1, imin[0], imin[1], imax[1], imin[2], imax[2]};
    imin[0]--;
    const int x0max_face_range[6] = {imax[0], imax[0] + 1, imin[1], imax[1], imin[2], imax[2]};
    imax[0]++;
    const int x1min_face_range[6] = {imin[0], imax[0], imin[1] - 1, imin[1], imin[2], imax[2]};
    imin[1]--;
    const int x1max_face_range[6] = {imin[0], imax[0], imax[1], imax[1] + 1, imin[2], imax[2]};
    imax[1]++;
    const int x2min_face_range[6] = {imin[0], imax[0], imin[1], imax[1], imin[2] - 1, imin[2]};
    imin[2]--;
    const int x2max_face_range[6] = {imin[0], imax[0], imin[1], imax[1], imax[2], imax[2] + 1};
    imax[2]++;

    int face = 0;
    ////////////////////////
    // x0min and x0max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x0min and x0max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMalloc(&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2], 
               sizeof(outerpt_bc_struct) * 2 * (
                (x0min_face_range[1] - x0min_face_range[0]) * 
                (x0min_face_range[3] - x0min_face_range[2]) * 
                (x0min_face_range[5] - x0min_face_range[4])
    ));
    // x0min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x0min_face_range[i];
    }
    face++;
    // x0max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x0max_face_range[i];
    }
    face++;
    ////////////////////////

    ////////////////////////
    // x1min and x1max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x1min and x1max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMalloc(&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2],
               sizeof(outerpt_bc_struct) * 2 * (
                (x1min_face_range[1] - x1min_face_range[0]) * 
                (x1min_face_range[3] - x1min_face_range[2]) * 
                (x1min_face_range[5] - x1min_face_range[4])
    ));
    // x1min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x1min_face_range[i];
    }
    face++;
    // x1max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x1max_face_range[i];
    }
    face++;
    ////////////////////////

    ////////////////////////
    // x2min and x2max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x2min and x2max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMalloc(&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2],
               sizeof(outerpt_bc_struct) * 2 * (
                (x2min_face_range[1] - x2min_face_range[0]) * 
                (x2min_face_range[3] - x2min_face_range[2]) * 
                (x2min_face_range[5] - x2min_face_range[4])
    ));
    // x2min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x2min_face_range[i];
    }
    face++;
    // x2max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x2max_face_range[i];
    }
    face++;
    ////////////////////////
  }
  set_pure_outer_bc_array(xx, bcstruct);
}