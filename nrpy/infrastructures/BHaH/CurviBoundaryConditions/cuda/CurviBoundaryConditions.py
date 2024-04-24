"""
Module providing functions for setting up Curvilinear boundary conditions for CUDA codes


Authors: Zachariah B. Etienne
         zachetie **at** gmail **dot** com
         Terrence Pierre Jacques
         Samuel D. Tootle
         sdtootle **at** gmail **dot** com
"""

# Step P1: Import needed NRPy+ core modules:
from typing import List
import nrpy.c_function as cfc
import nrpy.params as par  # NRPy+: Parameter interface
from nrpy.infrastructures.BHaH import griddata_commondata
from nrpy.infrastructures.BHaH import BHaH_defines_h
import nrpy.infrastructures.BHaH.CurviBoundaryConditions.base_CurviBoundaryConditions as base_cbc_classes
import nrpy.helpers.gpu_kernel as gputils

_ = par.CodeParameter(
    "char[50]", __name__, "outer_bc_type", "radiation", commondata=True
)

# Update core_modules to use correct key for ordering
for i, key in enumerate(BHaH_defines_h.core_modules_list):
    if "nrpy.infrastructures.BHaH.CurviBoundaryConditions" in key:
        BHaH_defines_h.core_modules_list[i] = (
            str(__name__)
        )


# bcstruct_set_up():
#      This function is documented in desc= and body= fields below.
class register_CFunction_bcstruct_set_up(
    base_cbc_classes.base_register_CFunction_bcstruct_set_up
):
    """
    Register C function for setting up bcstruct.

    This function prescribes how inner and outer boundary points on the
    computational grid are filled, based on the given coordinate system (CoordSystem).

    :param CoordSystem: The coordinate system for which to set up boundary conditions.
    :param fp_type: Floating point type, e.g., "double".
    """

    def __init__(self, CoordSystem: str, fp_type: str = "double") -> None:
        super().__init__(CoordSystem, fp_type=fp_type)
        self.params = "const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], bc_struct *restrict bcstruct_gpu"
        self.body = r"""bc_struct* bcstruct = new bc_struct;
  ////////////////////////////////////////
  // STEP 1: SET UP INNER BOUNDARY STRUCTS
  {
    // First count the number of inner points.
    int num_inner = 0;
    LOOP_OMP("omp parallel for reduction(+:num_inner)",
             i0,0,Nxx_plus_2NGHOSTS0,  i1,0,Nxx_plus_2NGHOSTS1,  i2,0,Nxx_plus_2NGHOSTS2) {
      const int i0i1i2[3] = { i0,i1,i2 };
      if(!IS_IN_GRID_INTERIOR(i0i1i2, Nxx_plus_2NGHOSTS0,Nxx_plus_2NGHOSTS1,Nxx_plus_2NGHOSTS2, NGHOSTS)) {
        REAL x0x1x2_inbounds[3];
        int i0i1i2_inbounds[3];
        EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(commondata, params, xx, i0,i1,i2, x0x1x2_inbounds,i0i1i2_inbounds);
        if(i0 == i0i1i2_inbounds[0] && i1==i0i1i2_inbounds[1] && i2==i0i1i2_inbounds[2]) {
          // this is a pure outer boundary point.
        } else {
          // this is an inner boundary point, which maps either
          //  to the grid interior or to an outer boundary point
          num_inner++;
        }
      }
    }
    // Store num_inner to bc_info:
    bcstruct->bc_info.num_inner_boundary_points = num_inner;
    bcstruct_gpu->bc_info.num_inner_boundary_points = num_inner;

    // Next allocate memory for inner_boundary_points:
    cudaMallocHost((void**)&bcstruct->inner_bc_array, sizeof(innerpt_bc_struct) * num_inner);
    cudaCheckErrors(cudaMallocHost, "Pinned malloc inner_bc_array failed.");
    cudaMalloc(&bcstruct_gpu->inner_bc_array, sizeof(innerpt_bc_struct) * num_inner);
    cudaCheckErrors(cudaMalloc, "memory failure");
  }

  // Then set inner_bc_array using device bcstruct:
  {
    int which_inner = 0;
    LOOP_NOOMP(i0,0,Nxx_plus_2NGHOSTS0,  i1,0,Nxx_plus_2NGHOSTS1,  i2,0,Nxx_plus_2NGHOSTS2) {
      const int i0i1i2[3] = { i0,i1,i2 };
      if(!IS_IN_GRID_INTERIOR(i0i1i2, Nxx_plus_2NGHOSTS0,Nxx_plus_2NGHOSTS1,Nxx_plus_2NGHOSTS2, NGHOSTS)) {
        REAL x0x1x2_inbounds[3];
        int i0i1i2_inbounds[3];
        EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(commondata, params, xx, i0,i1,i2, x0x1x2_inbounds,i0i1i2_inbounds);
        if(i0 == i0i1i2_inbounds[0] && i1==i0i1i2_inbounds[1] && i2==i0i1i2_inbounds[2]) {
          // this is a pure outer boundary point.
        } else {
          bcstruct->inner_bc_array[which_inner].dstpt = IDX3(i0,i1,i2);
          bcstruct->inner_bc_array[which_inner].srcpt = IDX3(i0i1i2_inbounds[0],i0i1i2_inbounds[1],i0i1i2_inbounds[2]);
          //printf("%d / %d\n",which_inner, bc_info->num_inner_boundary_points);
          set_parity_for_inner_boundary_single_pt(commondata, params, xx[0][i0],xx[1][i1],xx[2][i2],
                                                  x0x1x2_inbounds, which_inner, bcstruct->inner_bc_array);

          which_inner++;
        }
      }
    }
  }

  // Asynchronously copy data to device
  cudaMemcpyAsync(bcstruct_gpu->inner_bc_array, 
                  bcstruct->inner_bc_array, 
                  sizeof(innerpt_bc_struct) * bcstruct->bc_info.num_inner_boundary_points, 
                  cudaMemcpyHostToDevice, streams[nstreams-1]);
  cudaCheckErrors(cudaMemcpy, "Memcpy failed - inner_bc_array");

  ////////////////////////////////////////
  // STEP 2: SET UP OUTER BOUNDARY STRUCTS
  // First set up loop bounds for outer boundary condition updates,
  //   store to bc_info->bc_loop_bounds[which_gz][face][]. Also
  //   allocate memory for outer_bc_array[which_gz][face][]:
  int imin[3] = { NGHOSTS, NGHOSTS, NGHOSTS };
  int imax[3] = { Nxx_plus_2NGHOSTS0-NGHOSTS, Nxx_plus_2NGHOSTS1-NGHOSTS, Nxx_plus_2NGHOSTS2-NGHOSTS };
  for(int which_gz=0;which_gz<NGHOSTS;which_gz++) {
    const int x0min_face_range[6] = { imin[0]-1,imin[0], imin[1],imax[1], imin[2],imax[2] };  imin[0]--;
    const int x0max_face_range[6] = { imax[0],imax[0]+1, imin[1],imax[1], imin[2],imax[2] };  imax[0]++;
    const int x1min_face_range[6] = { imin[0],imax[0], imin[1]-1,imin[1], imin[2],imax[2] };  imin[1]--;
    const int x1max_face_range[6] = { imin[0],imax[0], imax[1],imax[1]+1, imin[2],imax[2] };  imax[1]++;
    const int x2min_face_range[6] = { imin[0],imax[0], imin[1],imax[1], imin[2]-1,imin[2] };  imin[2]--;
    const int x2max_face_range[6] = { imin[0],imax[0], imin[1],imax[1], imax[2],imax[2]+1 };  imax[2]++;

    int face=0;
    ////////////////////////
    // x0min and x0max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x0min and x0max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMallocHost((void**)&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2], 
                  sizeof(outerpt_bc_struct) * 2 * (
                (x0min_face_range[1] - x0min_face_range[0]) * 
                (x0min_face_range[3] - x0min_face_range[2]) * 
                (x0min_face_range[5] - x0min_face_range[4])));
    cudaCheckErrors(cudaMallocHost, "Pinned malloc pure_outer_bc_array failed.");
    cudaMalloc(&bcstruct_gpu->pure_outer_bc_array[3 * which_gz + face / 2], 
               sizeof(outerpt_bc_struct) * 2 * (
                (x0min_face_range[1] - x0min_face_range[0]) * 
                (x0min_face_range[3] - x0min_face_range[2]) * 
                (x0min_face_range[5] - x0min_face_range[4])
    ));
    
    // x0min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x0min_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x0min_face_range[i];
    }
    face++;
    // x0max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x0max_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x0max_face_range[i];
    }
    face++;
    ////////////////////////

    ////////////////////////
    // x1min and x1max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x1min and x1max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMallocHost((void**)&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2], 
                  sizeof(outerpt_bc_struct) * 2 * (
                (x1min_face_range[1] - x1min_face_range[0]) * 
                (x1min_face_range[3] - x1min_face_range[2]) * 
                (x1min_face_range[5] - x1min_face_range[4])));
    cudaCheckErrors(cudaMallocHost, "Pinned malloc pure_outer_bc_array failed.");
    cudaMalloc(&bcstruct_gpu->pure_outer_bc_array[3 * which_gz + face / 2], 
               sizeof(outerpt_bc_struct) * 2 * (
                (x1min_face_range[1] - x1min_face_range[0]) * 
                (x1min_face_range[3] - x1min_face_range[2]) * 
                (x1min_face_range[5] - x1min_face_range[4])
    ));
    // x1min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x1min_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x1min_face_range[i];
    }
    face++;
    // x1max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x1max_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x1max_face_range[i];
    }
    face++;
    ////////////////////////


    ////////////////////////
    // x2min and x2max faces: Allocate memory for outer_bc_array and set bc_loop_bounds:
    //                        Note that x2min and x2max faces have exactly the same size.
    //                   Also, note that face/2 --v   offsets this factor of 2 ------------------------------------------v
    cudaMallocHost((void**)&bcstruct->pure_outer_bc_array[3 * which_gz + face / 2], 
                  sizeof(outerpt_bc_struct) * 2 * (
                (x2min_face_range[1] - x2min_face_range[0]) * 
                (x2min_face_range[3] - x2min_face_range[2]) * 
                (x2min_face_range[5] - x2min_face_range[4])));
    cudaCheckErrors(cudaMallocHost, "Pinned malloc pure_outer_bc_array failed.");
    cudaMalloc(&bcstruct_gpu->pure_outer_bc_array[3 * which_gz + face / 2], 
               sizeof(outerpt_bc_struct) * 2 * (
                (x2min_face_range[1] - x2min_face_range[0]) * 
                (x2min_face_range[3] - x2min_face_range[2]) * 
                (x2min_face_range[5] - x2min_face_range[4])
    ));    
    // x2min face: Can't set bc_info->bc_loop_bounds[which_gz][face] = { i0min,i0max, ... } since it's not const :(
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x2min_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x2min_face_range[i];
    }
    face++;
    // x2max face: Set loop bounds & allocate memory for outer_bc_array:
    for (int i = 0; i < 6; i++) {
      bcstruct->bc_info.bc_loop_bounds[which_gz][face][i] = x2max_face_range[i];
      bcstruct_gpu->bc_info.bc_loop_bounds[which_gz][face][i] = x2max_face_range[i];
    }
    face++;
    ////////////////////////
  }

  for(int which_gz=0;which_gz<NGHOSTS;which_gz++) for(int dirn=0;dirn<3;dirn++) {
      int idx2d = 0;
      // LOWER FACE: dirn=0 -> x0min; dirn=1 -> x1min; dirn=2 -> x2min
      {
        const int face = dirn*2;
#define IDX2D_BCS(i0,i0min,i0max, i1,i1min,i1max ,i2,i2min,i2max)       \
        ( ((i0)-(i0min)) + ((i0max)-(i0min)) * ( ((i1)-(i1min)) + ((i1max)-(i1min)) * ((i2)-(i2min)) ) )
        const int FACEX0=(face==0) - (face==1); // +1 if face==0 (x0min) ; -1 if face==1 (x0max). Otherwise 0.
        const int FACEX1=(face==2) - (face==3); // +1 if face==2 (x1min) ; -1 if face==3 (x1max). Otherwise 0.
        const int FACEX2=(face==4) - (face==5); // +1 if face==4 (x2min) ; -1 if face==5 (x2max). Otherwise 0.
        LOOP_NOOMP(i0,bcstruct->bc_info.bc_loop_bounds[which_gz][face][0],bcstruct->bc_info.bc_loop_bounds[which_gz][face][1],
                   i1,bcstruct->bc_info.bc_loop_bounds[which_gz][face][2],bcstruct->bc_info.bc_loop_bounds[which_gz][face][3],
                   i2,bcstruct->bc_info.bc_loop_bounds[which_gz][face][4],bcstruct->bc_info.bc_loop_bounds[which_gz][face][5]) {
          REAL x0x1x2_inbounds[3];
          int i0i1i2_inbounds[3];
          EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(commondata, params, xx, i0,i1,i2, x0x1x2_inbounds,i0i1i2_inbounds);
          if(i0 == i0i1i2_inbounds[0] && i1==i0i1i2_inbounds[1] && i2==i0i1i2_inbounds[2]) {
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i0 = i0;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i1 = i1;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i2 = i2;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX0 = FACEX0;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX1 = FACEX1;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX2 = FACEX2;
            cpy_pure_outer_bc_array(bcstruct, bcstruct_gpu, dirn + (3 * which_gz), idx2d);
            idx2d++;
          }
        }
      }
      // UPPER FACE: dirn=0 -> x0max; dirn=1 -> x1max; dirn=2 -> x2max
      {
        const int face = dirn*2+1;
        const int FACEX0=(face==0) - (face==1); // +1 if face==0 ; -1 if face==1. Otherwise 0.
        const int FACEX1=(face==2) - (face==3); // +1 if face==2 ; -1 if face==3. Otherwise 0.
        const int FACEX2=(face==4) - (face==5); // +1 if face==4 ; -1 if face==5. Otherwise 0.
        LOOP_NOOMP(i0,bcstruct->bc_info.bc_loop_bounds[which_gz][face][0],bcstruct->bc_info.bc_loop_bounds[which_gz][face][1],
                   i1,bcstruct->bc_info.bc_loop_bounds[which_gz][face][2],bcstruct->bc_info.bc_loop_bounds[which_gz][face][3],
                   i2,bcstruct->bc_info.bc_loop_bounds[which_gz][face][4],bcstruct->bc_info.bc_loop_bounds[which_gz][face][5]) {
          REAL x0x1x2_inbounds[3];
          int i0i1i2_inbounds[3];
          EigenCoord_set_x0x1x2_inbounds__i0i1i2_inbounds_single_pt(commondata, params, xx, i0,i1,i2, x0x1x2_inbounds,i0i1i2_inbounds);
          if(i0 == i0i1i2_inbounds[0] && i1==i0i1i2_inbounds[1] && i2==i0i1i2_inbounds[2]) {
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i0 = i0;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i1 = i1;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i2 = i2;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX0 = FACEX0;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX1 = FACEX1;
            bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX2 = FACEX2;
            cpy_pure_outer_bc_array(bcstruct, bcstruct_gpu, dirn + (3 * which_gz), idx2d);
            idx2d++;
          }
        }
      }
      bcstruct->bc_info.num_pure_outer_boundary_points[which_gz][dirn] = idx2d;
      bcstruct_gpu->bc_info.num_pure_outer_boundary_points[which_gz][dirn] = idx2d;
    }
    cudaDeviceSynchronize();
    cudaFreeHost(bcstruct->inner_bc_array);
    for(int i = 0; i < NGHOSTS * 3; ++i)
      cudaFreeHost(bcstruct->pure_outer_bc_array[i]);
    delete bcstruct;
"""
        self.prefunc = """
static void cpy_pure_outer_bc_array(bc_struct *restrict bcstruct_h, bc_struct *restrict bcstruct_d,
  const int idx, const int idx2d) {
    const int streamid = idx2d % nstreams;
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].i0, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].i0,
          sizeof(short), 
          cudaMemcpyHostToDevice, streams[streamid]);
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array1");
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].i1, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].i1,
          sizeof(short), 
          cudaMemcpyHostToDevice, streams[streamid]);                  
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array2");
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].i2, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].i2,
          sizeof(short), 
          cudaMemcpyHostToDevice, streams[streamid]);
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array2");
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].FACEX0, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].FACEX0,
          sizeof(int8_t), 
          cudaMemcpyHostToDevice, streams[streamid]);
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array Face0");
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].FACEX1, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].FACEX1,
          sizeof(int8_t), 
          cudaMemcpyHostToDevice, streams[streamid]);
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array Face1");
    cudaMemcpyAsync(
          &bcstruct_d->pure_outer_bc_array[idx][idx2d].FACEX2, 
          &bcstruct_h->pure_outer_bc_array[idx][idx2d].FACEX2,
          sizeof(int8_t), 
          cudaMemcpyHostToDevice, streams[streamid]);
    cudaCheckErrors(cudaMemcpy, "Memcpy failed - pure_outer_bc_array Face2");
  }
""" + self.prefunc
        cfc.register_CFunction(
            includes=self.includes,
            prefunc=self.prefunc,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            CoordSystem_for_wrapper_func=self.CoordSystem,
            name=self.name,
            params=self.params,
            include_CodeParameters_h=True,
            body=self.body,
        )


###############################
## apply_bcs_inner_only(): Apply inner boundary conditions.
##  Function is documented below in desc= and body=.
class register_CFunction_apply_bcs_inner_only(
    base_cbc_classes.base_register_CFunction_apply_bcs_inner_only
):
    """Register C function for filling inner boundary points on the computational grid."""

    def __init__(self) -> None:
        super().__init__()

        self.body = r"""
  // Unpack bc_info from bcstruct
  const bc_info_struct *bc_info = &bcstruct->bc_info;

  // collapse(2) results in a nice speedup here, esp in 2D. Two_BHs_collide goes from
  //    5550 M/hr to 7264 M/hr on a Ryzen 9 5950X running on all 16 cores with core affinity.
#pragma omp parallel for collapse(2)  // spawn threads and distribute across them
  for(int which_gf=0;which_gf<NUM_EVOL_GFS;which_gf++) {
    for(int pt=0;pt<bc_info->num_inner_boundary_points;pt++) {
      const int dstpt = bcstruct->inner_bc_array[pt].dstpt;
      const int srcpt = bcstruct->inner_bc_array[pt].srcpt;
      gfs[IDX4pt(which_gf, dstpt)] = bcstruct->inner_bc_array[pt].parity[evol_gf_parity[which_gf]] * gfs[IDX4pt(which_gf, srcpt)];
    } // END for(int pt=0;pt<num_inner_pts;pt++)
  } // END for(int which_gf=0;which_gf<NUM_EVOL_GFS;which_gf++)
"""

        
        cfc.register_CFunction(
            includes=self.includes,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=self.params,
            include_CodeParameters_h=True,
            body=self.body,
        )


###############################
## apply_bcs_outerextrap_and_inner(): Apply extrapolation outer boundary conditions.
##  Function is documented below in desc= and body=.
class register_CFunction_apply_bcs_outerextrap_and_inner(
    base_cbc_classes.base_register_CFunction_apply_bcs_outerextrap_and_inner
):
    """Register C function for filling boundary points with extrapolation and prescribed bcstruct."""

    def __init__(self) -> None:
        super().__init__()
        self.prefunc=""
        self.body = r"""
  // Unpack bc_info from bcstruct
  const bc_info_struct *bc_info = &bcstruct->bc_info;

  ////////////////////////////////////////////////////////
  // STEP 1 of 2: Apply BCs to pure outer boundary points.
  //              By "pure" we mean that these points are
  //              on the outer boundary and not also on
  //              an inner boundary.
  //              Here we fill in the innermost ghost zone
  //              layer first and move outward. At each
  //              layer, we fill in +/- x0 faces first,
  //              then +/- x1 faces, finally +/- x2 faces,
  //              filling in the edges as we go.
  // Spawn N OpenMP threads, either across all cores, or according to e.g., taskset.
  apply_bcs_outerextrap_and_inner_only(params, bcstruct, gfs);

  ///////////////////////////////////////////////////////
  // STEP 2 of 2: Apply BCs to inner boundary points.
  //              These map to either the grid interior
  //              ("pure inner") or to pure outer boundary
  //              points ("inner maps to outer"). Those
  //              that map to outer require that outer be
  //              populated first; hence this being
  //              STEP 2 OF 2.
  apply_bcs_inner_only(commondata, params, bcstruct, gfs);
"""
        self.generate_prefunc__apply_bcs_outerextrap_and_inner_only()
        cfc.register_CFunction(
            prefunc=self.prefunc,
            includes=self.includes,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            name=self.name,
            params=self.params,
            include_CodeParameters_h=False,
            body=self.body,
        )
        
    def generate_prefunc__apply_bcs_outerextrap_and_inner_only(self) -> None:
        """
        Generate the prefunction string for apply_bcs_outerextrap_and_inner.
        
        This requires a function that will launch the device kernel as well
        as the device kernel itself.
        """
        
        # Header details for function that will launch the GPU kernel
        desc = "Apply BCs to pure boundary points"
        params = "const params_struct *restrict params, const bc_struct *restrict bcstruct, REAL *restrict gfs"
        name = "apply_bcs_outerextrap_and_inner_only"
        cfunc_type = "static void"
        
        # Start specifying the function body for launching the kernel
        kernel_launch_body = """
const bc_info_struct *bc_info = &bcstruct->bc_info;
  for (int which_gz = 0; which_gz < NGHOSTS; which_gz++) {
    for (int dirn = 0; dirn < 3; dirn++) {
      if (bc_info->num_pure_outer_boundary_points[which_gz][dirn] > 0) {
        size_t gz_idx = dirn + (3 * which_gz);
        const outerpt_bc_struct *restrict pure_outer_bc_array = bcstruct->pure_outer_bc_array[gz_idx];
        int num_pure_outer_boundary_points = bc_info->num_pure_outer_boundary_points[which_gz][dirn];
      """
        
        # Specify kernel launch body
        kernel_body = ""
        for i in range(3):
            kernel_body+=f"int const Nxx_plus_2NGHOSTS{i} = d_params.Nxx_plus_2NGHOSTS{i};\n"
        kernel_body += """  
// Thread indices
// Global data index - expecting a 1D dataset
const int tid0 = threadIdx.x + blockIdx.x*blockDim.x;

// Thread strides
const int stride0 = blockDim.x * gridDim.x;
    
for (int idx2d = tid0; idx2d < num_pure_outer_boundary_points; idx2d+=stride0) {
    const short i0 = pure_outer_bc_array[idx2d].i0;
    const short i1 = pure_outer_bc_array[idx2d].i1;
    const short i2 = pure_outer_bc_array[idx2d].i2;
    const short FACEX0 = pure_outer_bc_array[idx2d].FACEX0;
    const short FACEX1 = pure_outer_bc_array[idx2d].FACEX1;
    const short FACEX2 = pure_outer_bc_array[idx2d].FACEX2;
    const int idx_offset0 = IDX3(i0, i1, i2);
    const int idx_offset1 = IDX3(i0 + 1 * FACEX0, i1 + 1 * FACEX1, i2 + 1 * FACEX2);
    const int idx_offset2 = IDX3(i0 + 2 * FACEX0, i1 + 2 * FACEX1, i2 + 2 * FACEX2);
    const int idx_offset3 = IDX3(i0 + 3 * FACEX0, i1 + 3 * FACEX1, i2 + 3 * FACEX2);
    for (int which_gf = 0; which_gf < NUM_EVOL_GFS; which_gf++) {
      // *** Apply 2nd-order polynomial extrapolation BCs to all outer boundary points. ***
      gfs[IDX4pt(which_gf, idx_offset0)] =
          + 3.0 * gfs[IDX4pt(which_gf, idx_offset1)] 
          - 3.0 * gfs[IDX4pt(which_gf, idx_offset2)] 
          + 1.0 * gfs[IDX4pt(which_gf, idx_offset3)];
    }
  }
"""
        # Generate a GPU Kernel
        device_kernel = gputils.GPU_Kernel(
            kernel_body,
            {
                'num_pure_outer_boundary_points' : 'const int',
                'which_gz' : 'const int',
                'dirn' : 'const int',
                'pure_outer_bc_array' : 'const outerpt_bc_struct *restrict',
                'gfs' : 'REAL *restrict'
            },
            f"{name}_gpu",
            launch_dict= {
                'blocks_per_grid' : ["(num_pure_outer_boundary_points + threads_in_x_dir -1) / threads_in_x_dir"],
                'threads_per_block' : ["32"],
                'stream' : f"params->grid_idx % nstreams"
            },
            # fp_type=self.fp_type,
            comments=f"GPU Kernel to apply extrapolation BCs to pure points.",
        )
        # Add device Kernel to prefunc
        self.prefunc += device_kernel.CFunction.full_function
        # Add launch configuration to Launch kernel body
        kernel_launch_body+=device_kernel.launch_block
        kernel_launch_body+=device_kernel.c_function_call()
        # Close the launch kernel
        kernel_launch_body+="""
      }
    }
  }
"""
        # Generate the Launch kernel CFunction
        kernel_launch_CFunction = cfc.CFunction(
            includes=[],
            desc=desc,
            cfunc_type=cfunc_type,
            name=name,
            params=params,
            body=kernel_launch_body,
        )
        
        # Append Launch kernel to prefunc
        self.prefunc += kernel_launch_CFunction.full_function

# apply_bcs_outerradiation_and_inner():
#   Apply radiation BCs at outer boundary points, and
#   inner boundary conditions at inner boundary points.
class register_CFunction_apply_bcs_outerradiation_and_inner(
    base_cbc_classes.base_register_CFunction_apply_bcs_outerradiation_and_inner
):
    """
    Register a C function to apply boundary conditions to both pure outer and inner boundary points.

    :param CoordSystem: The coordinate system to use.
    :param radiation_BC_fd_order: Finite differencing order for the radiation boundary conditions. Default is 2.
    :param fp_type: Floating point type, e.g., "double".
    """

    def __init__(
        self,
        CoordSystem: str,
        radiation_BC_fd_order: int = 2,
        fp_type: str = "double",
    ) -> None:
        super().__init__(
            CoordSystem,
            radiation_BC_fd_order=radiation_BC_fd_order,
            fp_type=fp_type,
        )
        self.body = r"""
  // Unpack bc_info from bcstruct
  const bc_info_struct *bc_info = &bcstruct->bc_info;

  ////////////////////////////////////////////////////////
  // STEP 1 of 2: Apply BCs to pure outer boundary points.
  //              By "pure" we mean that these points are
  //              on the outer boundary and not also on
  //              an inner boundary.
  //              Here we fill in the innermost ghost zone
  //              layer first and move outward. At each
  //              layer, we fill in +/- x0 faces first,
  //              then +/- x1 faces, finally +/- x2 faces,
  //              filling in the edges as we go.
  // Spawn N OpenMP threads, either across all cores, or according to e.g., taskset.
#pragma omp parallel
  {
    for(int which_gz=0;which_gz<NGHOSTS;which_gz++) for(int dirn=0;dirn<3;dirn++) {
        // This option results in about 1.6% slower runtime for SW curvilinear at 64x24x24 on 8-core Ryzen 9 4900HS
        //#pragma omp for collapse(2)
        //for(int which_gf=0;which_gf<NUM_EVOL_GFS;which_gf++) for(int idx2d=0;idx2d<bc_info->num_pure_outer_boundary_points[which_gz][dirn];idx2d++) {
        //  {
        // Don't spawn a thread if there are no boundary points to fill; results in a nice little speedup.
        if(bc_info->num_pure_outer_boundary_points[which_gz][dirn] > 0) {
#pragma omp for  // threads have been spawned; here we distribute across them
          for(int idx2d=0;idx2d<bc_info->num_pure_outer_boundary_points[which_gz][dirn];idx2d++) {
            const short i0 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i0;
            const short i1 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i1;
            const short i2 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].i2;
            const short FACEX0 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX0;
            const short FACEX1 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX1;
            const short FACEX2 = bcstruct->pure_outer_bc_array[dirn + (3*which_gz)][idx2d].FACEX2;
            const int idx3 = IDX3(i0,i1,i2);
            for(int which_gf=0;which_gf<NUM_EVOL_GFS;which_gf++) {
              // *** Apply radiation BCs to all outer boundary points. ***
              rhs_gfs[IDX4pt(which_gf, idx3)] = radiation_bcs(commondata, params, bcstruct, xx, gfs, rhs_gfs, which_gf,
                                                               custom_wavespeed[which_gf], custom_f_infinity[which_gf],
                                                               i0,i1,i2, FACEX0,FACEX1,FACEX2);
            }
          }
        }
      }
  }

  ///////////////////////////////////////////////////////
  // STEP 2 of 2: Apply BCs to inner boundary points.
  //              These map to either the grid interior
  //              ("pure inner") or to pure outer boundary
  //              points ("inner maps to outer"). Those
  //              that map to outer require that outer be
  //              populated first; hence this being
  //              STEP 2 OF 2.
  apply_bcs_inner_only(commondata, params, bcstruct, rhs_gfs); // <- apply inner BCs to RHS gfs only
"""
        cfc.register_CFunction(
            includes=self.includes,
            prefunc=self.prefunc,
            desc=self.desc,
            cfunc_type=self.cfunc_type,
            CoordSystem_for_wrapper_func=self.CoordSystem,
            name=self.name,
            params=self.params,
            include_CodeParameters_h=True,
            body=self.body,
        )


class CurviBoundaryConditions_register_C_functions(
    base_cbc_classes.base_CurviBoundaryConditions_register_C_functions
):
    """
    Register various C functions responsible for handling boundary conditions.

    :param list_of_CoordSystems: List of coordinate systems to use.
    :param radiation_BC_fd_order: Finite differencing order for the radiation boundary conditions. Default is 2.
    :param set_parity_on_aux: If True, set parity on auxiliary grid functions.
    :param set_parity_on_auxevol: If True, set parity on auxiliary evolution grid functions.
    :param fp_type: Floating point type, e.g., "double".
    """

    def __init__(
        self,
        list_of_CoordSystems: List[str],
        radiation_BC_fd_order: int = 2,
        set_parity_on_aux: bool = False,
        set_parity_on_auxevol: bool = False,
        fp_type: str = "double",
    ) -> None:
        super().__init__(
            list_of_CoordSystems,
            radiation_BC_fd_order=radiation_BC_fd_order,
            set_parity_on_aux=set_parity_on_aux,
            set_parity_on_auxevol=set_parity_on_auxevol,
            fp_type=fp_type,
        )
        self.CBC_BHd_str = self.CBC_BHd_str.replace("outerpt_bc_struct *restrict", "outerpt_bc_struct *")
        self.CBC_BHd_str = self.CBC_BHd_str.replace("innerpt_bc_struct *restrict", "innerpt_bc_struct *")
        # self.post_register_BHAH_header(self)
        for CoordSystem in self.list_of_CoordSystems:
            # Register C function to set up the boundary condition struct.
            register_CFunction_bcstruct_set_up(
                CoordSystem=CoordSystem, fp_type=self.fp_type
            )

            # Register C function to apply boundary conditions to both pure outer and inner boundary points.
            register_CFunction_apply_bcs_outerradiation_and_inner(
                CoordSystem=CoordSystem,
                radiation_BC_fd_order=self.radiation_BC_fd_order,
                fp_type=self.fp_type,
            )

        # Register C function to apply boundary conditions to inner-only boundary points.
        register_CFunction_apply_bcs_inner_only()

        # Register C function to apply boundary conditions to outer-extrapolated and inner boundary points.
        register_CFunction_apply_bcs_outerextrap_and_inner()

        griddata_commondata.register_griddata_commondata(
            __name__,
            "bc_struct bcstruct",
            "all data needed to perform boundary conditions in curvilinear coordinates",
        )

        BHaH_defines_h.register_BHaH_defines(__name__, self.CBC_BHd_str)


if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
