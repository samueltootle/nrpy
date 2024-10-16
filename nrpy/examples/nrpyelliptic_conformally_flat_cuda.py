"""
Sets up a complete C code project for solving the hyperbolic relaxation equation in curvilinear coordinates on a cell-centered grid, using a reference metric.

Authors: Thiago Assumpção; assumpcaothiago **at** gmail **dot** com
         Zachariah B. Etienne; zachetie **at** gmail **dot* com
"""

#########################################################
# STEP 1: Import needed Python modules, then set codegen
#         and compile-time parameters.
import shutil
import os
from math import sqrt
from typing import Any, Dict
import nrpypn.eval_p_t_and_p_r as bbhp

import nrpy.params as par
from nrpy.helpers import simd
import nrpy.helpers.parallel_codegen as pcg
import nrpy.helpers.gpu_kernels.cuda_utilities as gputils

import nrpy.infrastructures.BHaH.cmdline_input_and_parfiles as cmdpar
import nrpy.infrastructures.BHaH.CodeParameters as CPs
import nrpy.infrastructures.BHaH.diagnostics.progress_indicator as progress
import nrpy.infrastructures.BHaH.Makefile_helpers as Makefile
from nrpy.infrastructures.BHaH import rfm_wrapper_functions
import nrpy.infrastructures.gpu.header_definitions.cuda.output_BHaH_defines_h as Bdefines_h
import nrpy.infrastructures.gpu.checkpoints.cuda.checkpointing as chkpt
import nrpy.infrastructures.gpu.grid_management.cuda.griddata_free as griddata_commondata
import nrpy.infrastructures.gpu.main_driver.cuda.main_c as main
from nrpy.infrastructures.gpu.MoLtimestepping.cuda import MoL
import nrpy.infrastructures.gpu.CurviBoundaryConditions.cuda.CurviBoundaryConditions as cbc
import nrpy.infrastructures.gpu.nrpyelliptic.cuda.conformally_flat_C_codegen_library as nrpyellClib
import nrpy.infrastructures.gpu.grid_management.cuda.numerical_grids_and_timestep as numericalgrids
import nrpy.infrastructures.gpu.grid_management.cuda.register_rfm_precompute as rfm_precompute
from nrpy.infrastructures.gpu.grid_management.cuda import xx_tofrom_Cart

par.set_parval_from_str("Infrastructure", "BHaH")

# Code-generation-time parameters:
project_name = "nrpyelliptic_conformally_flat_cuda"
fp_type = "double"
grid_physical_size = 1.0e6
t_final = grid_physical_size  # This parameter is effectively not used in NRPyElliptic
nn_max = 10000  # Sets the maximum number of relaxation steps
Q = 5
R = 128

def get_log10_residual_tolerance(fp_type_str: str = "double") -> float:
    """
    Determine the residual tolerance based on the fp_precision.

    :param fp_type_str: string representing the floating point type.
    :return: float of the residual tolerance based on fp_type.
    :raises ValueError: If the input fp_type_str branch is not defined.
    """
    res: float = -1.0
    if fp_type_str == "double":
        res = -15.8
    elif fp_type_str == "float":
        res = -10
    else:
        raise ValueError(f"residual tolerence not defined for {fp_type_str} precision")
    return res


# Set tolerance for log10(residual) to stop relaxation
log10_residual_tolerance = get_log10_residual_tolerance(fp_type_str=fp_type)
default_diagnostics_output_every = 100
default_checkpoint_every = 50.0
eta_damping = 11.0
MINIMUM_GLOBAL_WAVESPEED = 0.7
CFL_FACTOR = 1.0  # NRPyElliptic wave speed prescription assumes this parameter is ALWAYS set to 1
CoordSystem = "SinhSymTP"
Nxx_dict = {
    "SinhSymTP": [R, R, 16],
    "SinhCylindricalv2": [128, 16, 256],
    "SinhSpherical": [128, 128, 16],
}
# Set parameters specific to SinhSymTP coordinates
AMAX = grid_physical_size
bScale = 5.0
SINHWAA = 0.07
# Set parameters specific to SinhCylindricalv2 coordinates
AMPLRHO = grid_physical_size
AMPLZ = grid_physical_size
SINHWRHO = 0.04
SINHWZ = 0.04
const_drho = 2.0e-5
const_dz = 2.0e-5
# Set parameters specific to SinhSpherical coordinates
AMPL = grid_physical_size
SINHW = 0.06

OMP_collapse = 1
enable_checkpointing = True
enable_rfm_precompute = True
MoL_method = "RK4"
fd_order = 10
radiation_BC_fd_order = 6
enable_simd = False
parallel_codegen_enable = True
boundary_conditions_desc = "outgoing radiation"
# fmt: off
initial_data_type = "gw150914"  # choices are: "gw150914", "axisymmetric", and "single_puncture"

def set_gw150914_params() -> Dict[str, Any]:
    """
    Set parameters consistent with GW150914.
    Parameters are taken from http://einsteintoolkit.org/gallery/bbh/index.html

    :return: Dictionary of parameters
    """
    q = 36.0 / 29.0
    S0_y_dimless = 0.31
    S1_y_dimless = -0.46
    m0_adm = q / (1.0 + q)
    m1_adm = 1.0 / (1.0 + q)

    # Note, these momenta are the eccentricity reduced values
    # as obtained from the ETK gallery example, see:
    # http://einsteintoolkit.org/gallery/bbh/index.html
    Pr = -0.00084541526517121  # Radial linear momentum
    Pphi = 0.09530152296974252  # Azimuthal linear momentum


    gw150914_params = {
        "zPunc": 5.0,
        "q": q,
        "bare_mass_0": 0.51841993533587039,
        "bare_mass_1": 0.39193567996522616,
        "Pr": Pr,
        "Pphi": Pphi,
        "S0_y_dimless": S0_y_dimless,
        "S1_y_dimless": S1_y_dimless,
        "m0_adm": m0_adm,
        "m1_adm": m1_adm,
        "S0_y": S0_y_dimless * (m0_adm ** 2),
        "S1_y": S1_y_dimless * (m1_adm ** 2),
        "P0_x": Pphi,
        "P0_z": Pr,
        "P1_x": -Pphi,
        "P1_z": -Pr,
    }
    return gw150914_params

def compute_bare_mass(M: float, CHI: float) -> float:
    """
    Set analytical Kerr estimate of the bare puncture mass.

    :param M: puncture ADM mass
    :param CHI: dimensionless spin magnitude of the puncture
    :return: bare mass
    """
    res: float = M * sqrt(0.5 * (1.0 + sqrt(1.0 - CHI**2.0)))
    return res


def set_axisymmetric_params() -> Dict[str, Any]:
    """
    Set parameters for an axisymmetric BBH setup.

    This setup by default uses the analytical estimate from
    Kerr to compute bare masses unless the user specifies them
    manually.  Thus, the solution will not produce a binary with
    accurate puncture masses as measured by an apparent horizon
    finder that match the expect puncture ADM masses.
    A future extension could include an iterative root finder
    to make this more robust.

    :return: Dictionary of parameters
    """
    global Q
    q = Q
    M_total = 1.0
    distance = M_total * 5.0

    S0_y_dimless = 0.0
    S1_y_dimless = 0.0
    m0_adm = M_total * q   / (1.0 + q)
    m1_adm = M_total * 1.0 / (1.0 + q)

    bare_m0 = compute_bare_mass(m0_adm, S0_y_dimless)
    bare_m1 = compute_bare_mass(m1_adm, S1_y_dimless)

    # Compute 3.5PN estimates
    Pphi, Pr = bbhp.eval__P_t__and__P_r(
        q,
        distance,
        0.0,
        S0_y_dimless,
        0.0,
        0.0,
        S1_y_dimless,
        0.0,
    )

    axisymmetric_params = {
        "zPunc": distance,
        "q": q,
        "bare_mass_0": bare_m0,
        "bare_mass_1": bare_m1,
        "Pr": Pr,
        "Pphi": Pphi,
        "S0_y_dimless": S0_y_dimless,
        "S1_y_dimless": S1_y_dimless,
        "m0_adm": m0_adm,
        "m1_adm": m1_adm,
        "S0_y": S0_y_dimless * (m0_adm ** 2),
        "S1_y": S1_y_dimless * (m1_adm ** 2),
        "P0_x": Pphi,
        "P0_z": Pr,
        "P1_x": -Pphi,
        "P1_z": -Pr,
    }
    return axisymmetric_params

def set_single_puncture_params() -> Dict[str, Any]:
    """
    Set parameters for an axisymmetric BH setup.

    This setup by default uses the analytical estimate from
    Kerr to compute the bare mass unless the user specifies them
    manually.  Thus, the solution will not produce a binary with
    accurate puncture masses as measured by an apparent horizon
    finder that match the expect puncture ADM mass.
    A future extension could include an iterative root finder
    to make this more robust.

    :return: Dictionary of parameters
    """
    coordinate_location = 0.0
    m_adm = 0.5
    S_z_dimless = 0.2
    single_puncture_params = {
        "zPunc": coordinate_location,
        "bare_mass_0": compute_bare_mass(m_adm, S_z_dimless),
        "m0_adm": m_adm,
        "S0_z": S_z_dimless * m_adm**2.0,
    }
    return single_puncture_params
# fmt: on
# project_name += f"-q{Q}-R{R}"
project_dir = os.path.join("project", project_name)

# First clean the project directory, if it exists.
shutil.rmtree(project_dir, ignore_errors=True)

par.set_parval_from_str("parallel_codegen_enable", parallel_codegen_enable)
par.set_parval_from_str("fd_order", fd_order)
par.set_parval_from_str("CoordSystem_to_register_CodeParameters", CoordSystem)
par.adjust_CodeParam_default("t_final", t_final)


#########################################################
# STEP 2: Declare core C functions & register each to
#         cfc.CFunction_dict["function_name"]
gputils.register_CFunctions_HostDevice__operations()
gputils.register_CFunction_find_global_minimum(fp_type=fp_type)
gputils.register_CFunction_find_global_sum(fp_type=fp_type)

# Generate functions to set initial guess
nrpyellClib.register_CFunction_initial_guess_single_point(fp_type=fp_type)
nrpyellClib.register_CFunction_initial_guess_all_points(
    OMP_collapse=OMP_collapse,
    enable_checkpointing=enable_checkpointing,
    fp_type=fp_type,
)

# Generate function to set variable wavespeed
nrpyellClib.register_CFunction_variable_wavespeed_gfs_all_points(
    CoordSystem=CoordSystem, fp_type=fp_type
)

# Generate functions to set AUXEVOL gridfunctions
nrpyellClib.register_CFunction_auxevol_gfs_single_point(
    CoordSystem=CoordSystem, fp_type=fp_type
)
nrpyellClib.register_CFunction_auxevol_gfs_all_points(
    OMP_collapse=OMP_collapse, fp_type=fp_type
)

# Generate function that calls functions to set variable wavespeed and all other AUXEVOL gridfunctions
nrpyellClib.register_CFunction_initialize_constant_auxevol()

numericalgrids.register_CFunctions(
    list_of_CoordSystems=[CoordSystem],
    list_of_grid_physical_sizes=[grid_physical_size],
    Nxx_dict=Nxx_dict,
    enable_rfm_precompute=enable_rfm_precompute,
    enable_CurviBCs=True,
    fp_type=fp_type,
)
xx_tofrom_Cart.register_CFunction_xx_to_Cart(CoordSystem=CoordSystem, fp_type=fp_type)

nrpyellClib.register_CFunction_diagnostics(
    CoordSystem=CoordSystem,
    default_diagnostics_out_every=default_diagnostics_output_every,
)

if enable_rfm_precompute:
    rfm_precompute.register_CFunctions_rfm_precompute(
        list_of_CoordSystems=[CoordSystem], fp_type=fp_type
    )

# Generate function to compute RHSs
nrpyellClib.register_CFunction_rhs_eval(
    CoordSystem=CoordSystem,
    enable_rfm_precompute=enable_rfm_precompute,
    enable_simd=enable_simd,
    OMP_collapse=OMP_collapse,
    fp_type=fp_type,
)

# Generate function to compute residuals
nrpyellClib.register_CFunction_compute_residual_all_points(
    CoordSystem=CoordSystem,
    enable_rfm_precompute=enable_rfm_precompute,
    enable_simd=enable_simd,
    OMP_collapse=OMP_collapse,
    fp_type=fp_type,
)

# Generate diagnostics functions
nrpyellClib.register_CFunction_compute_L2_norm_of_gridfunction(
    CoordSystem=CoordSystem, fp_type=fp_type
)

# Register function to check for stop conditions
nrpyellClib.register_CFunction_check_stop_conditions()

if __name__ == "__main__" and parallel_codegen_enable:
    pcg.do_parallel_codegen()

cbc.CurviBoundaryConditions_register_C_functions(
    list_of_CoordSystems=[CoordSystem],
    radiation_BC_fd_order=radiation_BC_fd_order,
    fp_type=fp_type,
)
rhs_string = """rhs_eval(commondata, params, rfmstruct,  auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);
if (strncmp(commondata->outer_bc_type, "radiation", 50) == 0){
  REAL wavespeed_at_outer_boundary;
  cudaMemcpy(&wavespeed_at_outer_boundary, &auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, Nxx_plus_2NGHOSTS0-NGHOSTS-1, NGHOSTS, Nxx_plus_2NGHOSTS2/2)], sizeof(REAL), cudaMemcpyDeviceToHost);
  const REAL custom_gridfunctions_wavespeed[2] = {wavespeed_at_outer_boundary, wavespeed_at_outer_boundary};
  apply_bcs_outerradiation_and_inner(commondata, params, bcstruct, griddata->xx,
                                     custom_gridfunctions_wavespeed, gridfunctions_f_infinity,
                                     RK_INPUT_GFS, RK_OUTPUT_GFS);
}"""
if not enable_rfm_precompute:
    rhs_string = rhs_string.replace("rfmstruct", "xx")
MoL.register_CFunctions(
    MoL_method=MoL_method,
    rhs_string=rhs_string,
    post_rhs_string="""if (strncmp(commondata->outer_bc_type, "extrapolation", 50) == 0)
  apply_bcs_outerextrap_and_inner(commondata, params, bcstruct, RK_OUTPUT_GFS);""",
    enable_rfm_precompute=enable_rfm_precompute,
    enable_curviBCs=True,
    fp_type=fp_type,
)
chkpt.register_CFunctions(default_checkpoint_every=default_checkpoint_every)

# Define string with print statement for progress indicator
progress_str = r"""
  fprintf(stderr, "nn / nn_max = %d / %d ; log10(residual) / log10(residual_target) =  %.4f / %.4f \r",
    commondata->nn,
    commondata->nn_max,
    commondata->log10_current_residual,
    commondata->log10_residual_tolerance);
  fflush(stderr); // Flush the stderr buffer
"""
progress.register_CFunction_progress_indicator(
    progress_str=progress_str, compute_ETA=False
)
rfm_wrapper_functions.register_CFunctions_CoordSystem_wrapper_funcs()

# Update parameters needed for hyperbolic relaxation method
par.adjust_CodeParam_default("eta_damping", eta_damping)
par.adjust_CodeParam_default("MINIMUM_GLOBAL_WAVESPEED", MINIMUM_GLOBAL_WAVESPEED)
par.adjust_CodeParam_default("CFL_FACTOR", CFL_FACTOR)
par.adjust_CodeParam_default("nn_max", nn_max)
par.adjust_CodeParam_default("log10_residual_tolerance", log10_residual_tolerance)

# Update parameters specific to the coordinate system
if CoordSystem == "SinhSymTP":
    par.adjust_CodeParam_default("AMAX", AMAX)
    par.adjust_CodeParam_default("bScale", bScale)
    par.adjust_CodeParam_default("SINHWAA", SINHWAA)

if CoordSystem == "SinhCylindricalv2":
    par.adjust_CodeParam_default("AMPLRHO", AMPLRHO)
    par.adjust_CodeParam_default("AMPLZ", AMPLZ)
    par.adjust_CodeParam_default("SINHWRHO", SINHWRHO)
    par.adjust_CodeParam_default("SINHWZ", SINHWZ)
    par.adjust_CodeParam_default("const_drho", const_drho)
    par.adjust_CodeParam_default("const_dz", const_dz)

if CoordSystem == "SinhSpherical":
    par.adjust_CodeParam_default("AMPL", AMPL)
    par.adjust_CodeParam_default("SINHW", SINHW)

# Update parameters specific to initial data type
if initial_data_type == "single_puncture":
    puncture_params = set_single_puncture_params()
    for param, value in puncture_params.items():
        if param in [
            "zPunc",
            "bare_mass_0",
            "S0_z",
        ]:
            par.adjust_CodeParam_default(param, value)
else:
    if initial_data_type == "gw150914":
        puncture_params = set_gw150914_params()
    elif initial_data_type == "axisymmetric":
        puncture_params = set_axisymmetric_params()

    for param, value in puncture_params.items():
        if param in [
            "zPunc",
            "bare_mass_0",
            "bare_mass_1",
            "S0_y",
            "S1_y",
            "P0_x",
            "P0_z",
            "P1_x",
            "P1_z",
        ]:
            par.adjust_CodeParam_default(param, value)

#########################################################
# STEP 3: Generate header files, register C functions and
#         command line parameters, set up boundary conditions,
#         and create a Makefile for this project.
#         Project is output to project/[project_name]/
CPs.write_CodeParameters_h_files(project_dir=project_dir, decorator="[[maybe_unused]]")
CPs.register_CFunctions_params_commondata_struct_set_to_default()
cmdpar.generate_default_parfile(project_dir=project_dir, project_name=project_name)
cmdpar.register_CFunction_cmdline_input_and_parfile_parser(
    project_name=project_name, cmdline_inputs=["convergence_factor"]
)
Bdefines_h.output_BHaH_defines_h(
    project_dir=project_dir,
    enable_simd=enable_simd,
    REAL_means=fp_type,
    supplemental_defines_dict={
        "ADDITIONAL GPU DIAGNOSTICS": "#define L2_DVGF 0\n"
        "#define L2_SQUARED_DVGF 1\n",
        "ADDITIONAL HOST DIAGNOSTICS": "#define HOST_RESIDUAL_HGF 0\n"
        "#define HOST_UUGF 1\n"
        "#define NUM_HOST_DIAG 2\n",
    },
)
# Define post_MoL_step_forward_in_time string for main function
post_MoL_step_forward_in_time = r"""    check_stop_conditions(&commondata, griddata);
    if (commondata.stop_relaxation) {
      // Force a checkpoint when stop condition is reached.
      commondata.checkpoint_every = 1e-4*commondata.dt;
      write_checkpoint(&commondata, griddata_host, griddata);
      break;
    }
"""
main.register_CFunction_main_c(
    initial_data_desc="",
    post_non_y_n_auxevol_mallocs="initialize_constant_auxevol(&commondata, griddata);\n",
    pre_MoL_step_forward_in_time="write_checkpoint(&commondata, griddata_host, griddata);\n",
    post_MoL_step_forward_in_time=post_MoL_step_forward_in_time,
    MoL_method=MoL_method,
    boundary_conditions_desc=boundary_conditions_desc,
)
griddata_commondata.register_CFunction_griddata_free(
    enable_rfm_precompute=enable_rfm_precompute, enable_CurviBCs=True
)

if enable_simd:
    simd.copy_simd_intrinsics_h(project_dir=project_dir)

Makefile.output_CFunctions_function_prototypes_and_construct_Makefile(
    project_dir=project_dir,
    project_name=project_name,
    exec_or_library_name=project_name,
    CC="nvcc",
    code_ext="cu",
    compiler_opt_option="nvcc",
)
print(
    f"Finished! Now go into project/{project_name} and type `make` to build, then ./{project_name} to run."
)
print(f"    Parameter file can be found in {project_name}.par")