from typing import Dict, List
from pathlib import Path

import nrpy.grid as gri


class ScheduleCCL:
    """
    Class representing a ScheduleCCL object.
    """

    def __init__(
        self,
        function_name: str,
        bin: str,
        entry: str,
    ) -> None:
        """
        Initialize a ScheduleCCL object.

        :param entry: The scheduling entry.
        """
        self.function_name = function_name
        self.bin = bin
        self.entry = entry
        self.has_been_output = False


def construct_schedule_ccl(project_dir: str, thorn_name: str, STORAGE: str) -> None:
    """
    Construct the ScheduleCCL string based on its properties.

    :return: The constructed ScheduleCCL string.
    """
    outstr = """# This schedule.ccl file was automatically generated by NRPy+.
#   You are advised against modifying it directly; instead
#   modify the Python code that generates it.
"""
    outstr += f"""\n##################################################
# Step 0: Allocate memory for gridfunctions, using the STORAGE: keyword.
{STORAGE}
"""

    step = 1
    for bin in [
        "STARTUP",
        "Driver_BoundarySelect",
        "BASEGRID",
        "CCTK_INITIAL",
        "MoL_Register",
        "MoL_CalcRHS",
        "MoL_PostStep",
        "MoL_PseudoEvolution",
    ]:
        already_output_header = False
        for item in schedule_ccl_dict[thorn_name]:
            if item.bin.upper() == bin.upper() and not item.has_been_output:
                if not already_output_header:
                    outstr += f"""\n##################################################
# Step {step}: Schedule functions in the {bin} scheduling bin.
"""
                    already_output_header = True
                    step += 1
                outstr += item.entry.replace("FUNC_NAME", item.function_name)
                item.has_been_output = True

    for item in schedule_ccl_dict[thorn_name]:
        if not item.has_been_output:
            outstr += f"""\n##################################################
# Step {step}: Schedule functions in the remaining scheduling bins.
"""
            outstr += item.entry.replace("FUNC_NAME", item.function_name)

    with open(Path(project_dir) / thorn_name / "schedule.ccl", "w") as file:
        file.write(outstr)


schedule_ccl_dict: Dict[str, List[ScheduleCCL]] = {}


def register_ScheduleCCL(
    thorn_name: str, function_name: str, bin: str, entry: str
) -> None:
    """
    Registers a ScheduleCCL object to the schedule_ccl_dict.

    :param thorn_name: The name of the thorn.
    :param function_name: The name of the function.
    :param bin: The bin specification.
    :param entry: The entry description.
    :raises KeyError: Raised if thorn_name does not exist in schedule_ccl_dict.
    """
    schedule_ccl_dict.setdefault(thorn_name, []).append(
        ScheduleCCL(function_name=function_name, bin=bin, entry=entry)
    )


def auto_EVOL_AUXEVOL_AUX_STORAGE() -> str:
    outstr = ""
    for gfname, gf in gri.glb_gridfcs_dict.items():
        if gf.group == "EVOL":
            outstr += """
STORAGE: evol_variables[3]     # Evolution variables
STORAGE: evol_variables_rhs[1] # Variables storing right-hand-sides
"""
            break
    for gfname, gf in gri.glb_gridfcs_dict.items():
        if gf.group == "AUXEVOL":
            outstr += """
STORAGE: auxevol_variables[1]  # Single-timelevel storage of variables needed for evolutions.
"""
            break

    for gfname, gf in gri.glb_gridfcs_dict.items():
        if gf.group == "AUX":
            outstr += """
STORAGE: aux_variables[3]      # Diagnostics variables
"""
            break
    return outstr