from typing import Dict

from quam_builder.architecture.superconducting.cavity.cavity_mode import CavityMode
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ, XYDriveMW
from quam_builder.builder.qop_connectivity.channel_ports import (
    iq_out_channel_ports,
    mw_out_channel_ports,
)
from quam_builder.builder.qop_connectivity.get_digital_outputs import get_digital_outputs


def add_cavity_mode_drive_component(
    mode: CavityMode,
    wiring_path: str,
    ports: Dict[str, str],
):
    """Adds a drive component to a cavity mode based on the provided wiring path and ports.

    Mirrors add_transmon_drive_component but targets mode.cavity_mode_drive instead of
    transmon.xy.

    Args:
        mode: The CavityMode to which the drive component will be added.
        wiring_path: The wiring JSON reference path (e.g. "#/wiring/cavities/c1/alice/cavity").
        ports: Dictionary mapping port names to their wiring references.

    Raises:
        ValueError: If the port keys do not match any implemented mapping.
    """
    digital_outputs = get_digital_outputs(wiring_path, ports)

    if all(key in ports for key in iq_out_channel_ports):
        # OPX+ + Octave or LF-FEM + MW-FEM
        mode.cavity_mode_drive = XYDriveIQ(
            opx_output_I=f"{wiring_path}/opx_output_I",
            opx_output_Q=f"{wiring_path}/opx_output_Q",
            frequency_converter_up=f"{wiring_path}/frequency_converter_up",
            RF_frequency=None,
            digital_outputs=digital_outputs,
        )
        RF_out = mode.cavity_mode_drive.frequency_converter_up
        RF_out.channel = mode.cavity_mode_drive.get_reference()
        RF_out.output_mode = "always_on"

    elif all(key in ports for key in mw_out_channel_ports):
        # OPX1000 MW-FEM single channel
        mode.cavity_mode_drive = XYDriveMW(
            opx_output=f"{wiring_path}/opx_output",
            digital_outputs=digital_outputs,
            RF_frequency=None,
        )

    else:
        raise ValueError(
            f"Unimplemented mapping of port keys to channel for cavity mode drive ports: {ports}"
        )
