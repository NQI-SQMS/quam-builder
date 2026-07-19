from typing import Dict

from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ, XYDriveMW
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import (
    CavityTransmonPair,
)
from quam_builder.builder.qop_connectivity.channel_ports import (
    iq_out_channel_ports,
    mw_out_channel_ports,
)
from quam_builder.builder.qop_connectivity.get_digital_outputs import get_digital_outputs


def add_cavity_sideband_drive_component(
    pair: CavityTransmonPair,
    wiring_path: str,
    ports: Dict[str, str],
):
    """Adds a sideband drive component to a CavityTransmonPair.

    Mirrors add_transmon_drive_component but targets pair.sideband_drive for the
    |f,0⟩↔|g,1⟩ sideband transition channel.

    Args:
        pair: The CavityTransmonPair to which the sideband drive will be added.
        wiring_path: The wiring JSON reference path
            (e.g. "#/wiring/cavity_transmon_pairs/q1_alice/sideband").
        ports: Dictionary mapping port names to their wiring references.

    Raises:
        ValueError: If the port keys do not match any implemented mapping.
    """
    digital_outputs = get_digital_outputs(wiring_path, ports)

    if all(key in ports for key in iq_out_channel_ports):
        # OPX+ + Octave or LF-FEM + MW-FEM
        pair.sideband_drive = XYDriveIQ(
            opx_output_I=f"{wiring_path}/opx_output_I",
            opx_output_Q=f"{wiring_path}/opx_output_Q",
            frequency_converter_up=f"{wiring_path}/frequency_converter_up",
            RF_frequency=None,
            digital_outputs=digital_outputs,
        )
        RF_out = pair.sideband_drive.frequency_converter_up
        RF_out.channel = pair.sideband_drive.get_reference()
        RF_out.output_mode = "always_on"

    elif all(key in ports for key in mw_out_channel_ports):
        # OPX1000 MW-FEM single channel
        pair.sideband_drive = XYDriveMW(
            opx_output=f"{wiring_path}/opx_output",
            digital_outputs=digital_outputs,
            RF_frequency=None,
        )

    else:
        raise ValueError(
            f"Unimplemented mapping of port keys to channel for cavity sideband drive ports: {ports}"
        )
