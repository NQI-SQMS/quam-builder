from pathlib import Path
from typing import Union, Optional
from numpy import sqrt, ceil
from quam.components import Octave, LocalOscillator, FrequencyConverter
from quam_builder.architecture.superconducting.components.mixer import StandaloneMixer
from quam_builder.builder.superconducting.pulses import (
    add_default_transmon_pulses,
    add_default_transmon_pair_pulses,
    add_default_twpa_pulses,
)
from quam_builder.builder.superconducting.add_transmon_drive_component import (
    add_transmon_drive_component,
)
from quam_builder.builder.superconducting.add_twpa_pump_component import (
    add_twpa_pump_component,
)
from quam_builder.builder.superconducting.add_transmon_flux_component import (
    add_transmon_flux_component,
)
from quam_builder.builder.superconducting.add_transmon_pair_component import (
    add_transmon_pair_tunable_coupler_component,
    add_transmon_pair_cross_resonance_component,
    add_transmon_pair_zz_drive_component,
)
from quam_builder.builder.superconducting.add_transmon_resonator_component import (
    add_transmon_resonator_component,
)
from quam_builder.builder.superconducting.add_cavity_mode_drive_component import (
    add_cavity_mode_drive_component,
)
from quam_builder.builder.superconducting.add_cavity_sideband_drive_component import (
    add_cavity_sideband_drive_component,
)
from quam_builder.architecture.superconducting.cavity.cavity import Cavity
from quam_builder.architecture.superconducting.cavity.cavity_mode import CavityMode
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import (
    CavityTransmonPair,
)
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from quam_builder.architecture.superconducting.qpu import AnyQuam


def build_quam(
    machine: AnyQuam, calibration_db_path: Optional[Union[Path, str]] = None
) -> AnyQuam:
    """Builds the QuAM by adding various components and saving the machine configuration.

    Args:
        machine (AnyQuam): The QuAM to be built.
        calibration_db_path (Optional[Union[Path, str]]): The path to the Octave calibration database.

    Returns:
        AnyQuam: The built QuAM.
    """
    add_octaves(machine, calibration_db_path=calibration_db_path)
    add_external_mixers(machine)
    add_ports(machine)
    add_transmons(machine)
    add_twpas(machine)
    add_cavities(machine)
    add_pulses(machine)

    machine.save()

    return machine


def _create_ports_from_wiring(machine: AnyQuam, wiring_by_line_type):
    for ports in wiring_by_line_type.values():
        for port in ports:
            if "ports" in ports.get_unreferenced_value(port):
                machine.ports.reference_to_port(
                    ports.get_unreferenced_value(port), create=True
                )


def add_ports(machine: AnyQuam):
    """Creates and stores all input/output ports according to what has been allocated to each element in the machine's wiring.

    Args:
        machine (AnyQuam): The QuAM to which the ports will be added.
    """
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == "cavities":
            # Cavity wiring has an extra nesting level: cavity_id → mode_name → line_type → port_refs
            for wiring_by_mode in wiring_by_element.values():
                for wiring_by_line_type in wiring_by_mode.values():
                    _create_ports_from_wiring(machine, wiring_by_line_type)
        else:
            for wiring_by_line_type in wiring_by_element.values():
                _create_ports_from_wiring(machine, wiring_by_line_type)


def _set_default_grid_location(qubit_number: int, total_number_of_qubits: int) -> str:
    """Sets the default grid location for a qubit based on its number and the total number of qubits.

    Args:
        qubit_number (int): The number of the qubit.
        total_number_of_qubits (int): The total number of qubits.

    Returns:
        str: The grid location in the format "x,y".
    """
    number_of_rows = int(ceil(sqrt(total_number_of_qubits)))
    y = qubit_number % number_of_rows
    x = qubit_number // number_of_rows
    return f"{x},{y}"


def add_transmons(machine: AnyQuam):
    """Adds transmon qubits and qubit pairs to the machine based on the wiring configuration.

    Args:
        machine (AnyQuam): The QuAM to which the transmons will be added.
    """
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == "qubits":
            machine.active_qubit_names = []
            number_of_qubits = len(wiring_by_element.items())
            qubit_number = 0
            for qubit_id, wiring_by_line_type in wiring_by_element.items():
                qubit_class = machine.qubit_type
                transmon = qubit_class(id=qubit_id)
                machine.qubits[qubit_id] = transmon
                machine.qubits[qubit_id].grid_location = _set_default_grid_location(
                    qubit_number, number_of_qubits
                )
                qubit_number += 1
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_id}/{line_type}"
                    if line_type == WiringLineType.RESONATOR.value:
                        add_transmon_resonator_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.DRIVE.value:
                        add_transmon_drive_component(transmon, wiring_path, ports)
                    elif line_type == WiringLineType.FLUX.value:
                        add_transmon_flux_component(transmon, wiring_path, ports)
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                machine.active_qubit_names.append(transmon.name)

        elif element_type == "qubit_pairs":
            machine.active_qubit_pair_names = []
            for qubit_pair_id, wiring_by_line_type in wiring_by_element.items():
                qc, qt = qubit_pair_id.split("-")
                qt = f"q{qt}"
                transmon_pair = machine.qubit_pair_type(
                    id=qubit_pair_id,
                    qubit_control=f"#/qubits/{qc}",
                    qubit_target=f"#/qubits/{qt}",
                )
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = f"#/wiring/{element_type}/{qubit_pair_id}/{line_type}"
                    if line_type == WiringLineType.COUPLER.value:
                        add_transmon_pair_tunable_coupler_component(
                            transmon_pair, wiring_path, ports
                        )
                    elif line_type == WiringLineType.CROSS_RESONANCE.value:
                        add_transmon_pair_cross_resonance_component(
                            transmon_pair, wiring_path, ports
                        )
                    elif line_type == WiringLineType.ZZ_DRIVE.value:
                        add_transmon_pair_zz_drive_component(
                            transmon_pair, wiring_path, ports
                        )
                    else:
                        raise ValueError(f"Unknown line type: {line_type}")
                    machine.qubit_pairs[transmon_pair.name] = transmon_pair
                    machine.active_qubit_pair_names.append(transmon_pair.name)

        elif element_type in ("cavities", "cavity_transmon_pairs"):
            pass  # handled by add_cavities()


def add_cavities(machine: AnyQuam):
    """Builds Cavity, CavityMode, and CavityTransmonPair objects from the wiring dict.

    Only runs when the machine has both 'cavities' and 'cavity_transmon_pairs' attributes
    (i.e. the Quam class mixes in CavityQuam).

    Args:
        machine (AnyQuam): The QuAM to which the cavity components will be added.
    """
    if not hasattr(machine, "cavities") or not hasattr(machine, "cavity_transmon_pairs"):
        return

    for element_type, wiring_by_element in machine.wiring.items():
        if element_type == "cavities":
            for cavity_id, wiring_by_mode in wiring_by_element.items():
                cavity = Cavity(id=cavity_id)
                machine.cavities[cavity_id] = cavity
                for mode_name, wiring_by_line_type in wiring_by_mode.items():
                    mode = CavityMode(id=mode_name)
                    setattr(cavity, mode_name, mode)  # sets cavity.alice / cavity.bob
                    for line_type, ports in wiring_by_line_type.items():
                        wiring_path = (
                            f"#/wiring/cavities/{cavity_id}/{mode_name}/{line_type}"
                        )
                        if line_type == "cavity":
                            add_cavity_mode_drive_component(mode, wiring_path, ports)

        elif element_type == "cavity_transmon_pairs":
            for pair_id, wiring_by_line_type in wiring_by_element.items():
                # pair_id format: "{qubit_name}_{mode_name}", e.g. "q1_alice"
                qubit_name, mode_name = pair_id.split("_", 1)
                pair = CavityTransmonPair(
                    qubit_name=qubit_name, cavity_mode_name=mode_name
                )
                machine.cavity_transmon_pairs[pair_id] = pair
                for line_type, ports in wiring_by_line_type.items():
                    wiring_path = (
                        f"#/wiring/cavity_transmon_pairs/{pair_id}/{line_type}"
                    )
                    if line_type == "sideband":
                        add_cavity_sideband_drive_component(pair, wiring_path, ports)


def add_twpas(machine: AnyQuam):
    """Adds TWPAs to the machine based on the wiring configuration.

    For each TWPA in ``machine.wiring["twpas"]``, creates a TWPA component and builds its pump
    channel(s) from the wiring. Supported line keys: ``"pump"`` (sticky, continuous pump) and
    ``"pump_"`` (non-sticky, calibration); the ``WiringLineType.TWPA_PUMP`` value ("p") is also
    accepted and mapped to ``"pump"``. No-op when the wiring has no ``twpas`` section.

    Args:
        machine (AnyQuam): The QuAM to which the TWPAs will be added.
    """
    for element_type, wiring_by_element in machine.wiring.items():
        if element_type != "twpas":
            continue
        machine.active_twpa_names = []
        for twpa_id, wiring_by_line_type in wiring_by_element.items():
            twpa = machine.twpa_type(id=twpa_id)
            machine.twpas[twpa_id] = twpa
            for line_type, ports in wiring_by_line_type.items():
                wiring_path = f"#/wiring/{element_type}/{twpa_id}/{line_type}"
                if line_type in ("pump", WiringLineType.TWPA_PUMP.value):
                    add_twpa_pump_component(
                        twpa, wiring_path, ports, attr="pump", sticky=True
                    )
                elif line_type == "pump_":
                    add_twpa_pump_component(
                        twpa, wiring_path, ports, attr="pump_", sticky=False
                    )
                else:
                    raise ValueError(f"Unknown TWPA line type: {line_type}")
            machine.active_twpa_names.append(twpa.name)


def add_pulses(machine: AnyQuam):
    """Adds default pulses to the transmon qubits, qubit pairs, and TWPAs in the machine.

    Args:
        machine (AnyQuam): The QuAM to which the pulses will be added.
    """
    if hasattr(machine, "qubits"):
        for transmon in machine.qubits.values():
            add_default_transmon_pulses(transmon)

    if hasattr(machine, "qubit_pairs"):
        for qubit_pair in machine.qubit_pairs.values():
            add_default_transmon_pair_pulses(qubit_pair)

    if hasattr(machine, "twpas"):
        for twpa in machine.twpas.values():
            add_default_twpa_pulses(twpa)


def add_octaves(
    machine: AnyQuam, calibration_db_path: Optional[Union[Path, str]] = None
) -> AnyQuam:
    """Adds octave components to the machine based on the wiring configuration and initializes their frequency converters.

    Args:
        machine (AnyQuam): The QuAM to which the octaves will be added.
        calibration_db_path (Optional[Union[Path, str]]): The path to the calibration database.

    Returns:
        AnyQuam: The QuAM with the added octaves.
    """
    if calibration_db_path is None:
        serializer = machine.get_serialiser()
        calibration_db_path = serializer._get_state_path().parent

    if isinstance(calibration_db_path, str):
        calibration_db_path = Path(calibration_db_path)

    for wiring_by_element in machine.wiring.values():
        for qubit, wiring_by_line_type in wiring_by_element.items():
            for line_type, references in wiring_by_line_type.items():
                for reference in references:
                    if "octaves" in references.get_unreferenced_value(reference):
                        octave_name = references.get_unreferenced_value(
                            reference
                        ).split("/")[2]
                        octave = Octave(
                            name=octave_name,
                            calibration_db_path=str(calibration_db_path),
                        )
                        machine.octaves[octave_name] = octave
                        octave.initialize_frequency_converters()

    return machine


def add_external_mixers(machine: AnyQuam) -> AnyQuam:
    """Adds external mixers to the machine based on the wiring configuration.

    Args:
        machine (AnyQuam): The QuAM to which the external mixers will be added.

    Returns:
        AnyQuam: The QuAM with the added external mixers.
    """
    for wiring_by_element in machine.wiring.values():
        for qubit, wiring_by_line_type in wiring_by_element.items():
            for line_type, references in wiring_by_line_type.items():
                for reference in references:
                    if "mixers" in references.get_unreferenced_value(reference):
                        mixer_name = references.get_unreferenced_value(reference).split(
                            "/"
                        )[2]
                        transmon_channel = {
                            WiringLineType.DRIVE.value: "xy",
                            WiringLineType.RESONATOR.value: "resonator",
                        }
                        frequency_converter = FrequencyConverter(
                            local_oscillator=LocalOscillator(),
                            mixer=StandaloneMixer(
                                intermediate_frequency=f"#/qubits/{qubit}/{transmon_channel[line_type]}/intermediate_frequency",
                            ),
                        )
                        machine.mixers[mixer_name] = frequency_converter

    return machine
