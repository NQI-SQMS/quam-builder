"""
Concrete strategy implementations for different quantum element types.

This module provides specific wiring strategies for:
- Single qubits
- Qubit pairs
- Global elements
- Readout elements
"""

from typing import Dict

from qualang_tools.wirer.connectivity.element import QubitPairReference

from quam_builder.builder.qop_connectivity.wiring_strategy import WiringStrategy, WiringContext
from quam_builder.builder.qop_connectivity.paths import QUBITS_BASE_JSON_PATH


class QubitWiringStrategy(WiringStrategy):
    """Strategy for single-qubit wiring.

    Handles wiring for individual qubits including drive, flux, resonator,
    and plunger lines.
    """

    def get_base_path(self, context: WiringContext) -> str:
        """Get base path for qubit wiring.

        Format: qubits/{qubit_id}/{line_type}

        Args:
            context: The wiring context

        Returns:
            Base path string
        """
        return f"qubits/{context.element_id}/{context.line_type.value}"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Get additional references for qubits.

        Qubits don't need additional references beyond their port mappings.

        Args:
            context: The wiring context

        Returns:
            Empty dictionary
        """
        return {}


class QubitPairWiringStrategy(WiringStrategy):
    """Strategy for qubit-pair wiring.

    Handles wiring for qubit pairs including coupler, cross-resonance,
    ZZ-drive, and barrier lines. Includes references to control and target qubits.
    """

    def get_base_path(self, context: WiringContext) -> str:
        """Get base path for qubit pair wiring.

        Format: qubit_pairs/{pair_id}/{line_type}

        Args:
            context: The wiring context

        Returns:
            Base path string
        """
        return f"qubit_pairs/{context.element_id}/{context.line_type.value}"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Get additional references for qubit pairs.

        Adds control_qubit and target_qubit JSON references.

        Args:
            context: The wiring context

        Returns:
            Dictionary with control_qubit and target_qubit references
        """
        if not isinstance(context.element_id, QubitPairReference):
            raise TypeError(f"Expected QubitPairReference, got {type(context.element_id)}")

        element_id = context.element_id
        return {
            "control_qubit": f"{QUBITS_BASE_JSON_PATH}/q{element_id.control_index}",
            "target_qubit": f"{QUBITS_BASE_JSON_PATH}/q{element_id.target_index}",
        }


class GlobalElementWiringStrategy(WiringStrategy):
    """Strategy for global element wiring.

    Handles wiring for global gates that affect multiple qubits simultaneously.
    """

    def get_base_path(self, context: WiringContext) -> str:
        """Get base path for global element wiring.

        Format: globals/{element_id}/{line_type}

        Args:
            context: The wiring context

        Returns:
            Base path string
        """
        return f"globals/{context.element_id}/{context.line_type.value}"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Get additional references for global elements.

        Global elements don't need additional references.

        Args:
            context: The wiring context

        Returns:
            Empty dictionary
        """
        return {}


class TwpaWiringStrategy(WiringStrategy):
    """Strategy for TWPA wiring.

    A TWPA pump is allocated as a single line (``WiringLineType.TWPA_PUMP``, value "p"), but the
    QUAM ``TWPA`` component has two MW elements that share the same physical port: a sticky
    ``pump`` (continuous) and a non-sticky ``pump_`` (calibration). This strategy therefore emits
    BOTH sub-paths pointing to the same allocated port, matching the known-good KRISS TWPA wiring:

        twpas/{twpa_id}/pump/opx_output
        twpas/{twpa_id}/pump_/opx_output
    """

    def _twpa_id(self, context: WiringContext) -> str:
        # add_twpa_lines names the element by its index (e.g. "twpaA"); fall back to str().
        return str(getattr(context.element_id, "index", context.element_id))

    def get_base_path(self, context: WiringContext) -> str:
        """Format: twpas/{twpa_id} (the pump/pump_ sub-keys are added in generate_wiring)."""
        return f"twpas/{self._twpa_id(context)}"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """TWPAs need no additional references."""
        return {}

    def generate_wiring(self, context: WiringContext) -> Dict[str, str]:
        """Duplicate the single pump channel into pump (sticky) and pump_ (calibration) sub-paths."""
        wiring: Dict[str, str] = {}
        for channel in context.channels:
            if self._should_process_channel(channel):
                key, reference = self.port_factory.create_port_reference(channel, context)
                wiring[f"pump/{key}"] = reference
                wiring[f"pump_/{key}"] = reference
        return wiring


class CavityModeWiringStrategy(WiringStrategy):
    """Strategy for SRF cavity-mode drive wiring.

    Element IDs are ``"{cavity_id}_{mode_name}"`` (e.g. ``"c1_alice"``), matching
    ``CavityConnectivity.add_cavity_mode_drive_lines``. Mirrors the legacy
    ``create_wiring`` implementation's handling of ``CAVITY_LINE``:

        cavities/{cavity_id}/{mode_name}/cavity/{k}
    """

    def get_base_path(self, context: WiringContext) -> str:
        cavity_id, mode_name = str(context.element_id).split("_", 1)
        return f"cavities/{cavity_id}/{mode_name}/cavity"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Cavity modes need no additional references."""
        return {}


class SidebandWiringStrategy(WiringStrategy):
    """Strategy for qubit-cavity sideband drive wiring.

    Element IDs are ``"{qubit_name}_{mode_name}"`` pair ids (e.g. ``"q1_alice"``),
    matching ``CavityConnectivity.add_cavity_sideband_lines``. Mirrors the legacy
    ``create_wiring`` implementation's handling of ``SIDEBAND_LINE``:

        cavity_transmon_pairs/{pair_id}/sideband/{k}
    """

    def get_base_path(self, context: WiringContext) -> str:
        return f"cavity_transmon_pairs/{context.element_id}/sideband"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Sideband drives need no additional references."""
        return {}


class ReadoutWiringStrategy(WiringStrategy):
    """Strategy for readout wiring.

    Handles wiring for readout elements including sensor gates and
    RF resonators.
    """

    def get_base_path(self, context: WiringContext) -> str:
        """Get base path for readout wiring.

        Format: readout/{element_id}/{line_type}

        Args:
            context: The wiring context

        Returns:
            Base path string
        """
        return f"readout/{context.element_id}/{context.line_type.value}"

    def get_additional_references(self, context: WiringContext) -> Dict[str, str]:
        """Get additional references for readout elements.

        Readout elements don't need additional references.

        Args:
            context: The wiring context

        Returns:
            Empty dictionary
        """
        return {}
