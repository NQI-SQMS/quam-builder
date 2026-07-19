from qualang_tools.wirer import Connectivity
from qualang_tools.wirer.connectivity.channel_spec import ChannelSpec
from qualang_tools.wirer.connectivity.element import Element, Reference
from qualang_tools.wirer.connectivity.wiring_spec import WiringFrequency, WiringIOType

CAVITY_LINE = "cavity"
SIDEBAND_LINE = "sideband"


class CavityConnectivity(Connectivity):
    """Extends Connectivity with convenience methods for cavity mode drives and sideband drives.

    Uses string line-type sentinels ("cavity", "sideband") which are natively
    supported by qualang_tools.wirer — no package modifications required.
    """

    def add_cavity_mode_drive_lines(
        self,
        cavity_id: str,
        mode_names: list,
        constraints: ChannelSpec = None,
        triggered: bool = True,
    ):
        """Register IQ/MW drive lines for one or more cavity modes.

        Element IDs are "{cavity_id}_{mode_name}", e.g. "c1_alice".

        Args:
            cavity_id: Cavity identifier (e.g. "c1").
            mode_names: List of mode names (e.g. ["alice", "bob"]).
            constraints: Optional channel spec to pin the allocation.
            triggered: Whether digital trigger outputs are included.
        """
        elements = [self._get_or_create_ref(f"{cavity_id}_{m}") for m in mode_names]
        return self.add_wiring_spec(
            WiringFrequency.RF,
            WiringIOType.OUTPUT,
            CAVITY_LINE,
            triggered,
            constraints,
            elements,
        )

    def add_cavity_sideband_lines(
        self,
        pair_ids: list,
        constraints: ChannelSpec = None,
        triggered: bool = True,
    ):
        """Register IQ/MW sideband drive lines for qubit-cavity pairs.

        pair_ids format: "{qubit_name}_{mode_name}", e.g. "q1_alice".

        Args:
            pair_ids: List of pair identifiers (e.g. ["q1_alice", "q1_bob"]).
            constraints: Optional channel spec to pin the allocation.
            triggered: Whether digital trigger outputs are included.
        """
        elements = [self._get_or_create_ref(p) for p in pair_ids]
        return self.add_wiring_spec(
            WiringFrequency.RF,
            WiringIOType.OUTPUT,
            SIDEBAND_LINE,
            triggered,
            constraints,
            elements,
        )

    def _get_or_create_ref(self, name: str) -> Element:
        ref = Reference(name)
        if ref not in self.elements:
            self.elements[ref] = Element(ref)
        return self.elements[ref]
