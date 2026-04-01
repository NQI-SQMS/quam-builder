from typing import Dict, Any, Union, Optional
from dataclasses import field

from quam.core import quam_dataclass
from quam.components.quantum_components import Qubit

from quam_builder.architecture.superconducting.cavity.cavity_mode import CavityMode


__all__ = ["Cavity"]


@quam_dataclass
class Cavity(Qubit):
    """
    QUAM component representing a superconducting cavity with one or more modes.

    The Cavity is a container for CavityMode objects (e.g. alice, bob).  Each
    CavityMode holds the drive channel, coherence times, chi, etc. that are
    specific to that mode.

    Attributes:
        id (Union[int, str]): Cavity identifier.
            Integer ids are prefixed with the class default label.
        alice (CavityMode): The first cavity mode. Default is None.
        bob (CavityMode): The second cavity mode. Default is None.
        grid_location (str): Location in the plotting grid as "column, row".
        extras (Dict[str, Any]): Arbitrary extra metadata.
    """

    id: Union[int, str]

    alice: CavityMode = None
    bob: CavityMode = None

    grid_location: str = None
    extras: Dict[str, Any] = field(default_factory=dict)
