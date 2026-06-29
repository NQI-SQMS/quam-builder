from dataclasses import field
from typing import Dict

from quam.core import quam_dataclass

from quam_builder.architecture.superconducting.qubit import (
    FluxTunableTransmon,
)
from quam_builder.architecture.superconducting.qubit_pair import (
    FluxTunableTransmonPair,
    CavityTransmonPair,
)
from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
    FluxTunableQuam,
)
from quam_builder.architecture.superconducting.cavity.cavity import Cavity

__all__ = [
    "CavityQuam",
    "FluxTunableQuam",
    "FluxTunableTransmon",
    "FluxTunableTransmonPair",
    "CavityTransmonPair",
    "Cavity",
]


@quam_dataclass
class CavityQuam:
    """SRF-cavity QUAM mixin: adds cavities/cavity-transmon coupling state on top of
    whichever qubit-type QPU class it's combined with — combine via multiple
    inheritance, e.g. ``class Quam(CavityQuam, FluxTunableQuam): ...``.

    Note: this mixin does not itself activate TWPAs. `FluxTunableQuam.initialize_qpu`
    already loops over `self.twpas` and calls `.initialize()`, so it's inherited for
    free when combined with `FluxTunableQuam`. If combined with a QPU class whose
    `initialize_qpu` doesn't activate TWPAs (e.g. plain `FixedFrequencyQuam`), add
    that call explicitly in the concrete `Quam` class.

    Attributes:
        cavities (Dict[str, Cavity]): SRF storage cavities (e.g. alice, bob) coupled to the qubits.
        cavity_transmon_pairs (Dict[str, CavityTransmonPair]): qubit-cavity coupling parameters
            (chi, displacement k).
    """

    cavities: Dict[str, Cavity] = field(default_factory=dict)
    cavity_transmon_pairs: Dict[str, CavityTransmonPair] = field(default_factory=dict)
