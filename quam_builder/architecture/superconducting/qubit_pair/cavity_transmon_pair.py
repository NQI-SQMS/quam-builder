"""QUAM component for a qubit–cavity dispersive coupling pair.

Stores the hardware and calibration parameters that describe the coupling
between one transmon qubit and one storage cavity mode (e.g. Alice or Bob),
including the dedicated IQ drive channel used for the |f,0⟩↔|g,1⟩ sideband
(f0g1) transition.

This component is the cavity analogue of
:class:`~quam_builder.architecture.superconducting.qubit_pair.FixedFrequencyTransmonPair`:
it represents a *qubit–cavity* coupling rather than a *qubit–qubit* coupling,
and stores the per-pair calibration data that cannot live on either the qubit
or the cavity alone.
"""
from dataclasses import field
from typing import Any, Dict, Optional

from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ

__all__ = ["CavityTransmonPair"]


@quam_dataclass
class CavityTransmonPair(QuamComponent):
    """QUAM component describing the dispersive coupling between a transmon qubit
    and a single storage cavity mode.

    This object stores:

    * The names of the coupled qubit and cavity mode (used to look them up in the
      QUAM root ``qubits`` and ``cavities`` dictionaries).
    * Calibrated coupling parameters: dispersive shift ``chi`` and displacement
      calibration constant ``displacement_k``.
    * The dedicated sideband drive channel (``sideband_drive``) used to drive the
      |f,0⟩↔|g,1⟩ transition for photon-number-resolved measurements.

    **Calibration workflow:**

    1. Node 21 (f0g1 spectroscopy) finds the sideband RF frequency and stores it
       in ``sideband_drive.RF_frequency``.
    2. Node 22 (f0g1 Rabi / power Rabi) calibrates the sideband π-pulse amplitude
       and stores it in ``sideband_drive.operations["f0g1_pi"].amplitude``.
    3. Node 26 / 28 measure the dispersive shift ``chi``.
    4. Node 28 (PNS displacement calibration) or 30 (Ramsey displacement calibration)
       calibrates the displacement constant ``displacement_k``.

    Attributes:
        qubit_name:       Name of the coupled transmon qubit (e.g. ``"q1"``).
                          Must match a key in the QUAM root ``qubits`` dict.
        cavity_mode_name: Name of the cavity mode (e.g. ``"alice"`` or ``"bob"``).
                          Must match a key in the parent ``Cavity`` object.
        chi:              Dispersive shift χ/2π [Hz].
                          Convention: ω_q(n) = ω_q − 2χ·n, where n is the cavity
                          photon number.  Calibrated by nodes 26 / 28 / 29.
                          ``None`` until first calibration.
        displacement_k:   Displacement calibration constant k such that the mean
                          photon number n̄ = k · A², where A is the
                          ``displacement`` pulse ``amplitude_scale``.
                          Calibrated by node 28 or 30.  ``None`` until first calibration.
        sideband_drive:   IQ drive channel for the |f,0⟩↔|g,1⟩ sideband transition.
                          Typically wired to a dedicated Octave RF output with an
                          external LO at ~8 GHz; the IF spans ±500 MHz to reach
                          RF_frequency ≈ 2·f_ge + α − f_cavity (~3–4 GHz).
                          ``None`` if no sideband drive is wired for this pair.
        extras:           Free-form metadata dict for project-specific extensions.
    """

    qubit_name: str
    cavity_mode_name: str
    chi: Optional[float] = None
    displacement_k: Optional[float] = None
    sideband_drive: Optional[XYDriveIQ] = None
    extras: Dict[str, Any] = field(default_factory=dict)
