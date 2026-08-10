"""Cavity QUA operations: displacement pulse and SNAP gate.

Both operations are collected here so all cavity-mode QUA primitives live in one
place.  ``CavityMode`` (in ``cavity_mode.py``) exposes them via:

* ``cavity_mode.displacement(...)``        — thin wrapper around ``_play_displacement``
* ``cavity_mode.snap_gate.apply(...)``      — ``SNAPGate`` QuAM component
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import field
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ, XYDriveMW

from qm.qua import align, amp, frame_rotation, play, reset_if_phase, strict_timing_

if TYPE_CHECKING:
    pass  # CavityMode imported only for type hints to avoid circular import

__all__ = ["SNAPElementDrive", "SNAPGate"]

_logger = logging.getLogger(__name__)


def _play_displacement(
    cavity_mode_drive,
    amplitude=None,
    alpha_re=None,
    alpha_im=None,
    length: Optional[int] = None,
) -> None:
    """QUA displacement pulse on *cavity_mode_drive*.

    Args:
        cavity_mode_drive: The ``XYDriveIQ`` / ``XYDriveMW`` channel to play on.
        amplitude: Real amplitude scale factor (Python ``float`` or QUA ``fixed``).
            Used for simple displacements where the phase is fixed by the pulse
            calibration.  Ignored when *alpha_re* or *alpha_im* is given.
        alpha_re: In-phase (I) amplitude component (Python float or QUA fixed variable).
            Together with *alpha_im* this sets an arbitrary complex displacement via the
            QUA ``amp()`` IQ matrix.
        alpha_im: Quadrature (Q) amplitude component, paired with *alpha_re*.
        length: Optional pulse duration override in QUA clock cycles (4 ns each).
    """
    dur_kwarg: dict = {"duration": length} if length is not None else {}

    if alpha_re is not None or alpha_im is not None:
        _re = alpha_re if alpha_re is not None else 0.0
        _im = alpha_im if alpha_im is not None else 0.0
        play("displacement" * amp(_re, 0, _im, 0), cavity_mode_drive.name, **dur_kwarg)
    elif amplitude is not None:
        cavity_mode_drive.play("displacement", amplitude_scale=amplitude, **dur_kwarg)
    else:
        cavity_mode_drive.play("displacement", **dur_kwarg)


@quam_dataclass
class SNAPElementDrive(XYDriveIQ):
    """XYDriveIQ element for a parallel SNAP gate at a fixed Fock level.

    The intermediate frequency is computed live from QUAM references rather than
    stored as a fixed integer.  This means it automatically stays correct whenever
    ``chi`` is re-calibrated — no need to recreate the elements.

    Fields ``chi_hz`` and ``delta_f_focka_hz`` should be set as absolute QUAM
    references when the element is created::

        el = SNAPElementDrive(
            opx_output_I=qubit.xy.opx_output_I,
            opx_output_Q=qubit.xy.opx_output_Q,
            frequency_converter_up=qubit.xy.frequency_converter_up,
            RF_frequency="#/qubits/q1/xy/RF_frequency",
            fock_level=k,
            chi_hz="#/cavity_transmon_pairs/q1_alice/chi",
            delta_f_focka_hz="#/cavity_transmon_pairs/q1_alice/transitions/f0g1/delta_f_focka",
        )

    When ``machine.generate_config()`` is called, QUAM resolves the references to
    their current values and ``inferred_intermediate_frequency`` computes::

        IF = (RF_frequency − LO_frequency) + fock_level × chi + delta_f_focka
           = qubit_IF + fock_level × chi + delta_f_focka

    Attributes:
        fock_level: Fock level index (0-based) this element addresses.
        chi_hz: Per-photon dispersive shift [Hz].  Set as a QUAM reference to
            ``CavityTransmonPair.chi`` so updates propagate automatically.
            ``None`` is treated as 0 Hz (vacuum, no shift).
        delta_f_focka_hz: Kerr correction [Hz] at Fock |fock_level⟩ (deviation
            from linear chi·n).  Set as a QUAM reference to
            ``SidebandTransition.delta_f_focka`` for Fock level ``fock_level − 1``
            (relevant for fock_level > 0).  ``None`` or null in state means 0 Hz
            (correction not yet calibrated — uses linear chi only).
    """

    fock_level: int = 0
    chi_hz: Optional[float] = None
    delta_f_focka_hz: Optional[float] = None

    @property
    def inferred_intermediate_frequency(self) -> int:
        """Compute IF = qubit_base_IF + fock_level × chi + delta_f_focka."""
        # base_if = RF_frequency (QUAM ref to qubit.xy.RF_frequency) - LO_frequency
        base_if = super().inferred_intermediate_frequency

        chi_raw = self.chi_hz
        chi = float(chi_raw) if chi_raw is not None else 0.0

        delta_raw = self.delta_f_focka_hz
        delta = float(delta_raw) if delta_raw is not None else 0.0

        return int(base_if + self.fock_level * chi + delta)


@quam_dataclass
class SNAPGate(QuamComponent):
    """Parallel SNAP gate for a single cavity mode.

    Each Fock level is assigned a :class:`SNAPElementDrive` element whose
    intermediate frequency auto-tracks ``chi`` via QUAM references.  All
    elements share the same physical output port as ``qubit.xy``; their signals
    sum at the DAC, implementing simultaneous, frequency-division-multiplexed
    selective pi pulses.

    **Usage in QUA programs**::

        # Apply pi phase to Fock |1> (D-SNAP step):
        cavity_mode.snap_gate.apply_single(fock_level=1, theta=np.pi,
                                           pair=pair, qubit=qubit)

        # General SNAP with arbitrary phases on levels 0–2:
        cavity_mode.snap_gate.apply(thetas=[0, np.pi, np.pi/2],
                                    pair=pair, qubit=qubit)

    **Initialisation** — run the "Initialise SNAP gate elements" cell in the
    calibration notebook after chi is calibrated (node 25/28).  Uses
    :class:`SNAPElementDrive` so the IF stays in sync with future chi updates.

    Attributes:
        snap_elements: Dict keyed ``"n0"``, ``"n1"``, … for Fock levels 0, 1, …
            Each value is a :class:`SNAPElementDrive` whose ``chi_hz`` field is a
            QUAM reference to ``CavityTransmonPair.chi``.
    """

    snap_elements: Dict[str, SNAPElementDrive] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        thetas: Sequence[float],
        pair,
        qubit,
        phi0: float = 0.0,
    ) -> None:
        """Apply the SNAP gate with per-Fock-level phases.

        For each Fock level *n* whose ``|thetas[n]| > 1e-12``:
          1. ``frame_rotation(phi0)``
          2. ``play("selective_x180")``  (at fixed IF of snap_elements["n{n}"])
          3. ``frame_rotation(phi0 + π − thetas[n])``  (cumulative)
          4. ``play("selective_x180")``
          5. ``reset_if_phase()``

        With ``phi0=0, thetas[n]=π`` (D-SNAP convention) the two
        ``frame_rotation(0)`` calls are no-ops, reproducing the bare two-pi-pulse
        behaviour.

        All alignment with ``qubit.xy`` and ``cavity_mode_drive`` is handled
        internally — callers need **no** ``align()`` around this call.

        Args:
            thetas: Phase angles [rad] indexed by Fock level.  Entries whose
                absolute value is below 1e-12 are skipped (element stays idle).
            pair: The :class:`~quam_builder.architecture.superconducting
                .qubit_pair.cavity_transmon_pair.CavityTransmonPair` connecting
                this cavity mode to *qubit*.  Kept for API consistency and
                validation.
            qubit: The coupled transmon qubit object (provides ``qubit.xy``).
            phi0: Global reference frame angle [rad] applied before each pair
                of pi pulses.  Default ``0.0``.
        """
        if not self.snap_elements:
            raise ValueError(
                "snap_gate.snap_elements is empty.  Run the 'Initialise SNAP gate "
                "elements' cell in the calibration notebook after calibrating chi "
                "(node 25/28)."
            )

        cavity_mode = self.parent  # CavityMode
        cav_drive = cavity_mode.cavity_mode_drive
        snap_names = [el.name for el in self.snap_elements.values()]
        all_names = snap_names + [cav_drive.name, qubit.xy.name]

        align(*all_names)

        with strict_timing_():
            for n, el in enumerate(self.snap_elements.values()):
                if n >= len(thetas):
                    break
                th = float(thetas[n])
                if abs(th) < 1e-12:
                    continue
                frame_rotation(phi0, el.name)
                el.play("selective_x180")
                frame_rotation(phi0 + np.pi - th, el.name)  # cumulative
                el.play("selective_x180")
                reset_if_phase(el.name)

        align(*all_names)

    def apply_single(
        self,
        fock_level: int,
        theta: float,
        pair,
        qubit,
        phi0: float = 0.0,
    ) -> None:
        """Convenience wrapper: apply SNAP to a single Fock level.

        Equivalent to ``apply([0]*fock_level + [theta], pair, qubit, phi0)``.

        Args:
            fock_level: Index of the Fock level to address (0-based).
            theta: Phase [rad] to apply to ``|fock_level⟩``.
            pair: ``CavityTransmonPair`` for the qubit-cavity coupling.
            qubit: The coupled transmon qubit object.
            phi0: Reference frame angle [rad].
        """
        thetas = [0.0] * fock_level + [float(theta)]
        self.apply(thetas, pair, qubit, phi0)
