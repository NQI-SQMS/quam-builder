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

__all__ = ["CavityTransmonPair", "SidebandTransition"]


@quam_dataclass
class SidebandTransition(QuamComponent):
    """Calibration data for a single f{k}g{k+1} sideband transition.

    Each entry in ``CavityTransmonPair.transitions`` stores calibration data for
    one f{k}g{k+1} sideband transition.  The key is ``"f{k}g{k+1}"``
    (e.g. ``"f0g1"``, ``"f1g2"``).

    Calibration workflow per transition k:

    1. Node 26  (fNgN1 spectroscopy)      → ``RF_frequency``
    2. Node 26b (fNgN1 time Rabi)         → ``pi_flat_top_length_ns``, ``rabi_rate_hz``
    3. Node 26c (fNgN1 Ramsey)            → refined ``RF_frequency``
    4. Node 26d (ge IQ blobs @ Fock k+1)  → ``ge_iq_threshold``
    5. Node 26e (qubit ge @ Fock k)       → ``delta_f_focka`` (and ``CavityTransmonPair.chi`` when k=0)
    6. Node 26f (ge Ramsey @ Fock k)      → refined ``delta_f_focka`` (and ``CavityTransmonPair.chi`` when k=0)
    7. Node 26g (qubit ef @ Fock k)       → ``ef_delta_f_focka``
    8. Node 26h (ef Ramsey @ Fock k)      → refined ``ef_delta_f_focka``
    9. Node 26i (resonator spec @ Fock k) → ``resonator_f_fock_hz``
   10. Node 35  (cavity reset test)        → ``sideband_cooling_time`` (f0g1 only)

    Attributes:
        RF_frequency:         Calibrated sideband RF frequency [Hz].
        pi_flat_top_length_ns: Flat-top duration [ns] of the shaped (ramp + flat-top
                              + ramp) π-pulse.  The total pulse length is
                              ``pi_flat_top_length_ns + 2 × ramp_length``.
        rabi_rate_hz:         Sideband Rabi frequency [Hz] extracted from the
                              time-Rabi fit (node 26b).
        ge_iq_threshold:      Readout I-quadrature threshold [QUA demod units] for
                              qubit ge state discrimination when the cavity is in
                              Fock state |k+1⟩.  Required because the large chi
                              dispersive shift (~282 MHz) moves the readout resonator
                              IQ response, invalidating the vacuum-calibrated threshold.
                              Calibrated by node 26d.
        delta_f_focka:        Nonlinear correction [Hz] to the qubit ge dispersive shift
                              at Fock state |k+1⟩, defined as the deviation from the
                              linear approximation: ``delta_f_focka = chi_measured - (k+1) × chi``,
                              where ``chi`` is the per-photon dispersive shift stored on
                              ``CavityTransmonPair``.  By convention ``delta_f_focka = 0``
                              for the f0g1 transition (k=0), since ``chi`` itself is
                              defined from that measurement.  Non-zero values at higher k
                              capture Kerr nonlinearity.  Calibrated by node 26e.
        ef_delta_f_focka:     Deviation [Hz] of the qubit ef transition frequency from the
                              vacuum anharmonicity when the cavity is in Fock state |k+1⟩,
                              defined as ``[ω_ef(k+1) - ω_ge(k+1)] - α``, where α is the
                              bare anharmonicity.  Zero means the ef-ge gap equals α; a
                              non-zero value is the photon-number-dependent Kerr correction
                              to the ef transition.  Calibrated by node 26g.
        T2_star_ns:           Sideband coherence time T2* [ns].
        resonator_f_fock_hz:  Readout resonator RF frequency [Hz] when the storage
                              cavity is in Fock state |k+1⟩.  Tracks the photon-number-
                              dependent frequency shift of the readout resonator due to
                              higher-order qubit-cavity-resonator cross-Kerr coupling.
                              Calibrated by node 26i.
        sideband_cooling_time: Flat-top duration [ns] at which the sideband drive
                              achieves ≥95% cavity cooling probability (t95), measured
                              by node 35 (cavity reset test).  Used as the default
                              sideband pulse duration during active cavity cooling when
                              no explicit duration is passed.  ``None`` until calibrated
                              by node 35.  Currently only populated for the f0g1
                              transition.  Must be a multiple of 4 ns.
    """

    RF_frequency: Optional[float] = None
    pi_flat_top_length_ns: Optional[int] = None
    rabi_rate_hz: Optional[float] = None
    ge_iq_threshold: Optional[float] = None
    delta_f_focka: Optional[float] = None
    ef_delta_f_focka: Optional[float] = None
    T2_star_ns: Optional[float] = None
    resonator_f_fock_hz: Optional[float] = None
    sideband_cooling_time: Optional[int] = None


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
    * Per-transition sideband calibration data in ``transitions`` (replaces the old
      ad-hoc ``extras`` dict for sideband parameters).

    **Calibration workflow:**

    1. Node 21 (f0g1 spectroscopy) finds the sideband RF frequency and stores it
       in ``sideband_drive.RF_frequency`` and ``transitions["f0g1"].RF_frequency``.
    2. Node 22 (f0g1 Rabi / power Rabi) calibrates the sideband π-pulse amplitude
       and stores it in ``sideband_drive.operations["f0g1_pi"].amplitude``.
    3. Node 26 / 28 measure the dispersive shift ``chi``.
    4. Node 28 (PNS displacement calibration) or 30 (Ramsey displacement calibration)
       calibrates the displacement constant ``displacement_k``.
    5. Nodes 28–28f calibrate per-transition sideband data stored in ``transitions``.

    Attributes:
        qubit_name:       Name of the coupled transmon qubit (e.g. ``"q1"``).
                          Must match a key in the QUAM root ``qubits`` dict.
        cavity_mode_name: Name of the cavity mode (e.g. ``"alice"`` or ``"bob"``).
                          Must match a key in the parent ``Cavity`` object.
        chi:              Full per-photon qubit frequency shift [Hz], defined as
                          ω_q(n) = ω_q + chi·n.  Stored with its physical sign:
                          for typical transmon-cavity systems chi < 0 (more
                          photons lower the qubit frequency).
                          ``|chi|`` equals the PNRS peak spacing (n=0→n=1).
                          Calibrated by nodes 24 / 25 / 28 / 30.
                          ``None`` until first calibration.
        displacement_k:   Displacement calibration constant k such that the mean
                          photon number n̄ = k · A², where A is the
                          ``displacement`` pulse ``amplitude_scale``.
                          Calibrated by node 28 or 30.  ``None`` until first calibration.
        displacement_alpha_max: Maximum photon number (|α|) reachable with amplitude_scale=1.
                          Computed by node 22 from the OPX+ DAC ceiling (0.5 V) and
                          firmware headroom (1.9): alpha_max = 0.5 / (base_amp × sigma × 1.9).
                          The displacement pulse amplitude is set to base_amp × sigma × alpha_max
                          ≈ 0.263 V, so amplitude_scale ∈ [0, 1.9] spans [0, alpha_max × 1.9]
                          photons.  ``None`` until first calibration by node 22.
        parity_time:      Experimentally calibrated parity time τ [seconds] for
                          Wigner tomography.  This is the dispersive Ramsey wait
                          duration such that each cavity photon imprints phase π
                          on the qubit: χ_eff · τ = π.  Differs from the
                          analytical estimate 1/(2·|chi|) due to AC Stark shifts,
                          higher-order dispersive terms, and finite pulse lengths.
                          Calibrated by node 30.  ``None`` until first calibration.
        sideband_drive:   IQ drive channel for the |f,0⟩↔|g,1⟩ sideband transition.
                          Typically wired to a dedicated Octave RF output with an
                          external LO at ~8 GHz; the IF spans ±500 MHz to reach
                          RF_frequency ≈ 2·f_ge + α − f_cavity (~3–4 GHz).
                          ``None`` if no sideband drive is wired for this pair.
        transitions:      Per-transition sideband calibration data, keyed by
                          ``"f{k}g{k+1}"`` (e.g. ``"f0g1"``, ``"f1g2"``).
                          Each value is a :class:`SidebandTransition` holding the
                          calibrated RF frequency, π-pulse shape, Rabi rate, and
                          Fock-state-dependent qubit frequency shifts for that transition.
                          Populated by nodes 28–28f.
        extras:           Free-form metadata dict for ad-hoc parameters that do not
                          belong to any other field.  Sideband calibration data
                          should live in ``transitions``.
    """

    qubit_name: str
    cavity_mode_name: str

    @property
    def name(self) -> str:
        if self.parent is not None:
            return self.parent.get_attr_name(self)
        raise AttributeError(
            f"Cannot determine name of CavityTransmonPair: no parent set"
        )

    chi: Optional[float] = None
    displacement_k: Optional[float] = None
    displacement_alpha_max: Optional[float] = None
    parity_time: Optional[float] = None
    sideband_drive: Optional[XYDriveIQ] = None
    transitions: Dict[str, SidebandTransition] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


