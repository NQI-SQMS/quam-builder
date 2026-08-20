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
import logging
from dataclasses import field
from typing import Any, Dict, Optional

from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ
from qm.qua import align, assign, declare, fixed, if_, else_, strict_timing_, wait, while_

_logger = logging.getLogger(__name__)

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
    ge_iq_threshold_displaced: Optional[float] = None
    """Readout I-quadrature threshold [QUA demod units] for qubit ge discrimination when the
    cavity holds a coherent (displaced) state.  Calibrated by node 26j.  Use this threshold
    in sideband/cavity nodes instead of the vacuum readout.threshold when the cavity is not
    in its ground state."""
    parity_time: Optional[float] = None
    parity_contrast: Optional[float] = None
    sideband_drive: Optional[XYDriveIQ] = None
    transitions: Dict[str, SidebandTransition] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Frequency helpers
    # ------------------------------------------------------------------

    def get_transition_rf(self, k: int) -> float:
        """Return the calibrated RF frequency [Hz] for the f{k}g{k+1} sideband transition.

        Priority: transitions[f{k}g{k+1}].RF_frequency → extras[f{k}g{k+1}_RF_frequency]
        → sideband_drive.RF_frequency.
        """
        tr = self.transitions.get(f"f{k}g{k+1}")
        if tr is not None and tr.RF_frequency is not None:
            return float(tr.RF_frequency)
        extras_key = f"f{k}g{k+1}_RF_frequency"
        if extras_key in (self.extras or {}):
            return float(self.extras[extras_key])
        return float(self.sideband_drive.RF_frequency)

    def ge_if_at_fock(self, qubit, k: int) -> int:
        """Return the qubit ge IF [Hz] when the cavity is in Fock |k⟩.

        Adds k × chi (linear dispersive shift) to the vacuum ge IF, then applies
        the nonlinear correction delta_f_focka from transitions["f{k-1}g{k}"] when
        calibrated (zero for k=0 by convention).
        """
        chi = self.chi if self.chi is not None else 0.0
        base = int(qubit.xy.intermediate_frequency) + int(k * chi)
        if k > 0:
            tr = self.transitions.get(f"f{k-1}g{k}")
            if tr is not None and tr.delta_f_focka is not None:
                return base + int(tr.delta_f_focka)
        return base

    def ef_if_at_fock(self, qubit, k: int) -> int:
        """Return the qubit ef IF [Hz] when the cavity is in Fock |k⟩.

        Adds the bare anharmonicity and photon-number-dependent Kerr correction
        ef_delta_f_focka (from transitions["f{k-1}g{k}"]) to ge_if_at_fock.
        """
        ge_if_k = self.ge_if_at_fock(qubit, k)
        if k > 0:
            tr = self.transitions.get(f"f{k-1}g{k}")
            if tr is not None and tr.ef_delta_f_focka is not None:
                return ge_if_k + int(qubit.anharmonicity) + int(tr.ef_delta_f_focka)
        chi = self.chi if self.chi is not None else 0.0
        return ge_if_k + int(qubit.anharmonicity) + int(k * chi)

    def resolve_sb_op(self, k: int) -> str:
        """Return the sideband operation name for transition f{k}g{k+1}.

        Returns the per-level operation "f{k}g{k+1}_pi" if it exists on
        sideband_drive, otherwise falls back to "sideband_flat_top".
        """
        per_level = f"f{k}g{k+1}_pi"
        return per_level if per_level in self.sideband_drive.operations else "sideband_flat_top"

    def play_sideband_flattop(
        self,
        flat_top_duration_clk=None,
        flat_top_duration_ns: Optional[int] = None,
        amplitude_scale=None,
    ) -> None:
        """Play the shaped sideband flat-top pulse inside ``strict_timing_()``.

        Plays ``sideband_ramp_up → sideband_square → sideband_ramp_down`` as a
        single coherent envelope.  The caller is responsible for setting the
        sideband drive IF to the correct frequency before this call.

        Args:
            flat_top_duration_clk: Flat-top duration in QUA clock cycles (4 ns
                each).  Accepts a Python ``int`` or a QUA variable.  When both
                this and ``flat_top_duration_ns`` are ``None``, the square pulse
                is played at its own default length.
            flat_top_duration_ns: Convenience alternative to
                ``flat_top_duration_clk``; duration in nanoseconds (must be a
                multiple of 4).  Converted to clock cycles automatically.
                Ignored when ``flat_top_duration_clk`` is also supplied.
            amplitude_scale: Multiplicative amplitude scaling applied uniformly
                to all three pulses.  Accepts a Python ``float`` or a QUA
                ``fixed`` variable.  When ``None`` the calibrated amplitudes are
                used unchanged.
        """
        sideband_drive = self.sideband_drive

        if flat_top_duration_clk is None and flat_top_duration_ns is not None:
            flat_top_duration_clk = flat_top_duration_ns // 4

        with strict_timing_():
            sideband_drive.play("sideband_ramp_up", amplitude_scale=amplitude_scale)
            if flat_top_duration_clk is not None:
                sideband_drive.play("sideband_square", amplitude_scale=amplitude_scale, duration=flat_top_duration_clk)
            else:
                sideband_drive.play("sideband_square", amplitude_scale=amplitude_scale)
            sideband_drive.play("sideband_ramp_down", amplitude_scale=amplitude_scale)

    # ------------------------------------------------------------------
    # QUA generation
    # ------------------------------------------------------------------

    def fock_prep_qua(
        self,
        fock_level: int,
        qubit,
    ) -> None:
        """Generate QUA statements to prepare Fock |fock_level⟩ in the cavity.

        Iterates j = 0 … fock_level-1; each step:
          - drives qubit ge → ef at Fock-j-corrected frequencies
          - plays the f{j}g{j+1} sideband π-pulse → cavity |j⟩ → |j+1⟩, qubit → |g⟩.
        """
        sideband_drive = self.sideband_drive
        for j in range(fock_level):
            sb_rf_j = self.get_transition_rf(j)
            if_offset_j = int(sb_rf_j - sideband_drive.RF_frequency)
            target_if_j = int(sideband_drive.intermediate_frequency) + if_offset_j

            tr_j = self.transitions.get(f"f{j}g{j+1}")
            flat_top_clk_j = (tr_j.pi_flat_top_length_ns // 4) if (tr_j and tr_j.pi_flat_top_length_ns) else None

            ge_if_j = self.ge_if_at_fock(qubit, j)
            ef_if_j = self.ef_if_at_fock(qubit, j)

            qubit.xy.update_frequency(ge_if_j)
            qubit.xy.play("x180")
            qubit.xy.update_frequency(ef_if_j)
            qubit.xy.play("EF_x180")

            align(qubit.xy.name, sideband_drive.name)
            sideband_drive.update_frequency(target_if_j)
            self.play_sideband_flattop(flat_top_duration_clk=flat_top_clk_j)

            align(sideband_drive.name, qubit.xy.name)

    def fock_prep_qua_sfp(
        self,
        fock_level: int,
        qubit,
        ff_repeat: int = 1,
        sfp_max_retries: int = 20,
    ) -> None:
        """Generate QUA statements to prepare Fock |fock_level⟩ with measurement feedforward.

        After each sideband step j, calls qubit.readout_state_gef() and corrects:
          state 0 (|g⟩): success → increment ff_counter
          state 1 (|e⟩): apply EF_x180 → |f⟩, retry sideband
          state 2 (|f⟩): retry sideband directly
        Continues until ff_repeat consecutive ground-state outcomes OR sfp_max_retries
        total attempts, whichever comes first.  The hard cutoff prevents the loop
        from blocking indefinitely when readout or sideband calibration is imperfect.

        Args:
            fock_level:       Target Fock level to prepare.
            qubit:            QuAM transmon object.
            ff_repeat:        Consecutive |g⟩ measurements required per step (default 1).
            sfp_max_retries:  Maximum total feedforward attempts per sideband step
                              before giving up and moving on (default 20).

        Requires qubit.resonator.gef_centers and GEF_frequency_shift to be calibrated.
        """
        sideband_drive = self.sideband_drive
        if not getattr(qubit.resonator, "gef_centers", None):
            _logger.warning(
                "qubit.resonator.gef_centers not found. "
                "Run the GEF IQ blobs calibration node (15) first."
            )

        for j in range(fock_level):
            sb_rf_j = self.get_transition_rf(j)
            if_offset_j = int(sb_rf_j - sideband_drive.RF_frequency)
            target_if_j = int(sideband_drive.intermediate_frequency) + if_offset_j

            tr_j = self.transitions.get(f"f{j}g{j+1}")
            flat_top_clk_j = (tr_j.pi_flat_top_length_ns // 4) if (tr_j and tr_j.pi_flat_top_length_ns) else None

            ge_if_j = self.ge_if_at_fock(qubit, j)
            ef_if_j = self.ef_if_at_fock(qubit, j)

            state_ff = declare(int)
            ff_counter = declare(int)
            ff_total = declare(int)

            qubit.xy.update_frequency(ge_if_j)
            qubit.xy.play("x180")
            qubit.xy.update_frequency(ef_if_j)
            qubit.xy.play("EF_x180")

            align(qubit.xy.name, sideband_drive.name)
            sideband_drive.update_frequency(target_if_j)
            self.play_sideband_flattop(flat_top_duration_clk=flat_top_clk_j)

            align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)

            assign(ff_counter, 0)
            assign(ff_total, 0)
            with while_((ff_counter < ff_repeat) & (ff_total < sfp_max_retries)):
                qubit.readout_state_gef(state_ff)
                align(qubit.resonator.name, qubit.xy.name, sideband_drive.name)
                assign(ff_total, ff_total + 1)

                with if_(state_ff == 0):
                    assign(ff_counter, ff_counter + 1)
                with else_():
                    assign(ff_counter, 0)
                    with if_(state_ff == 1):
                        qubit.xy.update_frequency(ef_if_j)
                        qubit.xy.play("EF_x180")
                    align(qubit.xy.name, sideband_drive.name)
                    sideband_drive.update_frequency(target_if_j)
                    self.play_sideband_flattop(flat_top_duration_clk=flat_top_clk_j)
                    align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)

            align(sideband_drive.name, qubit.xy.name)

    def parity_filter_qua(
        self,
        qubit,
        expected_parity: str,
        I_pf,
        pf_accepted,
        parity_clk: int = None,
    ) -> None:
        """Run a parity Ramsey and set pf_accepted based on expected_parity.

        Performs x90 → wait(parity_clk) → x90 → resonator measure.
        The parity wait time is resolved from pair.parity_time when parity_clk
        is not supplied, falling back to 1 / (4 · |chi|) when parity_time is
        also unset.

        Convention (derived from the Wigner reconstruction formula
        parity = P_e(+1) − P_e(−1) and W(0,0) > 0 for vacuum):
          even-photon state → qubit EXCITED after Ramsey (I_pf > ge threshold)
          odd-photon  state → qubit GROUND  after Ramsey (I_pf ≤ ge threshold)

        Sets pf_accepted = True when the measured parity matches expected_parity:
          'even' → I_pf >  ge threshold
          'odd'  → I_pf ≤ ge threshold

        After this call the qubit is in a projected state (not necessarily |g⟩).
        Caller must reset the qubit (e.g. conditional x180 on I_pf) before the
        next gate.

        Args:
            qubit:            QuAM transmon object (provides .xy and .resonator).
            expected_parity:  'even' or 'odd' — resolved at Python compile time.
            I_pf:             QUA fixed variable to receive the I-quadrature result.
            pf_accepted:      QUA bool variable set True on parity match.
            parity_clk:       Wait time in QUA clock cycles (4 ns each).  When None
                              (default), resolved from pair.parity_time or pair.chi.
        """
        if parity_clk is None:
            if self.parity_time is not None:
                parity_clk = int(round(self.parity_time * 1e9 / 4))
            elif self.chi is not None and self.chi != 0:
                parity_clk = int(round(1.0 / (4.0 * abs(self.chi)) * 1e9 / 4))
            else:
                raise ValueError(
                    "parity_filter_qua: cannot resolve parity_clk — "
                    "set pair.parity_time or pair.chi in QuAM."
                )

        ge_threshold = float(qubit.resonator.operations["readout"].threshold)
        Q_pf = declare(fixed)

        with strict_timing_():
            qubit.xy.play("x90")
            qubit.xy.wait(parity_clk)
            qubit.xy.play("x90")

        align(qubit.xy.name, qubit.resonator.name)
        qubit.resonator.measure("readout", qua_vars=(I_pf, Q_pf))
        qubit.resonator.wait(qubit.resonator.depletion_time // 4)
        align(qubit.resonator.name, qubit.xy.name)

        if expected_parity == "odd":
            # Odd-photon state → qubit ground after Ramsey
            with if_(I_pf <= ge_threshold):
                assign(pf_accepted, True)
            with else_():
                assign(pf_accepted, False)
        else:  # even
            # Even-photon state → qubit excited after Ramsey
            with if_(I_pf > ge_threshold):
                assign(pf_accepted, True)
            with else_():
                assign(pf_accepted, False)

