from typing import Callable, Dict, Any, Union, Optional, Literal
from dataclasses import field
from logging import getLogger

from quam.core import quam_dataclass
from quam.components.quantum_components import Qubit
from quam_builder.architecture.superconducting.components.xy_drive import (
    XYDriveIQ,
    XYDriveMW,
)
from quam_builder.architecture.superconducting.cavity.cavity_operations import (
    SNAPGate,
    _play_displacement,
)

from qm.qua import (
    align,
    wait,
    update_frequency,
)


__all__ = ["CavityMode"]


@quam_dataclass
class CavityMode(Qubit):
    """
    Example QUAM component for a transmon qubit.

    Attributes:
        id (Union[int, str]): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add `Channel._default_label`.
        xy (Union[MWChannel, IQChannel]): The xy drive component.
        resonator (Union[ReadoutResonatorIQ, ReadoutResonatorMW]): The readout resonator component.
        f_01 (float): The 0-1 transition frequency in Hz. Default is None.
        f_12 (float): The 1-2 transition frequency in Hz. Default is None.
        anharmonicity (int): The transmon anharmonicity in Hz. Default is None.
        T1 (float): The transmon T1 in seconds. Default is None.
        T2ramsey (float): The transmon T2* in seconds.
        T2echo (float): The transmon T2 in seconds.
        thermalization_time_factor (int): Thermalization time in units of T1. Default is 5.
        sigma_time_factor (int): Sigma time factor for pulse shaping. Default is 5.
        GEF_frequency_shift (int): The frequency shift for the GEF states. Default is None.
        grid_location (str): Qubit location in the plot grid as "column, row".
        gate_fidelity (Dict[str, Any]): Collection of single qubit gate fidelity.
        extras (Dict[str, Any]): Additional attributes for the transmon.

    Methods:
        name: Returns the name of the transmon.
        inferred_f_12: Returns the 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity.
        inferred_anharmonicity: Returns the transmon anharmonicity in Hz, derived from f_01 and f_12.
        sigma: Returns the sigma value for a given pulse.
        thermalization_time: Returns the transmon thermalization time in ns.
        set_gate_shape: Sets the shape of the single qubit gates.
        reset: Reset the cavity mode with a chosen method ("thermal" or "active_sideband").
        reset_cavity_thermal: Wait thermalization_time_factor * T1 for the cavity to decay to vacuum.
        reset_cavity_active_sideband: Actively cool to vacuum via repeated f0g1 π-pulses (|n,g⟩→|n-1,f⟩).
    """

    id: Union[int, str]

    cavity_mode_drive: Union[XYDriveIQ, XYDriveMW] = None
    snap_gate: Optional[SNAPGate] = None

    T1: float = None
    T2ramsey: float = None
    T2echo: float = None
    thermalization_time_factor: int = 5

    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def inferred_f_12(self) -> float:
        """The 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity"""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.f_01=} is not a number"
            )
        if not isinstance(self.anharmonicity, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.anharmonicity=} is not a number"
            )
        return self.f_01 + self.anharmonicity

    @property
    def inferred_anharmonicity(self) -> float:
        """The transmon anharmonicity in Hz, derived from f_01 and f_12."""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_01=} is not a number"
            )
        if not isinstance(self.f_12, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_12=} is not a number"
            )
        return self.f_12 - self.f_01

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns."""
        if self.T1 is not None:
            return int(self.thermalization_time_factor * self.T1 * 1e9 / 4) * 4
        else:
            return int(self.thermalization_time_factor * 10e-6 * 1e9 / 4) * 4

    def displacement(
        self,
        amplitude=None,
        alpha_re=None,
        alpha_im=None,
        length: Optional[int] = None,
    ) -> None:
        """Play a displacement pulse on the cavity drive.

        Args:
            amplitude: Real amplitude scale factor (Python ``float`` or QUA ``fixed``).
                Used for simple displacements; ignored when *alpha_re* or *alpha_im*
                is given.
            alpha_re: In-phase (I) component for complex displacement (Python float or
                QUA fixed variable).
            alpha_im: Quadrature (Q) component paired with *alpha_re*.
            length: Optional pulse duration override in QUA clock cycles (4 ns each).

        Examples::

            # Simple real displacement (D-SNAP step):
            cavity_mode.displacement(amplitude=amp_scale)

            # Complex displacement (Wigner probe, a_re/a_im are QUA variables):
            cavity_mode.displacement(alpha_re=a_re, alpha_im=a_im)
        """
        _play_displacement(self.cavity_mode_drive, amplitude, alpha_re, alpha_im, length)

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            if f"{gate}_{gate_shape}" in self.xy.operations:
                self.xy.operations[gate] = f"#./{gate}_{gate_shape}"
            else:
                raise AttributeError(
                    f"The gate '{gate}_{gate_shape}' is not part of the existing operations for {self.xy.name} --> {self.xy.operations.keys()}."
                )

    def reset(
        self,
        reset_type: Literal["thermal", "active_sideband", "active_sideband_v2"] = "thermal",
        simulate: bool = False,
        log_callable: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Reset the cavity mode with the specified method.

        Args:
            reset_type: ``"thermal"`` waits ``thermalization_time_factor * T1`` for the
                cavity photons to decay naturally.  ``"active_sideband"`` performs active
                sideband cooling via repeated sideband π-pulses (see
                :meth:`reset_cavity_active_sideband` for the required kwargs).
            simulate: When ``True`` the reset is skipped so the simulation runs faster.
            log_callable: Called with a warning string when the reset is skipped in
                simulation mode.  Defaults to the module logger.
            **kwargs: Forwarded verbatim to :meth:`reset_cavity_active_sideband` when
                ``reset_type="active_sideband"``.
        """
        if not simulate:
            if reset_type == "thermal":
                self.reset_cavity_thermal()
            elif reset_type == "active_sideband":
                self.reset_cavity_active_sideband(**kwargs)
            elif reset_type == "active_sideband_v2":
                self.reset_cavity_active_sideband_v2(**kwargs)
        else:
            if log_callable is None:
                log_callable = getLogger(__name__).warning
            log_callable(
                "For simulating the QUA program, the cavity mode reset has been skipped."
            )

    def reset_cavity_thermal(self):
        """Wait ``thermalization_time_factor * T1`` for the cavity to decay to vacuum."""
        self.wait(self.thermalization_time // 4)

    def reset_cavity_active_sideband(
        self,
        sideband_drive,
        qubit_thermalization_time: int,
        f0g1_pi_pulse_name: str = "sideband_square",
        fock_n: int = 1,
        sideband_pulse_duration_ns: int = None,
        chi_hz: float = None,
        pair=None,
    ):
        """
        Actively cool the cavity mode to vacuum using sideband π-pulses.

        The protocol removes photons one at a time, starting from Fock state |fock_n⟩
        down to vacuum:

          For n = fock_n, fock_n-1, …, 1:
            1. Update the sideband drive IF to target the photon-number-resolved
               |n, g⟩ ↔ |n-1, f⟩ transition (i.e. the f{n-1}g{n} transition).
               When ``pair`` is supplied, the calibrated RF frequency from
               ``pair.transitions["f{n-1}g{n}"].RF_frequency`` is used.
               Otherwise falls back to ``IF_base - (n-1) × |chi_hz|``.
            2. Play the sideband π-pulse  →  maps |n, g⟩ → |n-1, f⟩.
               When ``pair`` is supplied and the sideband drive has
               ``"sideband_ramp_up"`` / ``"sideband_ramp_down"`` operations,
               the flat-top structure is used: ramp_up + flat_top + ramp_down.
               If ``sideband_pulse_duration_ns`` is set it overrides the flat-top
               duration (use a long value to ensure photon decoherence).
            3. Wait ``2 × qubit_thermalization_time`` for the transmon to relax
               |f⟩ → |e⟩ → |g⟩, taking the cavity from |n-1, f⟩ to |n-1, g⟩.

        After the loop the sideband drive IF is restored to its original value.

        Args:
            sideband_drive: The sideband drive channel — typically
                ``pair.sideband_drive``.
            qubit_thermalization_time: Time (ns) for a single qubit decay step.
                Pass ``qubit.thermalization_time``.  Waits
                ``2 × qubit_thermalization_time`` per step.
            f0g1_pi_pulse_name: Fallback pulse name (flat-top middle segment) when
                per-level operations are absent.  Default is ``"sideband_square"``.
            fock_n: Starting photon number.  Default is 1.
            sideband_pulse_duration_ns: Override the flat-top duration [ns] for every
                cooling step.  When ``None``, the calibrated ``sideband_cooling_time``
                (t95 from node 35) is used if available, falling back to
                ``pi_flat_top_length_ns``, then the pulse's own length.
                Must be a multiple of 4 ns.
            chi_hz: Per-photon qubit frequency shift [Hz] (``pair.chi``).
                Used only as fallback when ``pair`` is not supplied or calibrated
                frequencies are unavailable.  Stored negative for typical
                transmon-cavity systems.
            pair: ``CavityTransmonPair`` instance.  When provided, calibrated
                sideband RF frequencies (``pair.transitions["f{k}g{k+1}"].RF_frequency``)
                and π-pulse lengths (``pair.transitions[...].pi_flat_top_length_ns``)
                are used, and the flat-top pulse structure is applied automatically
                when the ramp operations are present.
        """
        base_if = int(sideband_drive.intermediate_frequency)
        use_ramps = (
            "sideband_ramp_up" in sideband_drive.operations
            and "sideband_ramp_down" in sideband_drive.operations
        )

        # Fallback chi step (positive) used when pair.transitions are unavailable
        chi_step = int(-chi_hz) if chi_hz is not None else 0

        for n in range(fock_n, 0, -1):
            # f{k}g{k+1} with k = n-1 is the transition that removes the nth photon:
            # |g, n⟩ ↔ |f, n-1⟩
            k = n - 1
            tr_key = f"f{k}g{k+1}"
            op_name = tr_key + "_pi"

            # -- Resolve target IF ---------------------------------------------
            target_if = base_if
            if pair is not None:
                tr = pair.transitions.get(tr_key)
                if tr is not None and tr.RF_frequency is not None:
                    target_if = base_if + int(float(tr.RF_frequency) - float(sideband_drive.RF_frequency))
            if target_if == base_if and chi_step != 0:
                # chi-linear approximation: sideband IF shifts by chi_step per photon
                target_if = base_if - k * chi_step

            if target_if != base_if:
                update_frequency(sideband_drive.name, target_if)

            # -- Resolve pulse operation name -----------------------------------
            play_name = op_name if op_name in sideband_drive.operations else f0g1_pi_pulse_name

            # -- Resolve flat-top duration --------------------------------------
            flat_top_clk = None
            if sideband_pulse_duration_ns is not None:
                flat_top_clk = sideband_pulse_duration_ns // 4
            elif pair is not None:
                tr = pair.transitions.get(tr_key)
                if tr is not None and tr.sideband_cooling_time is not None:
                    flat_top_clk = tr.sideband_cooling_time // 4
                elif tr is not None and tr.pi_flat_top_length_ns is not None:
                    flat_top_clk = tr.pi_flat_top_length_ns // 4

            # -- Play sideband pulse --------------------------------------------
            if pair is not None and use_ramps:
                pair.play_sideband_flattop(flat_top_duration_clk=flat_top_clk)
            elif use_ramps:
                sideband_drive.play("sideband_ramp_up")
                if flat_top_clk is not None:
                    sideband_drive.play(play_name, duration=flat_top_clk)
                else:
                    sideband_drive.play(play_name)
                sideband_drive.play("sideband_ramp_down")
            else:
                if flat_top_clk is not None:
                    sideband_drive.play(play_name, duration=flat_top_clk)
                else:
                    sideband_drive.play(play_name)

            # Wait for the transmon to relax |f⟩ → |e⟩ → |g⟩ (two decay steps)
            wait(2 * qubit_thermalization_time // 4, sideband_drive.name)

        # Restore the original IF
        update_frequency(sideband_drive.name, base_if)

    def reset_cavity_active_sideband_v2(
        self,
        sideband_drive,
        qubit,
        qubit_thermalization_time: int,
        f0g1_pi_pulse_name: str = "sideband_square",
        fock_n: int = 1,
        sideband_pulse_duration_ns: int = None,
        chi_hz: float = None,
        pair=None,
        n_repeats: int = 3,
    ):
        """
        Actively cool the cavity mode to vacuum using a long sideband pulse followed by
        repeated (GEF active reset → calibrated f0g1 π → GEF active reset) cycles.

        Protocol per Fock level n = fock_n, …, 1  (transition f{k}g{k+1}, k = n-1):
          1. Update sideband drive IF to the photon-number-resolved transition.
          2. Play a long sideband pulse (uses sideband_cooling_time / t95 from node 35 when
             available, otherwise sideband_pulse_duration_ns or the pulse's own length).
          3. Repeat n_repeats times:
               align all elements
               qubit.reset_qubit_active_gef()   # ensures |g⟩ before π
               align all elements
               play calibrated f0g1 π-pulse      # |g,n⟩ → |f,n-1⟩
               align all elements
               qubit.reset_qubit_active_gef()   # relaxes |f⟩ → |g⟩ actively
               align all elements
        After the loop the sideband drive IF is restored.

        Args:
            sideband_drive: The sideband drive channel (typically pair.sideband_drive).
            qubit: The transmon Qubit object; must expose reset_qubit_active_gef().
            qubit_thermalization_time: Qubit T1-based wait [ns] — kept for API symmetry
                with reset_cavity_active_sideband; not used for passive waits here.
            f0g1_pi_pulse_name: Fallback flat-top pulse name when per-level ops are absent.
            fock_n: Starting photon number. Default is 1.
            sideband_pulse_duration_ns: Override for the long sideband pulse flat-top [ns].
                When None the priority is: sideband_cooling_time → pi_flat_top_length_ns →
                pulse own length. Must be a multiple of 4 ns.
            chi_hz: Per-photon qubit frequency shift [Hz]. Used as fallback when pair
                transitions lack calibrated RF frequencies.
            pair: CavityTransmonPair instance. When provided, calibrated RF frequencies and
                π-pulse lengths from pair.transitions are used.
            n_repeats: Number of (GEF reset → π → GEF reset) cycles per Fock level.
        """
        base_if = int(sideband_drive.intermediate_frequency)
        use_ramps = (
            "sideband_ramp_up" in sideband_drive.operations
            and "sideband_ramp_down" in sideband_drive.operations
        )

        chi_step = int(-chi_hz) if chi_hz is not None else 0

        # Build the list of element names needed for alignment
        qubit_el_names = [qubit.xy.name, qubit.resonator.name]
        all_el_names = qubit_el_names + [sideband_drive.name]

        for n in range(fock_n, 0, -1):
            k = n - 1
            tr_key = f"f{k}g{k+1}"
            op_name = tr_key + "_pi"

            # -- Resolve target IF ------------------------------------------------
            target_if = base_if
            if pair is not None:
                tr = pair.transitions.get(tr_key)
                if tr is not None and tr.RF_frequency is not None:
                    target_if = base_if + int(
                        float(tr.RF_frequency) - float(sideband_drive.RF_frequency)
                    )
            if target_if == base_if and chi_step != 0:
                target_if = base_if - k * chi_step

            if target_if != base_if:
                update_frequency(sideband_drive.name, target_if)

            # -- Resolve operation name -------------------------------------------
            play_name = (
                op_name if op_name in sideband_drive.operations else f0g1_pi_pulse_name
            )

            # -- Resolve long-pulse flat-top (clock cycles) -----------------------
            long_flat_top_clk = None
            if sideband_pulse_duration_ns is not None:
                long_flat_top_clk = sideband_pulse_duration_ns // 4
            elif pair is not None:
                tr = pair.transitions.get(tr_key)
                if tr is not None and tr.sideband_cooling_time is not None:
                    long_flat_top_clk = tr.sideband_cooling_time // 4
                elif tr is not None and tr.pi_flat_top_length_ns is not None:
                    long_flat_top_clk = tr.pi_flat_top_length_ns // 4

            # -- Resolve precise π flat-top (clock cycles) ------------------------
            pi_flat_top_clk = None
            if pair is not None:
                tr = pair.transitions.get(tr_key)
                if tr is not None and tr.pi_flat_top_length_ns is not None:
                    pi_flat_top_clk = tr.pi_flat_top_length_ns // 4

            # -- Play long sideband pulse -----------------------------------------
            if pair is not None and use_ramps:
                pair.play_sideband_flattop(flat_top_duration_clk=long_flat_top_clk)
            elif use_ramps:
                sideband_drive.play("sideband_ramp_up")
                if long_flat_top_clk is not None:
                    sideband_drive.play(play_name, duration=long_flat_top_clk)
                else:
                    sideband_drive.play(play_name)
                sideband_drive.play("sideband_ramp_down")
            else:
                if long_flat_top_clk is not None:
                    sideband_drive.play(play_name, duration=long_flat_top_clk)
                else:
                    sideband_drive.play(play_name)

            # -- n_repeats × (GEF reset → π → GEF reset) -------------------------
            for _ in range(n_repeats):
                align(*all_el_names)
                qubit.reset_qubit_active_gef()
                align(*all_el_names)
                if pair is not None and use_ramps:
                    pair.play_sideband_flattop(flat_top_duration_clk=pi_flat_top_clk)
                elif use_ramps:
                    sideband_drive.play("sideband_ramp_up")
                    if pi_flat_top_clk is not None:
                        sideband_drive.play(play_name, duration=pi_flat_top_clk)
                    else:
                        sideband_drive.play(play_name)
                    sideband_drive.play("sideband_ramp_down")
                else:
                    if pi_flat_top_clk is not None:
                        sideband_drive.play(play_name, duration=pi_flat_top_clk)
                    else:
                        sideband_drive.play(play_name)
                align(*all_el_names)
                qubit.reset_qubit_active_gef()
                align(*all_el_names)

        # Restore original IF
        update_frequency(sideband_drive.name, base_if)

    def wait(self, duration: int):
        """Wait for a given duration on all channels of the qubit.

        Args:
            duration (int): The duration to wait for in unit of clock cycles (4ns).
        """
        channel_names = [channel.name for channel in self.channels.values()]
        wait(duration, *channel_names)
