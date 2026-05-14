from typing import Callable, Dict, Any, Union, Optional, Literal, Tuple
from dataclasses import field
from logging import getLogger

from quam.core import quam_dataclass
from quam.components.quantum_components import Qubit
from quam_builder.architecture.superconducting.components.xy_drive import (
    XYDriveIQ,
    XYDriveMW,
)

from qm import QuantumMachine, logger
from qm.qua.type_hints import QuaVariable
from qm.octave.octave_mixer_calibration import MixerCalibrationResults
from qm.qua import (
    declare,
    fixed,
    assign,
    wait,
    update_frequency,
    Math,
    Cast,
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
        calibrate_octave: Calibrates the Octave channels (xy and resonator) linked to this transmon.
        set_gate_shape: Sets the shape of the single qubit gates.
        readout_state: Performs a readout of the qubit state using the specified pulse.
        reset: Reset the cavity mode with a chosen method ("thermal" or "active_sideband").
        reset_cavity_thermal: Wait thermalization_time_factor * T1 for the cavity to decay to vacuum.
        reset_cavity_active_sideband: Actively cool to vacuum via repeated f0g1 π-pulses (|n,g⟩→|n-1,f⟩).
        readout_state_gef: Perform a GEF state readout using the specified pulse and update the state variable.
    """

    id: Union[int, str]

    cavity_mode_drive: Union[XYDriveIQ, XYDriveMW] = None

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

    def calibrate_octave(
        self,
        QM: QuantumMachine,
        calibrate_drive: bool = True,
        calibrate_resonator: bool = True,
    ) -> Tuple[
        Union[None, MixerCalibrationResults], Union[None, MixerCalibrationResults]
    ]:
        """Calibrate the Octave channels (xy and resonator) linked to this transmon for the LO frequency, intermediate
        frequency and Octave gain as defined in the state.

        Args:
            QM (QuantumMachine): the running quantum machine.
            calibrate_drive (bool): flag to calibrate xy line.
            calibrate_resonator (bool): flag to calibrate the resonator line.

        Return:
            The Octave calibration results as (resonator, xy_drive)
        """
        if calibrate_resonator and self.resonator is not None:
            if hasattr(self.resonator, "frequency_converter_up"):
                logger.info(f"Calibrating {self.resonator.name}")
                resonator_calibration_output = QM.calibrate_element(
                    self.resonator.name,
                    {
                        self.resonator.frequency_converter_up.LO_frequency: (
                            self.resonator.intermediate_frequency,
                        )
                    },
                )
            else:
                raise RuntimeError(
                    f"{self.resonator.name} doesn't have a 'frequency_converter_up' attribute, it is thus most likely "
                    "not connected to an Octave."
                )
        else:
            resonator_calibration_output = None

        if calibrate_drive and self.xy is not None:
            if hasattr(self.xy, "frequency_converter_up"):
                logger.info(f"Calibrating {self.xy.name}")
                xy_drive_calibration_output = QM.calibrate_element(
                    self.xy.name,
                    {
                        self.xy.frequency_converter_up.LO_frequency: (
                            self.xy.intermediate_frequency,
                        )
                    },
                )
            else:
                raise RuntimeError(
                    f"{self.xy.name} doesn't have a 'frequency_converter_up' attribute, it is thus most likely not "
                    "connected to an Octave."
                )
        else:
            xy_drive_calibration_output = None
        return resonator_calibration_output, xy_drive_calibration_output

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            if f"{gate}_{gate_shape}" in self.xy.operations:
                self.xy.operations[gate] = f"#./{gate}_{gate_shape}"
            else:
                raise AttributeError(
                    f"The gate '{gate}_{gate_shape}' is not part of the existing operations for {self.xy.name} --> {self.xy.operations.keys()}."
                )

    def readout_state(
        self, state, pulse_name: str = "readout", threshold: Optional[float] = None
    ):
        """
        Perform a readout of the qubit state using the specified pulse.

        This function measures the qubit state using the specified readout pulse and assigns the result to the given state variable.
        If no threshold is provided, the default threshold for the specified pulse is used.

        Args:
            state: The variable to assign the readout result to.
            pulse_name (str): The name of the readout pulse to use. Default is "readout".
            threshold (float, optional): The threshold value for the readout. If None, the default threshold for the pulse is used.

        Returns:
            None

        The function declares fixed variables I and Q, measures the qubit state using the specified pulse, and assigns the result to the state variable based on the threshold.
        It then waits for the resonator depletion time.
        """
        I = declare(fixed)
        Q = declare(fixed)
        if threshold is None:
            threshold = self.resonator.operations[pulse_name].threshold
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        assign(state, Cast.to_int(I > threshold))
        wait(self.resonator.depletion_time // 4, self.resonator.name)

    def reset(
        self,
        reset_type: Literal["thermal", "active_sideband"] = "thermal",
        simulate: bool = False,
        log_callable: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Reset the cavity mode with the specified method.

        Args:
            reset_type: ``"thermal"`` waits ``thermalization_time_factor * T1`` for the
                cavity photons to decay naturally.  ``"active_sideband"`` performs active
                sideband cooling via repeated f0g1 π-pulses (see
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
        f0g1_pi_pulse_name: str = "f0g1_pi",
        fock_n: int = 1,
        f0g1_pulse_duration_ns: int = None,
        chi_hz: float = None,
    ):
        """
        Actively cool the cavity mode to vacuum using f0g1 sideband π-pulses.

        The protocol removes photons one at a time, starting from Fock state |fock_n⟩
        down to vacuum:

          For n = fock_n, fock_n-1, …, 1:
            1. Update the sideband drive IF to target the photon-number-resolved
               |n, g⟩ ↔ |n-1, f⟩ transition:
               ``IF = sideband_IF + (n-1) × χ``
            2. Play the f0g1 π-pulse  →  maps |n, g⟩ → |n-1, f⟩.
               If ``f0g1_pulse_duration_ns`` is set, the pulse is played at that
               duration instead of the calibrated length.  Use a longer duration
               (e.g. several T1_cavity) to ensure complete photon decoherence
               during the pulse to ensure complete photon decoherence.
            3. Wait ``2 × qubit_thermalization_time`` for the transmon to relax
               |f⟩ → |e⟩ → |g⟩ (two decay steps), taking the cavity from
               |n-1, f⟩ to |n-1, g⟩.

        After the loop the sideband drive IF is restored to its original value.

        Args:
            sideband_drive: The :class:`XYDriveIQ` channel used for the f0g1
                sideband transition — typically ``pair.sideband_drive`` where
                ``pair`` is the :class:`CavityTransmonPair` for this cavity mode.
            qubit_thermalization_time: Time (ns) to wait for a single qubit decay
                step.  Pass ``qubit.thermalization_time``.  The method waits
                ``2 × qubit_thermalization_time`` per cooling step to account for
                both |f⟩ → |e⟩ and |e⟩ → |g⟩ relaxation.
            f0g1_pi_pulse_name: Name of the sideband π-pulse operation on
                ``sideband_drive``.  Default is ``"f0g1_pi"``.
            fock_n: Starting photon number — cooling sweeps from |fock_n⟩ to |0⟩.
                Default is 1 (single-photon removal).
            f0g1_pulse_duration_ns: Override the sideband pulse duration [ns].
                When ``None`` (default), the calibrated pulse length is used.
                Set to a value longer than the calibrated π-pulse (e.g. several
                ms) to allow the cavity photon to decohere during the drive,
                ensuring the cooling step completes even for imperfect π-pulses.
                Must be a multiple of 4 ns.
            chi_hz: Full per-photon qubit frequency shift [Hz] from the
                corresponding CavityTransmonPair — ``pair.chi``.
                Stored as a **negative** value: for typical transmon-cavity
                systems more photons lower the qubit frequency, so
                ``pair.chi < 0`` and ``|pair.chi|`` equals the PNRS peak
                spacing.  Used to resolve photon-number-dependent sideband
                frequencies when fock_n > 1.  The f0g1 sideband frequency
                *decreases* with photon number — the method subtracts
                ``(n-1) × |chi_hz|`` from the base IF accordingly.
                When ``None`` (default), frequency updates are skipped —
                correct for ``fock_n=1`` but inaccurate for higher Fock states.
        """
        base_if = int(sideband_drive.intermediate_frequency)
        # pair.chi is stored negative (full per-photon qubit frequency shift).
        # |chi_hz| = PNRS peak spacing.  The f0g1 sideband frequency decreases
        # by |chi_hz| per additional photon: target = base - (n-1)*|chi_hz|.
        # Negate chi_hz to get the positive step size used in the subtraction.
        chi = int(-chi_hz) if chi_hz is not None else 0

        # Convert override duration to QUA clock cycles (4 ns each)
        duration_clk = (f0g1_pulse_duration_ns // 4) if f0g1_pulse_duration_ns is not None else None

        for n in range(fock_n, 0, -1):
            # Target |n,g⟩ → |n-1,f⟩: sideband IF decreases by (n-1)*|chi_hz| relative to base.
            # chi = |chi_hz| (positive step) so subtracting it lowers the IF for higher Fock states.
            target_if = base_if - (n - 1) * chi
            if target_if != base_if:
                update_frequency(sideband_drive.name, target_if)
            if duration_clk is not None:
                sideband_drive.play(f0g1_pi_pulse_name, duration=duration_clk)
            else:
                sideband_drive.play(f0g1_pi_pulse_name)
            # Wait for the transmon to relax |f⟩ → |e⟩ → |g⟩ (two decay steps)
            wait(2 * qubit_thermalization_time // 4, sideband_drive.name)

        # Restore the original IF (may have been changed for fock_n > 1)
        if chi != 0 and fock_n > 1:
            update_frequency(sideband_drive.name, base_if)

    def readout_state_gef(self, state: QuaVariable, pulse_name: str = "readout"):
        """
        Perform a GEF state readout using the specified pulse and update the state variable.

        This function measures the 'I' and 'Q' quadrature components of the resonator's response
        to a given pulse, calculates the squared Euclidean distance between the measured
        (I, Q) values and the predefined GEF state centers, and assigns the state variable
        to the index of the closest GEF state.

        Args:
            state (QuaVariableBool): The variable to store the readout state (0 for 'g', 1 for 'e', 2 for 'f').
            pulse_name (str, optional): The name of the pulse to use for the readout. Defaults to "readout".

        Returns:
            None
        """
        I = declare(fixed)
        Q = declare(fixed)
        diff = declare(fixed, size=3)

        self.resonator.update_frequency(
            int(
                self.resonator.intermediate_frequency
                + self.resonator.GEF_frequency_shift
            )
        )
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        self.resonator.update_frequency(self.resonator.intermediate_frequency)

        gef_centers = [
            self.resonator.gef_centers[0],
            self.resonator.gef_centers[1],
            self.resonator.gef_centers[2],
        ]
        for p in range(3):
            assign(
                diff[p],
                Math.abs(I - gef_centers[p][0]) + Math.abs(Q - gef_centers[p][1]),
            )
        assign(state, Math.argmin(diff))
        wait(self.resonator.depletion_time // 4, self.resonator.name)

    def wait(self, duration: int):
        """Wait for a given duration on all channels of the qubit.

        Args:
            duration (int): The duration to wait for in unit of clock cycles (4ns).
        """
        channel_names = [channel.name for channel in self.channels.values()]
        wait(duration, *channel_names)
