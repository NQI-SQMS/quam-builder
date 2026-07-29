from quam.core import quam_dataclass
from quam import QuamComponent
from typing import Union, ClassVar

from quam_builder.architecture.superconducting.components.xy_drive import XYDriveIQ, XYDriveMW

__all__ = ["TWPA"]


@quam_dataclass
class TWPA(QuamComponent):
    """
    Example QuAM component for a TWPA.

    Args:
        id (str, int): The id of the TWPA, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        pump (Union[XYDriveIQ, XYDriveMW]): The pump component(sticky element) used for continuous output.
        pump_ (Union[XYDriveIQ, XYDriveMW]): The pump component(non sticky element)used for TWPA calibration
        spectroscopy (Union[XYDriveIQ, XYDriveMW]): Probe tone used for calibrating the saturation power of the TWPA

        max_avg_gain (float): The maximum average gain around the readout resonators related to the TWPA
        max_avg_snr_improvement (float): The maximum average SNR improvement around the readout resonators related to the TWPA
        pump_frequency (float): calibrated pump frequency at which twpa gives the maximum average snr improvement
        pump_amplitude (float): calibrated absolute pump power [dBm] at which twpa gives the maximum average
            snr improvement. Baked directly into the "pump"/"pump_" pulses' own amplitude (see
            01_twpa_pump_power_sweep's update_state) rather than used as a play-time amplitude_scale,
            since amplitude_scale is only valid within ~+-6 dB of the wired reference power.
        mltpx_pump_frequency (float): calibrated pump frequency at which twpa gives proper snr improvement for multiplexed readout
        mltpx_pump_amplitude (float): calibrated absolute pump power [dBm] at which twpa gives proper snr improvement for multiplexed readout
        pump_ring_up_time_ns (float): time [ns] to let the pump settle after being (re)started, before it is
            relied on for a measurement. The sticky `pump` element has been observed to auto-decay after a
            hardware-imposed maximum hold window (~265 us) even while being held continuously, so this wait
            is internally broken into periodic re-triggers of `pump` shorter than that window (see
            `_STICKY_REFRESH_PERIOD_NS`) rather than a single uninterrupted `wait`.
        p_saturation (float): calibrated saturation power of the twpa
        avg_std_gain (float): standard deviation of the average gain around the readout resonators related to the TWPA
        avg_std_snr_improvement (float): standard deviation of the average snr improvement around the readout resonators related to the TWPA

        dispersive_feature (float): dispersive feature of the twpa defined from it's designed parameters
        qubits (list): list of qubits of which the signals are amplified by the twpa

        initialization (bool): whether to use the twpa in the QUA program or not
        _initialized_ids (ClassVar[set]): A class-level set to track initialized twpa object IDs externally.
            This won't be serialized since it's not an instance attribute.

    """

    id: Union[int, str]

    pump: Union[XYDriveIQ, XYDriveMW] = None
    pump_: Union[XYDriveIQ, XYDriveMW] = None
    spectroscopy: Union[XYDriveIQ, XYDriveMW] = None

    max_avg_gain: float = None
    max_avg_snr_improvement: float = None
    pump_frequency: float = None
    pump_amplitude: float = None
    mltpx_pump_frequency: float = None
    mltpx_pump_amplitude: float = None
    pump_ring_up_time_ns: float = None
    p_saturation: float = None
    avg_std_gain: float = None
    avg_std_snr_improvement: float = None

    dispersive_feature: float = None
    qubits: list = None

    initialization: bool = True
    _initialized_ids: ClassVar[set] = set()

    @property
    def name(self):
        """The name of the twpa"""
        return self.id if isinstance(self.id, str) else f"twpa{self.id}"

    def initialize(self):
        # dont use twpa for the QUA program if initialization is set to False
        if not self.initialization:
            return
        # Check initialization state using object ID (memory address)
        # Initialize TWPA pump only when it hasn't been initialized yet
        # This won't be serialized since it's stored in a class-level set
        obj_id = id(self)
        if obj_id in self._initialized_ids:
            return
        if self.pump_frequency is not None:
            self.pump.update_frequency(int(self.pump_frequency - self.pump.LO_frequency))
        if self.pump_amplitude is not None:
            self.pump.play("pump", amplitude_scale=self.pump_amplitude)
        else:
            self.pump.play("pump")

        self._initialized_ids.add(obj_id)
