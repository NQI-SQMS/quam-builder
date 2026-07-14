import numpy as np
from quam.core import quam_dataclass
from quam.components.pulses import Pulse


@quam_dataclass
class SineSqRampPulse(Pulse):
    """Sine-squared ramp pulse for sideband drives.

    Generates a smooth ramp from 0 → amplitude (direction="up") or
    amplitude → 0 (direction="down") using a sin² envelope. The endpoint
    reaches exactly amplitude (up) or 0 (down), so it connects seamlessly
    to a constant SquarePulse when played in strict_timing_.

    Args:
        amplitude: Peak amplitude in Volts.
        axis_angle: IQ axis angle in radians. None for a single channel or
            the I port of an IQ/MW channel. 0.0 drives along the X axis.
        direction: "up" for a rising ramp, "down" for a falling ramp.
    """

    amplitude: float
    axis_angle: float = None
    direction: str = "up"

    def waveform_function(self):
        n = self.length
        t = np.linspace(0, n, n)
        if self.direction == "up":
            waveform = self.amplitude * np.sin((np.pi / (2 * n)) * t) ** 2
        else:
            waveform = self.amplitude * np.sin((np.pi / (2 * n)) * (n - t)) ** 2
        if self.axis_angle is not None:
            waveform = waveform * np.exp(1j * self.axis_angle)
        return waveform
