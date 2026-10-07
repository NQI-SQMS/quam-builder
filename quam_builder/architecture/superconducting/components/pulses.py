# Copyright 2026 Fermi Forward Discovery Group, LLC.
# Authors: Leonardo Bove, Taeyoon Kim, Joey Yaker.
# Licensed under the terms in ../cavity/LICENSE (BSD-3-Clause-style, DOE/SQMS-funded work).

from typing import Optional

import numpy as np
from quam.core import quam_dataclass
from quam.components.pulses import Pulse, ReadoutPulse


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


@quam_dataclass
class ClearReadoutPulse(ReadoutPulse):
    """CLEAR-style readout+reset pulse: ring-up, steady-state readout, ring-down.

    Five piecewise-constant amplitude segments, following McClure et al.,
    PRApplied 5, 011001(R) (2016) (see also Huang et al., PRX 16, 011058
    (2026), App. H): two ring-up segments that quickly bring the resonator
    field to its steady-state readout amplitude, a steady-state segment used
    for readout, and two ring-down/reset segments that actively drive
    residual photons back to vacuum faster than passive decay — shortening
    the effective depletion time and reducing residual-photon dephasing of
    any measurement that follows.

    `length` is derived automatically as the sum of the five segment
    lengths — it is not settable directly.

    Args:
        readout_amplitude: Steady-state (readout) segment amplitude in Volts.
        ring_up1_amplitude: First ring-up segment amplitude in Volts.
        ring_up1_length: First ring-up segment length in samples (ns).
        ring_up2_amplitude: Second ring-up segment amplitude in Volts.
        ring_up2_length: Second ring-up segment length in samples (ns).
        readout_length: Steady-state (readout) segment length in samples (ns).
        ring_down1_amplitude: First ring-down/reset segment amplitude in Volts.
        ring_down1_length: First ring-down/reset segment length in samples (ns).
        ring_down2_amplitude: Second ring-down/reset segment amplitude in Volts.
        ring_down2_length: Second ring-down/reset segment length in samples (ns).
    """

    readout_amplitude: float
    ring_up1_amplitude: float
    ring_up1_length: int
    ring_up2_amplitude: float
    ring_up2_length: int
    readout_length: int
    ring_down1_amplitude: float
    ring_down1_length: int
    ring_down2_amplitude: float
    ring_down2_length: int

    # Length is derived from the five segment lengths, but still needs to be
    # declared to satisfy the dataclass, but we'll override its behavior
    length: Optional[int] = None  # pyright: ignore

    @property
    def length(self):  # noqa: 811
        return (
            self.ring_up1_length
            + self.ring_up2_length
            + self.readout_length
            + self.ring_down1_length
            + self.ring_down2_length
        )

    @length.setter
    def length(self, length: Optional[int]):
        if length is not None and not isinstance(length, property):
            raise AttributeError(f"length is not writable with value {length}")

    def waveform_function(self):
        segments = [
            (self.ring_up1_amplitude, self.ring_up1_length),
            (self.ring_up2_amplitude, self.ring_up2_length),
            (self.readout_amplitude, self.readout_length),
            (self.ring_down1_amplitude, self.ring_down1_length),
            (self.ring_down2_amplitude, self.ring_down2_length),
        ]
        return np.concatenate([amp * np.ones(length) for amp, length in segments])
