"""Shared signal data structures and time-axis helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_1d_float_array(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def infer_sample_rate_from_time(time: np.ndarray | list[float], tolerance: float = 1e-3) -> float:
    """Infer sample rate from a nearly uniform time array.

    Args:
        time: Monotonic time values in seconds.
        tolerance: Relative tolerance allowed for interval variation.

    Returns:
        Sampling rate in Hz.

    Raises:
        ValueError: If the time axis is too short, non-monotonic, or non-uniform.
    """
    time_array = _as_1d_float_array(time, "time")
    if time_array.size < 2:
        raise ValueError("time must contain at least two samples.")

    deltas = np.diff(time_array)
    if np.any(deltas <= 0):
        raise ValueError("time must be strictly increasing.")

    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        raise ValueError("time interval must be positive.")

    max_relative_error = float(np.max(np.abs(deltas - median_delta)) / median_delta)
    if max_relative_error > tolerance:
        raise ValueError(
            "time intervals are not uniform enough to infer sample rate "
            f"(relative error {max_relative_error:.3g})."
        )
    return 1.0 / median_delta


@dataclass
class SignalData:
    """Single-channel signal with a time axis and sampling metadata."""

    name: str
    time: np.ndarray
    values: np.ndarray
    sample_rate: float
    channel_names: list[str]
    unit_time: str = "s"
    unit_amplitude: str = "V"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _as_1d_float_array(self.time, "time")
        self.values = _as_1d_float_array(self.values, "values")
        if self.time.size != self.values.size:
            raise ValueError("time and values must have the same length.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not self.channel_names:
            self.channel_names = [self.name]


@dataclass
class MultiChannelSignal:
    """Multi-channel signal sharing one time axis."""

    name: str
    time: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _as_1d_float_array(self.time, "time")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not self.channels:
            raise ValueError("channels must not be empty.")

        cleaned: dict[str, np.ndarray] = {}
        for name, values in self.channels.items():
            channel = _as_1d_float_array(values, f"channel {name}")
            if channel.size != self.time.size:
                raise ValueError(f"channel {name} length does not match time.")
            cleaned[str(name)] = channel
        self.channels = cleaned

    @property
    def channel_names(self) -> list[str]:
        """Return channel names in insertion order."""
        return list(self.channels.keys())

    def select_channel(self, channel_name: str) -> SignalData:
        """Return one channel as a SignalData instance."""
        if channel_name not in self.channels:
            raise KeyError(f"Unknown channel: {channel_name}")
        return SignalData(
            name=channel_name,
            time=self.time.copy(),
            values=self.channels[channel_name].copy(),
            sample_rate=self.sample_rate,
            channel_names=[channel_name],
            metadata={**self.metadata, "source": self.name},
        )

