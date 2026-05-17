"""Unit conversion and formatting utilities for signal analysis.

All internal computation uses SI units (Hz, s). These functions handle conversion
at the UI boundary only.
"""

from __future__ import annotations

FREQ_UNITS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
DEFAULT_FREQ_UNIT = "kHz"
DEFAULT_TIME_UNIT = "us"


def frequency_to_hz(value: float, unit: str) -> float:
    """Convert a frequency value in the given unit to Hz.

    >>> frequency_to_hz(500, "kHz")
    500000.0
    >>> frequency_to_hz(0.5, "MHz")
    500000.0
    >>> frequency_to_hz(500000, "Hz")
    500000.0
    """
    if unit not in FREQ_UNITS:
        raise ValueError(f"Unknown frequency unit: {unit}. Use Hz, kHz, or MHz.")
    return value * FREQ_UNITS[unit]


def hz_to_frequency(value_hz: float, unit: str) -> float:
    """Convert a frequency in Hz to the given display unit.

    >>> hz_to_frequency(500000, "kHz")
    500.0
    >>> hz_to_frequency(500000, "MHz")
    0.5
    """
    if unit not in FREQ_UNITS:
        raise ValueError(f"Unknown frequency unit: {unit}. Use Hz, kHz, or MHz.")
    return value_hz / FREQ_UNITS[unit]


def format_frequency(value_hz: float, unit: str = "kHz", precision: int = 3) -> str:
    """Format a frequency in Hz for display.

    >>> format_frequency(500000, "kHz")
    '500.000 kHz'
    >>> format_frequency(500000, "MHz")
    '0.500 MHz'
    """
    display_value = hz_to_frequency(value_hz, unit)
    return f"{display_value:.{precision}f} {unit}"


def auto_freq_unit(max_hz: float) -> str:
    """Pick the best frequency unit for a given maximum frequency.

    >>> auto_freq_unit(500)
    'Hz'
    >>> auto_freq_unit(5000)
    'kHz'
    >>> auto_freq_unit(5e6)
    'MHz'
    """
    if max_hz >= 1e6:
        return "MHz"
    if max_hz >= 1e3:
        return "kHz"
    return "Hz"


def time_to_seconds(value: float, unit: str) -> float:
    """Convert a time value in the given unit to seconds.

    >>> time_to_seconds(1000, "us")
    0.001
    >>> time_to_seconds(1, "ms")
    0.001
    """
    if unit not in TIME_UNITS:
        raise ValueError(f"Unknown time unit: {unit}. Use s, ms, or us.")
    return value * TIME_UNITS[unit]


def seconds_to_time(value_s: float, unit: str) -> float:
    """Convert a time in seconds to the given display unit.

    >>> seconds_to_time(0.001, "ms")
    1.0
    >>> seconds_to_time(0.001, "us")
    1000.0
    """
    if unit not in TIME_UNITS:
        raise ValueError(f"Unknown time unit: {unit}. Use s, ms, or us.")
    return value_s / TIME_UNITS[unit]


def auto_time_unit(duration_s: float) -> str:
    """Pick the best time unit for a given duration.

    >>> auto_time_unit(0.0005)
    'us'
    >>> auto_time_unit(0.05)
    'ms'
    >>> auto_time_unit(2.0)
    's'
    """
    if duration_s < 1e-3:
        return "us"
    if duration_s < 1.0:
        return "ms"
    return "s"


def format_time(value_s: float, unit: str | None = None, precision: int = 3) -> str:
    """Format a time in seconds for display.

    >>> format_time(0.00005, "us")
    '50.000 us'
    >>> format_time(0.001, "ms")
    '1.000 ms'
    """
    if unit is None:
        unit = auto_time_unit(value_s)
    display_value = seconds_to_time(value_s, unit)
    return f"{display_value:.{precision}f} {unit}"
