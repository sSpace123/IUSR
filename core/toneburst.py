"""Toneburst preview helpers."""

from __future__ import annotations

import numpy as np


def generate_toneburst_preview(
    time: np.ndarray,
    center_frequency: float,
    cycles: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a smooth Hann-windowed toneburst on the provided time axis."""
    values = np.zeros_like(time, dtype=float)
    if time.size < 2 or center_frequency <= 0 or cycles <= 0:
        return values

    duration = cycles / center_frequency
    dt = float(np.median(np.diff(time)))
    count = max(int(round(duration / dt)), 3)
    count = min(count, time.size)
    start = max((time.size - count) // 4, 0)
    local_t = np.arange(count, dtype=float) * dt
    window = np.hanning(count)
    burst = amplitude * window * np.sin(2.0 * np.pi * center_frequency * local_t)
    values[start : start + count] = burst
    return values
