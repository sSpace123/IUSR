"""Frequency-domain analysis utilities."""

from __future__ import annotations

import numpy as np


def compute_fft(signal: np.ndarray | list[float], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided amplitude spectrum.

    Returns:
        A tuple of frequency bins in Hz and amplitudes.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains NaN or infinite values.")

    centered = values - np.mean(values)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    amplitudes = np.abs(np.fft.rfft(centered)) / centered.size
    if centered.size > 1:
        amplitudes[1:-1] *= 2.0
    return freqs, amplitudes


def dominant_frequency(signal: np.ndarray | list[float], fs: float) -> float:
    """Return the non-DC frequency with the largest amplitude."""
    freqs, amplitudes = compute_fft(signal, fs)
    if freqs.size <= 1:
        return 0.0
    index = int(np.argmax(amplitudes[1:]) + 1)
    return float(freqs[index])

