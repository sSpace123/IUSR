"""Continuous wavelet transform utilities."""

from __future__ import annotations

import numpy as np


def compute_cwt(
    signal: np.ndarray | list[float],
    fs: float,
    f_min: float,
    f_max: float,
    num_freqs: int,
    wavelet: str = "morl",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a CWT matrix for the requested frequency range.

    Returns:
        Frequencies in Hz and a coefficient matrix shaped
        (num_freqs, samples).
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if f_min <= 0 or f_max <= 0 or f_min >= f_max:
        raise ValueError("f_min and f_max must be positive and increasing.")
    if f_max >= fs / 2.0:
        raise ValueError("f_max must be below the Nyquist frequency.")
    if num_freqs < 2:
        raise ValueError("num_freqs must be at least 2.")

    freqs = np.linspace(f_min, f_max, num_freqs)
    try:
        import pywt
    except ImportError:
        return freqs, _morlet_cwt(values, fs, freqs)

    central_frequency = pywt.central_frequency(wavelet)
    scales = central_frequency * fs / freqs
    coefficients, pywt_freqs = pywt.cwt(values, scales, wavelet, sampling_period=1.0 / fs)
    return np.asarray(pywt_freqs), np.asarray(coefficients)


def _morlet_cwt(signal: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    coefficients = []
    for frequency in freqs:
        cycles = 6.0
        sigma = cycles / (2.0 * np.pi * frequency)
        half_width = max(int(np.ceil(4.0 * sigma * fs)), 8)
        t = np.arange(-half_width, half_width + 1, dtype=float) / fs
        wavelet = np.exp(2j * np.pi * frequency * t) * np.exp(-(t**2) / (2.0 * sigma**2))
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        coefficients.append(_convolve_same_length(centered, np.conj(wavelet[::-1])))
    return np.asarray(coefficients)


def _convolve_same_length(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.convolve(signal, kernel, mode="full")
    start = (result.size - signal.size) // 2
    return result[start : start + signal.size]
