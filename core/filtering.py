"""Filtering helpers for narrow-band ultrasonic signal extraction."""

from __future__ import annotations

import numpy as np


def bandpass_filter(
    signal: np.ndarray | list[float],
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter when scipy is available.

    If scipy is not installed, the function falls back to a deterministic
    FFT-domain band-pass mask. The fallback keeps the application usable for
    preview and tests, but production filtering should install scipy.
    """
    values = _validate_filter_inputs(signal, fs, lowcut, highcut, order)
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        return _fft_bandpass(values, fs, lowcut, highcut)

    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, values)


def narrowband_filter(
    signal: np.ndarray | list[float],
    fs: float,
    center_frequency: float,
    bandwidth: float,
    order: int = 4,
    cycles: float = 3.0,
) -> np.ndarray:
    """Filter around center_frequency and align by half the toneburst period."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    lowcut = center_frequency - bandwidth / 2.0
    highcut = center_frequency + bandwidth / 2.0
    filtered = bandpass_filter(signal, fs, lowcut, highcut, order=order)
    return align_by_toneburst_period(filtered, fs, center_frequency, cycles)


def align_by_toneburst_period(
    signal: np.ndarray | list[float],
    fs: float,
    center_frequency: float,
    cycles: float = 3.0,
) -> np.ndarray:
    """Shift a filtered signal left by half a toneburst window and zero-pad.

    This mirrors the MATLAB narrow-band extraction post-processing:
    LenT = floor(fs / fc * Cyc), then the result is advanced by floor(LenT / 2)
    samples. It is useful when comparing extracted toneburst-like responses.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    shift = toneburst_half_period_samples(fs, center_frequency, cycles)
    if shift <= 0:
        return values.copy()
    if shift >= values.size:
        return np.zeros_like(values)
    aligned = np.zeros_like(values)
    aligned[: values.size - shift] = values[shift:]
    return aligned


def toneburst_half_period_samples(fs: float, center_frequency: float, cycles: float = 3.0) -> int:
    """Return floor(floor(fs / fc * cycles) / 2)."""
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if center_frequency <= 0:
        raise ValueError("center_frequency must be positive.")
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    toneburst_length = int(np.floor(fs / center_frequency * cycles))
    return int(np.floor(toneburst_length / 2))


def _validate_filter_inputs(
    signal: np.ndarray | list[float],
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("signal must be a one-dimensional array with at least 3 samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal contains NaN or infinite values.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if order < 1:
        raise ValueError("order must be at least 1.")
    nyquist = fs / 2.0
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("filter cutoff frequencies must be positive.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be smaller than highcut.")
    if highcut >= nyquist:
        raise ValueError("highcut must be below the Nyquist frequency.")
    return values


def _fft_bandpass(signal: np.ndarray, fs: float, lowcut: float, highcut: float) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    mask = (freqs >= lowcut) & (freqs <= highcut)
    return np.fft.irfft(spectrum * mask, n=signal.size)
