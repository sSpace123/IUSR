"""Continuous wavelet transform utilities with performance safeguards."""

from __future__ import annotations

import warnings

import numpy as np


def compute_cwt(
    signal: np.ndarray | list[float],
    fs: float,
    f_min: float,
    f_max: float,
    num_freqs: int,
    wavelet: str = "morl",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a CWT matrix shaped as (frequencies, samples)."""
    values = np.asarray(signal, dtype=float)
    validate_cwt_params(fs, f_min, f_max, num_freqs)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")

    values = _prepare_values(values)
    freqs = np.linspace(f_min, f_max, num_freqs)
    try:
        import pywt

        central_frequency = pywt.central_frequency(wavelet)
        scales = central_frequency * fs / freqs
        coefficients, pywt_freqs = pywt.cwt(
            values, scales, wavelet, sampling_period=1.0 / fs
        )
        return np.asarray(pywt_freqs, dtype=float), np.asarray(coefficients)
    except ImportError:
        return freqs, _morlet_cwt(values, fs, freqs)


def validate_cwt_params(fs: float, f_min: float, f_max: float, num_freqs: int) -> None:
    """Validate CWT parameters."""
    if fs <= 0:
        raise ValueError("sample rate must be greater than 0.")
    if f_min <= 0:
        raise ValueError("minimum frequency must be greater than 0.")
    if f_max <= 0:
        raise ValueError("maximum frequency must be greater than 0.")
    if f_min >= f_max:
        raise ValueError("minimum frequency must be smaller than maximum frequency.")
    if f_max >= fs / 2.0:
        raise ValueError(f"maximum frequency {f_max:.3g} Hz must be below Nyquist {fs / 2.0:.3g} Hz.")
    if num_freqs < 20:
        raise ValueError("frequency points must be at least 20.")
    if num_freqs > 300:
        raise ValueError("frequency points must not exceed 300.")


def estimate_cwt_cost(input_points: int, num_freqs: int) -> int:
    """Return an estimated computational cost for CWT."""
    return input_points * num_freqs


def prepare_signal_for_cwt(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    time_range: tuple[float, float] | None = None,
    max_points: int = 30000,
) -> dict:
    """Slice and optionally decimate a signal for CWT computation."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or time.size != signal.size:
        raise ValueError("time and signal must be one-dimensional arrays with the same length.")

    original_points = signal.size
    if time_range is not None:
        t_start, t_end = time_range
        mask = (time >= t_start) & (time <= t_end)
        if not np.any(mask):
            raise ValueError("selected CWT time range contains no samples.")
        indices = np.flatnonzero(mask)
        i_start, i_end = int(indices[0]), int(indices[-1]) + 1
        time = time[i_start:i_end]
        signal = signal[i_start:i_end]

    decimation_factor = 1
    if signal.size > max_points:
        decimation_factor = int(np.ceil(signal.size / max_points))
        time = time[::decimation_factor]
        signal = signal[::decimation_factor]
        fs = fs / decimation_factor

    return {
        "time": time,
        "signal": signal,
        "fs": fs,
        "decimation_factor": decimation_factor,
        "original_points": original_points,
    }


def compute_cwt_optimized(
    signal: np.ndarray,
    fs: float,
    f_min: float,
    f_max: float,
    num_freqs: int = 100,
    wavelet: str = "cmor1.5-1.0",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CWT with parameter validation and cost warning."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")

    validate_cwt_params(fs, f_min, f_max, num_freqs)
    cost = estimate_cwt_cost(values.size, num_freqs)
    if cost > 10_000_000:
        warnings.warn(
            f"CWT cost is high ({cost:,}); consider a shorter time range or fewer frequency points.",
            RuntimeWarning,
            stacklevel=2,
        )
    return compute_cwt(values, fs, f_min, f_max, num_freqs, wavelet)


def _prepare_values(values: np.ndarray) -> np.ndarray:
    values = _interpolate_nonfinite(values)
    values = values - _robust_baseline(values)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak > 0:
        values = values / peak
    return values


def _morlet_cwt(signal: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    centered = _prepare_values(signal)
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


def _robust_baseline(values: np.ndarray) -> float:
    if values.size < 20:
        return float(np.mean(values))
    edge_count = max(10, values.size // 10)
    edges = np.concatenate([values[:edge_count], values[-edge_count:]])
    finite = edges[np.isfinite(edges)]
    if finite.size == 0:
        return float(np.mean(values))
    return float(np.median(finite))


def _interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if np.all(finite):
        return values
    if not np.any(finite):
        raise ValueError("signal contains no finite samples.")
    repaired = values.copy()
    if np.count_nonzero(finite) == 1:
        repaired[:] = float(values[finite][0])
        return repaired
    x = np.arange(values.size, dtype=float)
    repaired[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return repaired
