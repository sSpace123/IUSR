"""Feature extraction for single- and multi-channel ultrasonic signals."""

from __future__ import annotations

import numpy as np

from .spectrum_analysis import dominant_frequency


def compute_basic_features(
    time: np.ndarray | list[float],
    signal: np.ndarray | list[float],
    fs: float,
) -> dict[str, float]:
    """Compute common time- and frequency-domain features."""
    time_values = np.asarray(time, dtype=float)
    values = np.asarray(signal, dtype=float)
    if time_values.ndim != 1 or values.ndim != 1:
        raise ValueError("time and signal must be one-dimensional arrays.")
    if time_values.size != values.size or values.size == 0:
        raise ValueError("time and signal must be non-empty and have the same length.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    abs_values = np.abs(values)
    peak_index = int(np.argmax(abs_values))
    envelope = _envelope(values)
    envelope_peak_index = int(np.argmax(envelope))
    tof_index = _estimate_tof_index(envelope)
    duration = float(time_values[-1] - time_values[0]) if time_values.size > 1 else 0.0
    energy = float(np.sum(values**2))
    envelope_energy = float(np.sum(envelope**2))
    return {
        "peak": float(values[peak_index]),
        "abs_peak": float(abs_values[peak_index]),
        "peak_to_peak": float(np.ptp(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "energy": energy,
        "absolute_energy": float(np.sum(abs_values)),
        "average_power": float(energy / duration) if duration > 0 else 0.0,
        "mean_square_energy": float(np.mean(values**2)),
        "envelope_energy": envelope_energy,
        "envelope_area": float(np.trapezoid(envelope, time_values)),
        "peak_time": float(time_values[peak_index]),
        "tof": float(time_values[tof_index]),
        "dominant_frequency": dominant_frequency(values, fs),
        "envelope_peak": float(envelope[envelope_peak_index]),
        "envelope_peak_time": float(time_values[envelope_peak_index]),
    }


def hilbert_envelope(signal: np.ndarray | list[float]) -> np.ndarray:
    """Return the Hilbert envelope of a waveform."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    return _envelope(values)


def estimate_tof_by_envelope(
    time: np.ndarray | list[float],
    signal: np.ndarray | list[float],
    threshold_ratio: float = 0.1,
) -> float:
    """Estimate time-of-flight from the first envelope threshold crossing."""
    time_values = np.asarray(time, dtype=float)
    envelope = hilbert_envelope(signal)
    if time_values.size != envelope.size:
        raise ValueError("time and signal must have the same length.")
    index = _estimate_tof_index(envelope, threshold_ratio)
    return float(time_values[index])


def estimate_delay_by_xcorr(
    sig_ref: np.ndarray | list[float],
    sig_target: np.ndarray | list[float],
    fs: float,
) -> float:
    """Estimate target delay relative to reference using cross-correlation."""
    ref = np.asarray(sig_ref, dtype=float)
    target = np.asarray(sig_target, dtype=float)
    if ref.ndim != 1 or target.ndim != 1:
        raise ValueError("signals must be one-dimensional arrays.")
    if ref.size == 0 or target.size == 0:
        raise ValueError("signals must not be empty.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    ref = ref - np.mean(ref)
    target = target - np.mean(target)
    correlation = np.correlate(target, ref, mode="full")
    lag_samples = int(np.argmax(correlation) - (ref.size - 1))
    return lag_samples / fs


def compute_envelope_features(
    time: np.ndarray | list[float],
    signal: np.ndarray | list[float],
    fs: float,
) -> dict[str, float]:
    """Extract envelope-based features from a signal.

    Returns a dict with: envelope_peak, envelope_peak_time, envelope_rms,
    envelope_energy, envelope_rise_time, envelope_fall_time, tof_10pct, tof_50pct.
    Times are in seconds.
    """
    time_values = np.asarray(time, dtype=float)
    values = np.asarray(signal, dtype=float)
    envelope = _envelope(values)
    peak_idx = int(np.argmax(envelope))
    peak_val = float(envelope[peak_idx])
    peak_time = float(time_values[peak_idx])

    tof_10 = float(time_values[_estimate_tof_index(envelope, 0.1)])
    tof_50 = float(time_values[_estimate_tof_index(envelope, 0.5)])

    # Rise time: 10% → 90% of peak
    rise_idx = _estimate_tof_index(envelope, 0.9)
    rise_10_idx = _estimate_tof_index(envelope, 0.1)
    rise_time = float(time_values[rise_idx] - time_values[rise_10_idx]) if rise_idx > rise_10_idx else 0.0

    # Fall time: peak → 10% of peak (after peak)
    fall_10_idx = peak_idx + int(np.argmax(envelope[peak_idx:] <= peak_val * 0.1)) if peak_idx < envelope.size - 1 else envelope.size - 1
    fall_time = float(time_values[fall_10_idx] - time_values[peak_idx]) if fall_10_idx > peak_idx else 0.0

    return {
        "envelope_peak": peak_val,
        "envelope_peak_time": peak_time,
        "envelope_rms": float(np.sqrt(np.mean(envelope**2))),
        "envelope_energy": float(np.sum(envelope**2)),
        "envelope_rise_time": rise_time,
        "envelope_fall_time": fall_time,
        "tof_10pct": tof_10,
        "tof_50pct": tof_50,
    }


def estimate_arrival_time(
    time: np.ndarray | list[float],
    signal: np.ndarray | list[float],
    threshold_ratio: float = 0.1,
) -> float:
    """Estimate time-of-arrival from the first envelope threshold crossing (10% by default).

    Returns time in seconds.
    """
    time_values = np.asarray(time, dtype=float)
    envelope = hilbert_envelope(signal)
    if time_values.size != envelope.size:
        raise ValueError("时间序列和信号长度必须相同。")
    idx = _estimate_tof_index(envelope, threshold_ratio)
    return float(time_values[idx])


def _envelope(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import hilbert
    except ImportError:
        return np.abs(values)
    return np.abs(hilbert(values))


def _estimate_tof_index(envelope: np.ndarray, threshold_ratio: float = 0.1) -> int:
    if envelope.size == 0:
        return 0
    peak = float(np.max(envelope))
    if peak <= 0:
        return 0
    threshold = peak * threshold_ratio
    crossings = np.flatnonzero(envelope >= threshold)
    return int(crossings[0]) if crossings.size else int(np.argmax(envelope))
