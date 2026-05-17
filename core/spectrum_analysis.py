"""Frequency-domain analysis utilities."""

from __future__ import annotations

import numpy as np


# ── Existing API ──


def compute_fft(signal: np.ndarray | list[float], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided amplitude spectrum. Returns (freqs in Hz, amplitudes)."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("信号必须是非空的一维数组。")
    if fs <= 0:
        raise ValueError("采样率必须大于 0。")
    if not np.all(np.isfinite(values)):
        raise ValueError("信号包含 NaN 或无限值。")

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


# ── New: DC removal ──


def remove_dc_component(signal: np.ndarray | list[float]) -> np.ndarray:
    """Subtract the mean from a signal."""
    values = np.asarray(signal, dtype=float)
    return values - np.mean(values)


# ── New: dB-scale spectrum ──


def compute_fft_db(
    signal: np.ndarray | list[float],
    fs: float,
    ref: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided amplitude spectrum in dB.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sample rate in Hz.
    ref : str
        Reference for dB: "max" → 20*log10(amplitude / max_amplitude).

    Returns
    -------
    (freqs in Hz, amplitudes in dB)
    """
    freqs, amplitudes = compute_fft(signal, fs)
    if ref == "max":
        ref_val = np.max(amplitudes) if amplitudes.size else 1.0
    else:
        ref_val = float(ref)
    if ref_val <= 0:
        ref_val = 1e-12
    db = 20 * np.log10(np.maximum(amplitudes, ref_val * 1e-12) / ref_val)
    return freqs, db


# ── New: Dominant frequency with DC exclusion ──


def find_dominant_frequency(
    signal: np.ndarray | list[float],
    fs: float,
    exclude_dc: bool = True,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
) -> dict:
    """Find the dominant frequency, optionally excluding the DC region.

    The DC exclusion region is 0 to max(1, fs/N*3) Hz.

    Returns
    -------
    dict with keys:
        dominant_hz : float — dominant frequency in Hz (0 if not found)
        amplitude : float — amplitude at dominant frequency
        index : int — bin index
        num_bins_scanned : int — number of bins used (after exclusion)
    """
    values = np.asarray(signal, dtype=float)
    N = values.size
    freqs, amplitudes = compute_fft(values, fs)

    if freqs.size <= 1:
        return {"dominant_hz": 0.0, "amplitude": 0.0, "index": 0, "num_bins_scanned": 0}

    mask = np.ones_like(freqs, dtype=bool)
    if exclude_dc:
        dc_cutoff = max(1.0, fs / N * 3.0)
        mask &= freqs >= dc_cutoff
    if min_frequency is not None:
        mask &= freqs >= float(min_frequency)
    if max_frequency is not None:
        mask &= freqs <= float(max_frequency)
    if not np.any(mask):
        return {"dominant_hz": 0.0, "amplitude": 0.0, "index": 0, "num_bins_scanned": 0}
    region_freqs = freqs[mask]
    region_amps = amplitudes[mask]
    region_idx = int(np.argmax(region_amps))
    return {
        "dominant_hz": float(region_freqs[region_idx]),
        "amplitude": float(region_amps[region_idx]),
        "index": int(np.flatnonzero(mask)[region_idx]),
        "num_bins_scanned": int(np.sum(mask)),
    }

    idx = int(np.argmax(amplitudes))
    return {
        "dominant_hz": float(freqs[idx]),
        "amplitude": float(amplitudes[idx]),
        "index": idx,
        "num_bins_scanned": freqs.size,
    }
