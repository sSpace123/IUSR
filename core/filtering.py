"""Filtering helpers for narrow-band ultrasonic signal extraction."""

from __future__ import annotations

import numpy as np


# ── Existing API (backward compatible) ──


def bandpass_filter(
    signal: np.ndarray | list[float],
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter via sosfiltfilt.

    Falls back to an FFT-domain mask when scipy is unavailable.
    """
    values = _validate_filter_inputs(signal, fs, lowcut, highcut, order)
    return _apply_sos_bandpass(values, fs, lowcut, highcut, order)


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
    """Shift a filtered signal left by half a toneburst window and zero-pad."""
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


# ── New: Bandpass parameter validation ──


def validate_bandpass_params(
    fs: float,
    center_freq: float,
    bandwidth: float,
    lowcut: float | None = None,
    highcut: float | None = None,
) -> tuple[float, float]:
    """Validate narrow-band extraction parameters and return (lowcut, highcut) in Hz.

    Raises ValueError with Chinese messages on invalid inputs.
    """
    if fs <= 0:
        raise ValueError("采样率必须大于 0。")
    if center_freq <= 0:
        raise ValueError("中心频率必须大于 0。")
    if bandwidth <= 0:
        raise ValueError("带宽必须大于 0。")

    if lowcut is None or highcut is None:
        lowcut = center_freq - bandwidth / 2.0
        highcut = center_freq + bandwidth / 2.0

    nyquist = fs / 2.0
    if lowcut <= 0:
        raise ValueError(f"下限频率 {lowcut:.3g} Hz 必须大于 0。请减小带宽或增大中心频率。")
    if highcut >= nyquist:
        raise ValueError(
            f"上限频率 {highcut:.3g} Hz 不能超过 Nyquist 频率 {nyquist:.3g} Hz。"
            "请减小带宽或降低中心频率。"
        )
    if lowcut >= highcut:
        raise ValueError("下限频率必须小于上限频率。")
    return float(lowcut), float(highcut)


# ── New: Narrowband wave packet extraction ──


def extract_narrowband_wave_packet(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    center_freq: float,
    bandwidth: float,
    order: int = 4,
    zero_phase: bool = True,
    remove_dc: bool = True,
    auto_locate: bool = True,
    center_time: float | None = None,
    window_length: float | None = None,
    window_type: str = "tukey",
    output_mode: str = "segment",
    normalization: str = "max_abs",
) -> dict:
    """Extract a local narrowband wave packet from a signal.

    All internal units are SI: time in seconds, frequency in Hz.

    Parameters
    ----------
    time : ndarray
        Time vector in seconds.
    signal : ndarray
        Signal values.
    fs : float
        Sample rate in Hz.
    center_freq : float
        Center frequency in Hz.
    bandwidth : float
        Bandwidth in Hz.
    order : int
        Butterworth filter order (default 4).
    zero_phase : bool
        Use zero-phase sosfiltfilt.
    remove_dc : bool
        Subtract mean before filtering.
    auto_locate : bool
        Use Hilbert envelope max to locate the wave packet center.
    center_time : float or None
        Manual center time in seconds (used when auto_locate=False).
    window_length : float or None
        Extraction window length in seconds. If None, auto-set to 10/center_freq.
    window_type : str
        Window: "none", "hann", "hamming", "tukey".
    output_mode : str
        "segment" for local slice, "full_zero" for full-length zeroed outside.
    normalization : str
        "none", "max_abs", or "rms".

    Returns
    -------
    dict with keys:
        time, signal, filtered_full, envelope_full, peak_time,
        lowcut, highcut, params
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or time.size != signal.size:
        raise ValueError("时间序列和信号必须是一维数组，且长度相同。")
    if time.size < 3:
        raise ValueError("信号长度至少需要 3 个采样点。")

    # 1. Remove DC
    if remove_dc:
        signal = signal - _robust_baseline(signal)

    # 2. Compute cutoff frequencies
    lowcut = center_freq - bandwidth / 2.0
    highcut = center_freq + bandwidth / 2.0

    # 3. Validate
    lowcut, highcut = validate_bandpass_params(
        fs, center_freq, bandwidth, lowcut, highcut
    )

    # 4. Bandpass filter
    try:
        filtered = _apply_sos_bandpass(signal, fs, lowcut, highcut, order, zero_phase=zero_phase)
    except Exception as exc:
        raise RuntimeError(
            f"带通滤波失败：{exc}。请尝试降低滤波器阶数或检查频率参数。"
        ) from exc

    # 5. Hilbert envelope for peak localization
    envelope = _hilbert_envelope(filtered)

    # 6. Auto-locate peak_time
    if auto_locate:
        peak_idx = int(np.argmax(envelope))
        peak_time = float(time[peak_idx])
    else:
        if center_time is None:
            raise ValueError("请提供手动中心时间，或开启自动定位。")
        peak_time = float(center_time)

    # 7. Determine window_length
    if window_length is None or window_length <= 0:
        window_length = 10.0 / max(center_freq, 1e-9)
    # Clamp to available data
    available_dur = time[-1] - time[0]
    window_length = min(window_length, available_dur)
    half_win = window_length / 2.0

    # 8. Cut or zero
    t_start = peak_time - half_win
    t_end = peak_time + half_win

    if output_mode == "segment":
        mask = (time >= t_start) & (time <= t_end)
        if not np.any(mask):
            raise ValueError("时间窗口内没有数据点。请调整中心时间或窗口长度。")
        indices = np.flatnonzero(mask)
        i_start, i_end = int(indices[0]), int(indices[-1]) + 1
        packet_time = time[i_start:i_end].copy()
        packet_signal = filtered[i_start:i_end].copy()
    elif output_mode == "full_zero":
        packet_time = time.copy()
        packet_signal = np.zeros_like(filtered)
        mask = (time >= t_start) & (time <= t_end)
        packet_signal[mask] = filtered[mask]
    else:
        raise ValueError(f"未知输出模式: {output_mode}。支持 segment 或 full_zero。")

    # 9. Apply window function
    if window_type != "none" and output_mode == "segment" and packet_signal.size > 2:
        win = _make_window(packet_signal.size, window_type)
        packet_signal = packet_signal * win
    elif window_type != "none" and output_mode == "full_zero" and np.any(mask):
        win_len = int(np.sum(mask))
        if win_len > 2:
            full_win = np.zeros_like(packet_signal)
            full_win[mask] = _make_window(win_len, window_type)
            packet_signal = packet_signal * full_win

    # 10. Normalization
    packet_signal = _apply_normalization(packet_signal, normalization)

    return {
        "time": packet_time,
        "signal": packet_signal,
        "filtered_full": filtered,
        "envelope_full": envelope,
        "peak_time": peak_time,
        "lowcut": lowcut,
        "highcut": highcut,
        "params": {
            "fs": fs,
            "center_freq": center_freq,
            "bandwidth": bandwidth,
            "lowcut": lowcut,
            "highcut": highcut,
            "filter_order": order,
            "zero_phase": zero_phase,
            "remove_dc": remove_dc,
            "peak_time": peak_time,
            "window_length": window_length,
            "window_type": window_type,
            "output_mode": output_mode,
            "normalization": normalization,
        },
    }


# ── Internal helpers ──


def _apply_sos_bandpass(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    zero_phase: bool = True,
) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt, sosfiltfilt

        sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
        if zero_phase:
            return sosfiltfilt(sos, signal)
        return sosfilt(sos, signal)
    except ImportError:
        return _fft_bandpass(signal, fs, lowcut, highcut)
    except Exception as exc:
        raise RuntimeError(f"sosfiltfilt 滤波失败：{exc}") from exc


def _hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import hilbert

        return np.abs(hilbert(signal))
    except ImportError:
        return np.abs(signal)


def _robust_baseline(signal: np.ndarray) -> float:
    """Estimate baseline from quiet edges instead of the full waveform."""
    if signal.size < 20:
        return float(np.mean(signal))
    edge_count = max(10, signal.size // 10)
    edges = np.concatenate([signal[:edge_count], signal[-edge_count:]])
    finite = edges[np.isfinite(edges)]
    if finite.size == 0:
        return float(np.mean(signal))
    return float(np.median(finite))


def _make_window(n: int, window_type: str) -> np.ndarray:
    if window_type == "hann":
        return np.hanning(n)
    if window_type == "hamming":
        return np.hamming(n)
    if window_type == "tukey":
        try:
            from scipy.signal.windows import tukey
        except ImportError:
            from scipy.signal import tukey
        return tukey(n, alpha=0.25)
    if window_type == "none":
        return np.ones(n)
    raise ValueError(f"未知窗函数类型: {window_type}。支持 none, hann, hamming, tukey。")


def _apply_normalization(signal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return signal
    if mode == "max_abs":
        peak = float(np.max(np.abs(signal)))
        return signal / peak if peak > 0 else signal
    if mode == "rms":
        rms = float(np.sqrt(np.mean(signal**2)))
        return signal / rms if rms > 0 else signal
    raise ValueError(f"未知归一化方式: {mode}。支持 none, max_abs, rms。")


# ── Private legacy helpers ──


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
