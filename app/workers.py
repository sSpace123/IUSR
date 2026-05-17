"""Background worker threads for heavy signal processing operations."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal


class CWTWorker(QThread):
    """Compute CWT in the background, emitting progress and results."""

    finished = Signal(object)  # dict: {frequencies, coefficients, cost_info}
    error = Signal(str)
    progress = Signal(int)  # 0-100

    def __init__(
        self,
        signal: np.ndarray,
        fs: float,
        f_min: float,
        f_max: float,
        num_freqs: int,
        wavelet: str = "cmor1.5-1.0",
        parent=None,
    ):
        super().__init__(parent)
        self.signal = np.asarray(signal, dtype=float)
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.num_freqs = num_freqs
        self.wavelet = wavelet
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            from core.wavelet_analysis import compute_cwt, estimate_cwt_cost

            self.progress.emit(10)

            cost = estimate_cwt_cost(self.signal.size, self.num_freqs)
            self.progress.emit(20)

            freqs, coefficients = compute_cwt(
                self.signal, self.fs, self.f_min, self.f_max, self.num_freqs, self.wavelet
            )

            if self._is_cancelled:
                self.error.emit("计算已取消。")
                return

            self.progress.emit(90)

            self.finished.emit({
                "frequencies": freqs,
                "coefficients": coefficients,
                "cost_info": {
                    "input_points": self.signal.size,
                    "num_freqs": self.num_freqs,
                    "estimated_cost": cost,
                },
            })
            self.progress.emit(100)
        except Exception as exc:
            self.error.emit(f"小波变换失败：{exc}")


class FilteringWorker(QThread):
    """Run narrowband wave packet extraction in the background."""

    finished = Signal(object)  # dict: result from extract_narrowband_wave_packet
    error = Signal(str)
    progress = Signal(int)

    def __init__(
        self,
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
        parent=None,
    ):
        super().__init__(parent)
        self.time = np.asarray(time, dtype=float)
        self.signal = np.asarray(signal, dtype=float)
        self.fs = fs
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.order = order
        self.zero_phase = zero_phase
        self.remove_dc = remove_dc
        self.auto_locate = auto_locate
        self.center_time = center_time
        self.window_length = window_length
        self.window_type = window_type
        self.output_mode = output_mode
        self.normalization = normalization
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            from core.filtering import extract_narrowband_wave_packet

            self.progress.emit(20)
            result = extract_narrowband_wave_packet(
                self.time, self.signal, self.fs,
                center_freq=self.center_freq,
                bandwidth=self.bandwidth,
                order=self.order,
                zero_phase=self.zero_phase,
                remove_dc=self.remove_dc,
                auto_locate=self.auto_locate,
                center_time=self.center_time,
                window_length=self.window_length,
                window_type=self.window_type,
                output_mode=self.output_mode,
                normalization=self.normalization,
            )

            if self._is_cancelled:
                self.error.emit("计算已取消。")
                return

            self.progress.emit(90)
            self.finished.emit(result)
            self.progress.emit(100)
        except Exception as exc:
            self.error.emit(f"窄带波包提取失败：{exc}")


class FeatureWorker(QThread):
    """Compute features for multiple channels in the background."""

    finished = Signal(object)  # list[dict]
    error = Signal(str)
    progress = Signal(int)

    def __init__(
        self,
        time: np.ndarray,
        channels: dict[str, np.ndarray],
        fs: float,
        parent=None,
    ):
        super().__init__(parent)
        self.time = np.asarray(time, dtype=float)
        self.channels = channels
        self.fs = fs
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            from core.feature_extraction import compute_basic_features

            total = max(len(self.channels), 1)
            rows = []
            for i, (name, values) in enumerate(self.channels.items()):
                if self._is_cancelled:
                    self.error.emit("计算已取消。")
                    return
                feats = compute_basic_features(self.time, values, self.fs)
                rows.append({"channel": name, **feats})
                self.progress.emit(int((i + 1) / total * 100))

            self.finished.emit(rows)
            self.progress.emit(100)
        except Exception as exc:
            self.error.emit(f"特征计算失败：{exc}")
