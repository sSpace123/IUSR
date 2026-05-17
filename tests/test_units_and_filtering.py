"""Tests for units, filtering, wavelet, and spectrum enhancements."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.data_loader import load_signal_file
from core.units import (
    auto_freq_unit,
    auto_time_unit,
    format_frequency,
    format_time,
    frequency_to_hz,
    hz_to_frequency,
    seconds_to_time,
    time_to_seconds,
)
from core.filtering import extract_narrowband_wave_packet, validate_bandpass_params
from core.wavelet_analysis import (
    compute_cwt_optimized,
    estimate_cwt_cost,
    prepare_signal_for_cwt,
    validate_cwt_params,
)
from core.spectrum_analysis import (
    compute_fft_db,
    find_dominant_frequency,
    remove_dc_component,
)


class UnitsTests(unittest.TestCase):
    """Frequency and time unit conversion tests."""

    def test_frequency_to_hz_khz(self) -> None:
        self.assertAlmostEqual(frequency_to_hz(500, "kHz"), 500_000.0)

    def test_frequency_to_hz_mhz(self) -> None:
        self.assertAlmostEqual(frequency_to_hz(0.5, "MHz"), 500_000.0)

    def test_frequency_to_hz_hz(self) -> None:
        self.assertAlmostEqual(frequency_to_hz(500_000, "Hz"), 500_000.0)

    def test_hz_to_frequency_khz(self) -> None:
        self.assertAlmostEqual(hz_to_frequency(500_000, "kHz"), 500.0)

    def test_hz_to_frequency_mhz(self) -> None:
        self.assertAlmostEqual(hz_to_frequency(500_000, "MHz"), 0.5)

    def test_hz_roundtrip(self) -> None:
        """Switching units should preserve internal Hz value."""
        internal_hz = 500_000.0
        khz = hz_to_frequency(internal_hz, "kHz")
        mhz = hz_to_frequency(internal_hz, "MHz")
        self.assertAlmostEqual(frequency_to_hz(khz, "kHz"), internal_hz)
        self.assertAlmostEqual(frequency_to_hz(mhz, "MHz"), internal_hz)

    def test_unknown_freq_unit_raises(self) -> None:
        with self.assertRaises(ValueError):
            frequency_to_hz(1, "GHz")


class DataLoadingRepairTests(unittest.TestCase):
    """Non-finite samples should not collapse the shared time axis."""

    def test_inf_in_one_channel_is_repaired_without_dropping_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.csv"
            path.write_text(
                "TIME,CH1,CH2\n"
                "0.0,1.0,10.0\n"
                "0.1,2.0,inf\n"
                "0.2,3.0,30.0\n"
                "0.3,4.0,40.0\n",
                encoding="utf-8",
            )
            signal = load_signal_file(path)

        self.assertEqual(signal.time.size, 4)
        self.assertEqual(signal.channels["CH1"].size, 4)
        self.assertTrue(np.all(np.isfinite(signal.channels["CH2"])))
        self.assertAlmostEqual(signal.channels["CH2"][1], 20.0)

    def test_time_to_seconds_us(self) -> None:
        self.assertAlmostEqual(time_to_seconds(1000, "us"), 0.001)

    def test_time_to_seconds_ms(self) -> None:
        self.assertAlmostEqual(time_to_seconds(1, "ms"), 0.001)

    def test_seconds_to_time_ms(self) -> None:
        self.assertAlmostEqual(seconds_to_time(0.001, "ms"), 1.0)

    def test_time_roundtrip(self) -> None:
        internal_s = 0.001
        ms = seconds_to_time(internal_s, "ms")
        us = seconds_to_time(internal_s, "us")
        self.assertAlmostEqual(time_to_seconds(ms, "ms"), internal_s)
        self.assertAlmostEqual(time_to_seconds(us, "us"), internal_s)

    def test_auto_time_unit_us(self) -> None:
        self.assertEqual(auto_time_unit(0.0005), "us")

    def test_auto_time_unit_ms(self) -> None:
        self.assertEqual(auto_time_unit(0.05), "ms")

    def test_auto_time_unit_s(self) -> None:
        self.assertEqual(auto_time_unit(2.0), "s")

    def test_auto_freq_unit_hz(self) -> None:
        self.assertEqual(auto_freq_unit(500), "Hz")

    def test_auto_freq_unit_khz(self) -> None:
        self.assertEqual(auto_freq_unit(5000), "kHz")

    def test_auto_freq_unit_mhz(self) -> None:
        self.assertEqual(auto_freq_unit(5e6), "MHz")

    def test_format_frequency(self) -> None:
        self.assertIn("500.000 kHz", format_frequency(500_000, "kHz"))

    def test_format_time(self) -> None:
        self.assertIn("50.000 us", format_time(0.000050, "us"))


class BandpassValidationTests(unittest.TestCase):
    """Parameter validation for bandpass filtering."""

    def test_valid_params_pass(self) -> None:
        low, high = validate_bandpass_params(2e6, 500e3, 200e3)
        self.assertAlmostEqual(low, 400e3)
        self.assertAlmostEqual(high, 600e3)

    def test_center_freq_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_bandpass_params(1e6, 0, 200e3)

    def test_bandwidth_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_bandpass_params(1e6, 500e3, 0)

    def test_highcut_above_nyquist_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_bandpass_params(1e6, 800e3, 600e3)

    def test_lowcut_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_bandpass_params(1e6, 50e3, 200e3)  # lowcut = 50e3 - 100e3 = -50e3


class NarrowbandExtractionTests(unittest.TestCase):
    """Narrowband wave packet extraction tests."""

    def setUp(self) -> None:
        self.fs = 10e6  # 10 MHz
        self.duration = 100e-6  # 100 us
        self.time = np.arange(0, self.duration, 1 / self.fs)
        # Create a 500 kHz sinusoidal tone burst at 50 us, 20 cycles ≈ 40 us long
        center_f = 500e3
        t0 = 50e-6
        burst_width = 20e-6  # 20 us
        envelope = np.exp(-0.5 * ((self.time - t0) / (burst_width / 4)) ** 2)
        self.signal = envelope * np.sin(2 * np.pi * center_f * self.time)
        # Add some noise
        self.signal += 0.02 * np.random.randn(self.time.size)

    def test_extract_returns_oscillating_waveform(self) -> None:
        """The result should be an oscillating waveform, not just an envelope."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            output_mode="segment",
            normalization="none",
        )
        # Check the resulting waveform oscillates (has zero crossings)
        sig = result["signal"]
        zero_crossings = np.sum(np.diff(np.signbit(sig)))
        self.assertGreater(zero_crossings, 4, "波包应振荡穿过零轴，不应只是包络")

    def test_extract_peak_time_near_true_center(self) -> None:
        """Peak time should be near the true burst center (50 us)."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            output_mode="segment",
        )
        peak_t = result["peak_time"]
        self.assertAlmostEqual(peak_t, 50e-6, delta=5e-6,
                               msg=f"峰值时间 {peak_t * 1e6:.1f} us 应接近 50 us")

    def test_extract_normalization_max_abs(self) -> None:
        """max_abs normalization should produce peak == 1."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            normalization="max_abs",
        )
        self.assertAlmostEqual(float(np.max(np.abs(result["signal"]))), 1.0, places=2)

    def test_extract_output_mode_segment(self) -> None:
        """Segment mode should produce shorter output than input."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            output_mode="segment",
        )
        self.assertLess(result["time"].size, self.time.size,
                        "段模式输出应短于原始信号")

    def test_extract_output_mode_full_zero(self) -> None:
        """Full_zero mode should preserve input length."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            output_mode="full_zero",
        )
        self.assertEqual(result["time"].size, self.time.size,
                         "全长置零模式应保持原始长度")
        # Most of the result should be zero
        nonzero_ratio = np.sum(np.abs(result["signal"]) > 1e-9) / result["signal"].size
        self.assertLess(nonzero_ratio, 0.5, "大部分值应为零")

    def test_extract_with_tukey_window(self) -> None:
        """Tukey window should produce clean edges."""
        result = extract_narrowband_wave_packet(
            self.time, self.signal, self.fs,
            center_freq=500e3, bandwidth=200e3,
            window_length=20e-6,
            window_type="tukey",
            output_mode="segment",
            normalization="none",
        )
        sig = result["signal"]
        # Edges should be near zero
        self.assertLess(abs(sig[0]), abs(np.max(sig)) * 0.3,
                        "加窗后起始边缘应接近零")


class HilbertPeakTests(unittest.TestCase):
    """Hilbert envelope peak localization tests."""

    def test_envelope_peak_near_burst_center(self) -> None:
        fs = 10e6
        t = np.arange(0, 100e-6, 1 / fs)
        t0 = 50e-6
        width = 10e-6
        envelope = np.exp(-0.5 * ((t - t0) / (width / 4)) ** 2)
        sig = envelope * np.sin(2 * np.pi * 500e3 * t)
        result = extract_narrowband_wave_packet(
            t, sig, fs, center_freq=500e3, bandwidth=200e3,
            window_length=20e-6, auto_locate=True,
        )
        self.assertAlmostEqual(result["peak_time"], t0, delta=5e-6,
                               msg="Hilbert 包络应定位到波包中心")


class CWTValidationTests(unittest.TestCase):
    """CWT parameter validation tests."""

    def test_valid_cwt_params_pass(self) -> None:
        validate_cwt_params(1e6, 100e3, 400e3, 100)

    def test_f_min_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_cwt_params(1e6, 0, 400e3, 100)

    def test_f_max_above_nyquist_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_cwt_params(1e6, 100e3, 600e3, 100)

    def test_f_min_ge_f_max_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_cwt_params(1e6, 400e3, 100e3, 100)

    def test_num_freqs_too_low_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_cwt_params(1e6, 100e3, 400e3, 10)

    def test_num_freqs_too_high_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_cwt_params(1e6, 100e3, 400e3, 500)


class CWTDownsampleTests(unittest.TestCase):
    """CWT signal preparation and decimation tests."""

    def setUp(self) -> None:
        self.fs = 10e6
        self.time = np.arange(0, 1e-3, 1 / self.fs)  # 10000 points
        self.signal = np.sin(2 * np.pi * 500e3 * self.time)

    def test_prepare_cwt_no_decimation_needed(self) -> None:
        prep = prepare_signal_for_cwt(self.time, self.signal, self.fs, max_points=20000)
        self.assertEqual(prep["decimation_factor"], 1)
        self.assertEqual(prep["signal"].size, self.signal.size)

    def test_prepare_cwt_with_decimation(self) -> None:
        prep = prepare_signal_for_cwt(self.time, self.signal, self.fs, max_points=5000)
        self.assertGreater(prep["decimation_factor"], 1)
        self.assertLessEqual(prep["signal"].size, 5000)

    def test_prepare_cwt_fs_updated_after_decimation(self) -> None:
        prep = prepare_signal_for_cwt(self.time, self.signal, self.fs, max_points=5000)
        if prep["decimation_factor"] > 1:
            self.assertEqual(prep["fs"], self.fs / prep["decimation_factor"])

    def test_estimate_cwt_cost(self) -> None:
        cost = estimate_cwt_cost(10000, 100)
        self.assertEqual(cost, 1_000_000)


class SpectrumTests(unittest.TestCase):
    """Spectrum analysis tests."""

    def test_remove_dc(self) -> None:
        sig = np.array([5.0, 6.0, 4.0, 5.0])
        result = remove_dc_component(sig)
        self.assertAlmostEqual(float(np.mean(result)), 0.0)

    def test_find_dominant_excludes_dc(self) -> None:
        """DC + 500 kHz sine — dominant should be ~500 kHz not DC."""
        fs = 10e6
        t = np.arange(0, 100e-6, 1 / fs)
        sig = 10.0 + np.sin(2 * np.pi * 500e3 * t)  # large DC offset
        dom = find_dominant_frequency(sig, fs, exclude_dc=True)
        self.assertAlmostEqual(dom["dominant_hz"], 500e3, delta=10e3,
                               msg=f"主频 {dom['dominant_hz']:.0f} Hz 应接近 500000 Hz，非 DC")
        self.assertGreater(dom["num_bins_scanned"], 0)

    def test_compute_fft_db(self) -> None:
        fs = 1000
        t = np.arange(1000) / fs
        sig = np.sin(2 * np.pi * 50 * t)
        freqs, db = compute_fft_db(sig, fs)
        self.assertEqual(freqs.shape, db.shape)
        # Peak should be near 0 dB
        self.assertAlmostEqual(float(np.max(db)), 0.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
