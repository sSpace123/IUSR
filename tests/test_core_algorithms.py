"""Unit tests for the MVP signal-analysis core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.data_loader import DataImportOptions, load_signal_file, preview_signal_file
from core.export import export_signal_csv
from core.feature_extraction import (
    compute_basic_features,
    estimate_delay_by_xcorr,
    estimate_tof_by_envelope,
    hilbert_envelope,
)
from core.filtering import align_by_toneburst_period, bandpass_filter, toneburst_half_period_samples
from core.signal_data import infer_sample_rate_from_time
from core.spectrum_analysis import compute_fft
from core.wavelet_analysis import compute_cwt


class CoreAlgorithmTests(unittest.TestCase):
    def test_infer_sample_rate_from_uniform_time(self) -> None:
        time = np.arange(1000) / 2_000_000.0
        self.assertAlmostEqual(infer_sample_rate_from_time(time), 2_000_000.0)

    def test_infer_sample_rate_rejects_nonuniform_time(self) -> None:
        with self.assertRaises(ValueError):
            infer_sample_rate_from_time([0.0, 1.0, 2.3, 3.0])

    def test_load_csv_with_time_and_multiple_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("time,CH1,CH2\n0,1,2\n0.001,3,4\n0.002,5,6\n", encoding="utf-8")
            signal = load_signal_file(path)
        self.assertEqual(signal.channel_names, ["CH1", "CH2"])
        self.assertAlmostEqual(signal.sample_rate, 1000.0)

    def test_load_csv_without_time_requires_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("CH1,CH2\n1,2\n3,4\n5,6\n", encoding="utf-8")
            signal = load_signal_file(path, sample_rate=2000.0)
        self.assertEqual(signal.time.tolist(), [0.0, 0.0005, 0.001])

    def test_fft_finds_dominant_frequency(self) -> None:
        fs = 1000.0
        time = np.arange(1000) / fs
        values = np.sin(2 * np.pi * 50.0 * time)
        freqs, amplitudes = compute_fft(values, fs)
        dominant = freqs[np.argmax(amplitudes[1:]) + 1]
        self.assertAlmostEqual(dominant, 50.0)

    def test_bandpass_filter_preserves_frequency_inside_band(self) -> None:
        fs = 2000.0
        time = np.arange(2000) / fs
        signal = np.sin(2 * np.pi * 100.0 * time) + 0.6 * np.sin(2 * np.pi * 500.0 * time)
        filtered = bandpass_filter(signal, fs, 80.0, 130.0)
        freqs, amplitudes = compute_fft(filtered, fs)
        dominant = freqs[np.argmax(amplitudes[1:]) + 1]
        self.assertAlmostEqual(dominant, 100.0)

    def test_toneburst_period_alignment_matches_matlab_shift(self) -> None:
        self.assertEqual(toneburst_half_period_samples(5_000_000.0, 100_000.0, 3), 75)
        signal = np.arange(10, dtype=float)
        aligned = align_by_toneburst_period(signal, 10.0, 2.0, 1.0)
        np.testing.assert_array_equal(aligned, [2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    def test_basic_features_include_peak_and_rms(self) -> None:
        fs = 100.0
        time = np.arange(4) / fs
        features = compute_basic_features(time, [0.0, -2.0, 1.0, 0.0], fs)
        self.assertEqual(features["abs_peak"], 2.0)
        self.assertAlmostEqual(features["peak_time"], 0.01)
        self.assertGreater(features["rms"], 0.0)
        self.assertIn("tof", features)
        self.assertIn("envelope_energy", features)
        self.assertIn("average_power", features)

    def test_hilbert_envelope_and_tof(self) -> None:
        time = np.arange(100) / 1000.0
        signal = np.zeros(100)
        signal[30:40] = 1.0
        envelope = hilbert_envelope(signal)
        self.assertEqual(envelope.shape, signal.shape)
        tof = estimate_tof_by_envelope(time, signal)
        self.assertGreaterEqual(tof, 0.0)
        self.assertLess(tof, 0.04)

    def test_estimate_delay_by_xcorr(self) -> None:
        fs = 1000.0
        ref = np.zeros(100)
        target = np.zeros(100)
        ref[20] = 1.0
        target[25] = 1.0
        self.assertAlmostEqual(estimate_delay_by_xcorr(ref, target, fs), 0.005)

    def test_cwt_returns_frequency_by_time_matrix(self) -> None:
        fs = 1000.0
        time = np.arange(128) / fs
        values = np.sin(2 * np.pi * 80.0 * time)
        freqs, coefficients = compute_cwt(values, fs, 20.0, 200.0, 24)
        self.assertEqual(freqs.shape, (24,))
        self.assertEqual(coefficients.shape, (24, 128))

    def test_export_signal_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.csv"
            output = Path(tmp) / "out.csv"
            source.write_text("time,CH1\n0,1\n0.001,2\n", encoding="utf-8")
            signal = load_signal_file(source)
            export_signal_csv(signal, output)
            self.assertIn("CH1", output.read_text(encoding="utf-8"))

    def test_scope_csv_preview_detects_metadata_and_data_table(self) -> None:
        path = Path("testdata") / "SDS5034X_CSV_C1_1.csv"
        if not path.exists():
            self.skipTest("scope test data is not available")
        preview = preview_signal_file(path)
        self.assertEqual(preview.skip_rows, 11)
        self.assertEqual(preview.columns, ["Second", "Value"])
        self.assertIn("Sample Interval", preview.metadata)

    def test_scope_csv_loads_selected_columns(self) -> None:
        path = Path("testdata") / "SDS5034X_CSV_C1_1.csv"
        if not path.exists():
            self.skipTest("scope test data is not available")
        signal = load_signal_file(
            path,
            options=DataImportOptions(
                skip_rows=11,
                has_header=True,
                time_column="Second",
                value_columns=["Value"],
            ),
        )
        self.assertEqual(signal.channel_names, ["Value"])
        self.assertEqual(signal.time.size, 10000)
        self.assertAlmostEqual(signal.sample_rate, 1_000_000.0, places=3)


if __name__ == "__main__":
    unittest.main()
