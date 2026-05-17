"""Generate regression outputs for the bundled ultrasonic sample data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_loader import load_signal_file
from core.filtering import extract_narrowband_wave_packet
from core.spectrum_analysis import find_dominant_frequency
from core.toneburst import generate_toneburst_preview
from core.wavelet_analysis import compute_cwt, prepare_signal_for_cwt


OUTPUT_DIR = ROOT / "test_outputs" / "sample_regression"
SAMPLE_PATHS = [
    ROOT / "testdata" / "SDS5034X_CSV_C1_3.csv",
    ROOT / "testdata" / "tek0000ALL.csv",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for path in SAMPLE_PATHS:
        if not path.exists():
            all_results.append({"path": str(path), "error": "file not found"})
            continue
        all_results.append(_process_file(path))

    (OUTPUT_DIR / "parameters.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(all_results)


def _process_file(path: Path) -> dict:
    signal = load_signal_file(path, sample_rate=1_000_000.0)
    stem = _safe_stem(path)
    colors = ["#2563EB", "#F97316", "#10B981", "#EF4444"]
    suggestion = _auto_suggest(signal.sample_rate, signal.time, signal.channels)

    time_png = OUTPUT_DIR / f"{stem}_time_overview.png"
    _plot_time_overview(signal.time, signal.channels, suggestion, time_png, colors)

    first_name, first_values = next(iter(signal.channels.items()))
    packet = extract_narrowband_wave_packet(
        signal.time,
        first_values,
        signal.sample_rate,
        center_freq=suggestion["center_frequency_hz"],
        bandwidth=suggestion["bandwidth_hz"],
        order=4,
        zero_phase=True,
        remove_dc=True,
        auto_locate=True,
        window_length=None,
        window_type="tukey",
        output_mode="segment",
        normalization="max_abs",
    )

    packet_png = OUTPUT_DIR / f"{stem}_narrowband_packet.png"
    _plot_packet(packet, packet_png)

    cwt_min = suggestion["cwt_min_hz"]
    cwt_max = suggestion["cwt_max_hz"]
    cwt_range = (
        packet["peak_time"] - packet["params"]["window_length"] * 2.0,
        packet["peak_time"] + packet["params"]["window_length"] * 2.0,
    )
    prep = prepare_signal_for_cwt(
        signal.time,
        first_values,
        signal.sample_rate,
        time_range=cwt_range,
        max_points=30_000,
    )
    freqs, coeffs = compute_cwt(
        prep["signal"],
        prep["fs"],
        cwt_min,
        cwt_max,
        96,
        wavelet="cmor1.5-1.0",
    )
    cwt_png = OUTPUT_DIR / f"{stem}_wavelet_cwt.png"
    cwt_contrast = _plot_cwt(prep["time"], freqs, coeffs, cwt_png)

    packet_csv = OUTPUT_DIR / f"{stem}_narrowband_packet.csv"
    np.savetxt(
        packet_csv,
        np.column_stack([packet["time"], packet["signal"]]),
        delimiter=",",
        header="time_s,wave_packet",
        comments="",
    )

    return {
        "path": str(path),
        "sample_rate_hz": float(signal.sample_rate),
        "sample_count": int(signal.time.size),
        "time_start_s": float(signal.time[0]),
        "time_end_s": float(signal.time[-1]),
        "channels": list(signal.channels.keys()),
        "metadata": signal.metadata,
        "auto_parameters": suggestion,
        "narrowband": {
            "channel": first_name,
            "lowcut_hz": float(packet["lowcut"]),
            "highcut_hz": float(packet["highcut"]),
            "peak_time_s": float(packet["peak_time"]),
            "window_length_s": float(packet["params"]["window_length"]),
            "packet_samples": int(packet["signal"].size),
            "saturation_ratio_abs_gt_0_9": float(np.mean(np.abs(packet["signal"]) > 0.9)),
            "csv": str(packet_csv),
            "image": str(packet_png),
        },
        "wavelet": {
            "channel": first_name,
            "frequency_min_hz": float(cwt_min),
            "frequency_max_hz": float(cwt_max),
            "frequency_points": 96,
            "input_samples": int(prep["signal"].size),
            "decimation_factor": int(prep["decimation_factor"]),
            "energy_contrast": float(cwt_contrast),
            "image": str(cwt_png),
        },
        "images": {
            "time_overview": str(time_png),
            "narrowband_packet": str(packet_png),
            "wavelet_cwt": str(cwt_png),
        },
    }


def _auto_suggest(sample_rate: float, time: np.ndarray, channels: dict[str, np.ndarray]) -> dict:
    nyquist = sample_rate / 2.0
    min_freq = _frequency_floor(sample_rate, time.size)
    max_freq = nyquist * 0.95
    doms = []
    for values in channels.values():
        dominant = find_dominant_frequency(
            values,
            sample_rate,
            exclude_dc=True,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        if dominant["dominant_hz"] > 0:
            doms.append(float(dominant["dominant_hz"]))

    center = float(np.median(doms)) if doms else nyquist * 0.2
    center = min(max(center, min_freq), nyquist * 0.85)
    bandwidth = min(max(center * 0.45, nyquist * 0.005), nyquist * 0.35)
    if center - bandwidth / 2.0 <= 0:
        bandwidth = center * 1.5
    if center + bandwidth / 2.0 >= nyquist:
        bandwidth = max((nyquist - center) * 1.6, nyquist * 0.01)

    return {
        "dominant_candidates_hz": doms,
        "center_frequency_hz": center,
        "bandwidth_hz": bandwidth,
        "lowcut_hz": center - bandwidth / 2.0,
        "highcut_hz": center + bandwidth / 2.0,
        "cwt_min_hz": max(center - bandwidth, min_freq),
        "cwt_max_hz": min(center + bandwidth * 2.0, nyquist * 0.95),
        "wavelet": "cmor1.5-1.0",
    }


def _plot_time_overview(
    time: np.ndarray,
    channels: dict[str, np.ndarray],
    suggestion: dict,
    path: Path,
    colors: list[str],
) -> None:
    plot_channels = dict(list(channels.items())[:3])
    first_values = next(iter(channels.values()))
    amplitude = max(float(np.nanmax(np.abs(first_values))), 1.0)
    plot_channels["Toneburst"] = generate_toneburst_preview(
        time,
        suggestion["center_frequency_hz"],
        cycles=4,
        amplitude=amplitude,
    )
    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(12, 8), sharex=True)
    if len(plot_channels) == 1:
        axes = [axes]
    x = time * 1e6
    for idx, (ax, (name, values)) in enumerate(zip(axes, plot_channels.items())):
        xx, yy = _downsample(x, values, 20_000)
        ax.plot(xx, yy, color=colors[idx % len(colors)], linewidth=0.8)
        ax.set_title(name)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time / us")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_packet(packet: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(packet["time"] * 1e6, packet["signal"], color="#10B981", linewidth=1.1)
    ax.axvline(packet["peak_time"] * 1e6, color="#EF4444", linewidth=1.0, linestyle="--")
    ax.set_title("Narrowband Wave Packet")
    ax.set_xlabel("Time / us")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cwt(time: np.ndarray, freqs: np.ndarray, coeffs: np.ndarray, path: Path) -> float:
    magnitude = np.abs(coeffs)
    energy = magnitude.sum(axis=0)
    contrast = float(energy.max() / (energy.mean() + 1e-12))
    low, high = np.percentile(magnitude[np.isfinite(magnitude)], [1, 99.5])
    if low == high:
        low, high = float(magnitude.min()), float(magnitude.max() or 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(
        magnitude,
        origin="lower",
        aspect="auto",
        extent=[time[0] * 1e6, time[-1] * 1e6, freqs[0] / 1e3, freqs[-1] / 1e3],
        vmin=low,
        vmax=high,
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Magnitude")
    ax.set_title("CWT Scalogram")
    ax.set_xlabel("Time / us")
    ax.set_ylabel("Frequency / kHz")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return contrast


def _write_summary(results: list[dict]) -> None:
    lines = ["# Sample Regression Summary", ""]
    for result in results:
        lines.append(f"## {Path(result.get('path', 'unknown')).name}")
        if "error" in result:
            lines.append(f"- Error: {result['error']}")
            lines.append("")
            continue
        params = result["auto_parameters"]
        nb = result["narrowband"]
        wt = result["wavelet"]
        lines.extend(
            [
                f"- Sample rate: {result['sample_rate_hz']:.6g} Hz",
                f"- Samples: {result['sample_count']}",
                f"- Channels: {', '.join(result['channels'])}",
                f"- Center frequency: {params['center_frequency_hz']:.6g} Hz",
                f"- Bandwidth: {params['bandwidth_hz']:.6g} Hz",
                f"- Narrowband peak time: {nb['peak_time_s'] * 1e6:.6g} us",
                f"- CWT range: {wt['frequency_min_hz'] / 1e3:.6g} to {wt['frequency_max_hz'] / 1e3:.6g} kHz",
                f"- CWT energy contrast: {wt['energy_contrast']:.3f}",
                f"- Images: {result['images']['time_overview']}, {result['images']['narrowband_packet']}, {result['images']['wavelet_cwt']}",
                "",
            ]
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _frequency_floor(sample_rate: float, sample_count: int) -> float:
    nyquist = sample_rate / 2.0
    bin_width = sample_rate / max(sample_count, 1)
    return min(max(nyquist * 0.01, bin_width * 5.0, 1.0), nyquist * 0.5)


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = int(np.ceil(x.size / max_points))
    return x[::step], y[::step]


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


if __name__ == "__main__":
    main()
