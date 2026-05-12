"""CSV export helpers for signals and analysis results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .signal_data import MultiChannelSignal, SignalData


def export_signal_csv(
    signal: SignalData | MultiChannelSignal,
    path: str | Path,
    channels: list[str] | None = None,
) -> Path:
    """Export signal data to CSV and return the written path."""
    output_path = Path(path)
    if isinstance(signal, SignalData):
        frame = pd.DataFrame({"time": signal.time, signal.name: signal.values})
    else:
        selected = channels or signal.channel_names
        frame_data = {"time": signal.time}
        for channel in selected:
            if channel not in signal.channels:
                raise KeyError(f"Unknown channel: {channel}")
            frame_data[channel] = signal.channels[channel]
        frame = pd.DataFrame(frame_data)
    frame.to_csv(output_path, index=False)
    return output_path


def export_features_csv(features: dict[str, float], path: str | Path) -> Path:
    """Export a feature dictionary as a two-column CSV."""
    output_path = Path(path)
    pd.DataFrame(sorted(features.items()), columns=["feature", "value"]).to_csv(
        output_path, index=False
    )
    return output_path

