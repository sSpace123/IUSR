"""Optional OpenAI-compatible API helper for parameter suggestions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import FilePreview
from .spectrum_analysis import compute_fft


@dataclass
class AIConfig:
    """Connection settings for an OpenAI-compatible chat completions API."""

    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


def load_local_ai_config(enabled: bool = False) -> AIConfig:
    """Load optional API settings from environment variables or a local .env file."""
    env = _read_dotenv(Path.cwd() / ".env")
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or env.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or env.get("OPENAI_API_KEY")
        or ""
    )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or env.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model = (
        os.environ.get("DEEPSEEK_MODEL")
        or env.get("DEEPSEEK_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or env.get("OPENAI_MODEL")
        or "deepseek-chat"
    )
    return AIConfig(enabled=enabled, api_key=api_key, base_url=base_url, model=model)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lstrip("\ufeff")] = value.strip().strip('"').strip("'")
    return values


def suggest_import_options(preview: FilePreview, config: AIConfig) -> dict[str, Any]:
    """Ask the model to suggest rows and columns from a file preview."""
    if not config.enabled:
        raise ValueError("AI assistant is not enabled.")
    payload = {
        "file_name": preview.path.name,
        "file_size": preview.file_size,
        "total_lines": preview.total_lines,
        "detected_skip_rows": preview.skip_rows,
        "detected_has_header": preview.has_header,
        "columns": preview.columns,
        "metadata": preview.metadata,
        "preview_rows": preview.preview_rows[:8],
    }
    prompt = (
        "You are helping import oscilloscope or ultrasonic signal data. "
        "Return strict JSON only with keys: skip_rows, has_header, time_column, "
        "value_columns, sample_rate_hz, reason. Choose signal columns only."
    )
    return _chat_json(config, prompt, payload)


def suggest_analysis_parameters(
    time: np.ndarray,
    channels: dict[str, np.ndarray],
    sample_rate: float,
    config: AIConfig,
) -> dict[str, Any]:
    """Suggest narrow-band and CWT parameters from signal metadata and spectrum."""
    if not config.enabled:
        raise ValueError("AI assistant is not enabled.")
    spectra = {}
    for name, values in list(channels.items())[:3]:
        freqs, amplitudes = compute_fft(values, sample_rate)
        if freqs.size > 1:
            peak_index = int(np.argmax(amplitudes[1:]) + 1)
            spectra[name] = {
                "dominant_frequency_hz": float(freqs[peak_index]),
                "peak_amplitude": float(amplitudes[peak_index]),
            }
    payload = {
        "sample_rate_hz": sample_rate,
        "nyquist_hz": sample_rate / 2.0,
        "sample_count": int(time.size),
        "duration_s": float(time[-1] - time[0]) if time.size > 1 else 0.0,
        "channel_count": len(channels),
        "spectra": spectra,
    }
    prompt = (
        "You are configuring ultrasonic signal analysis. Return strict JSON only "
        "with keys: center_frequency_hz, bandwidth_hz, filter_cycles, wavelet, "
        "cwt_min_hz, cwt_max_hz, cwt_points, reason. Keep frequencies below Nyquist."
    )
    return _chat_json(config, prompt, payload)


def _chat_json(config: AIConfig, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = config.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"AI API request failed: {exc}") from exc

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)
