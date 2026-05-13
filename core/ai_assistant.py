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


class AIRequestError(RuntimeError):
    """User-facing AI request failure."""


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
    last_error: Exception | None = None
    for url in _candidate_chat_urls(config.base_url):
        for use_response_format in (True, False):
            try:
                data = _post_chat_json(config, url, system_prompt, payload, use_response_format)
                content = data["choices"][0]["message"]["content"]
                return _parse_json_content(content)
            except urllib.error.HTTPError as exc:
                last_error = exc
                # Some OpenAI-compatible services reject response_format; retry once without it.
                if exc.code in {400, 422} and use_response_format:
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
            except (KeyError, json.JSONDecodeError) as exc:
                raise AIRequestError("AI 返回内容不是可解析的 JSON，请重试或关闭 API 使用本地建议。") from exc

    raise AIRequestError(_friendly_ai_error(last_error)) from last_error


def _candidate_chat_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    bases = [base]
    if base.endswith("/v1"):
        bases.append(base[:-3])
    return [f"{candidate}/chat/completions" for candidate in dict.fromkeys(bases)]


def _post_chat_json(
    config: AIConfig,
    url: str,
    system_prompt: str,
    payload: dict[str, Any],
    use_response_format: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }
    if use_response_format:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _friendly_ai_error(error: Exception | None) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"AI API 请求失败：HTTP {error.code}，请检查 API Base、模型名和 Key。"
    if isinstance(error, urllib.error.URLError):
        reason = str(error.reason)
        if "SSL" in reason or "EOF" in reason:
            return "AI API 连接被网络或代理中断，已无法完成本次在线识别。可稍后重试，或关闭 API 使用本地自动建议。"
        return f"AI API 网络连接失败：{reason}"
    if error is None:
        return "AI API 请求失败，请检查网络、代理和 API 配置。"
    return f"AI API 请求失败：{error}"
