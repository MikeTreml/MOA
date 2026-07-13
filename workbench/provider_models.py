"""Provider-agnostic model discovery: read the model list and reachability
straight from an OpenAI-compatible ``/v1/models`` endpoint (LM Studio,
llama.cpp, OpenAI, any compatible server), enriched with whatever extra
metadata the server exposes (LM Studio's ``/api/v0/models``: quantization,
load state, context length; llama.cpp's ``meta.size``: real file size)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any


def _server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def _get_json(url: str, timeout: int = 4) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None


# "llama-3.3-70b" -> 70, "ernie-4.5-21b-a3b" -> 21 (total params, not the MoE
# active count "a3b" — file size follows the total). No match -> no estimate.
_PARAM_COUNT_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")

# Approximate bytes per parameter for common GGUF/MLX quantizations.
_BYTES_PER_PARAM = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8": 1.07, "q6": 0.82, "q5": 0.69, "q4": 0.60,
    "iq4": 0.55, "q3": 0.44, "iq3": 0.42, "q2": 0.35, "iq2": 0.32,
    "mxfp4": 0.56,
}


def estimate_size_bytes(model_id: str, quantization: str | None) -> int:
    """Rough on-disk size from the parameter count in the model id and the
    quantization's bytes-per-weight. LM Studio's REST API reports no size, so
    an honest estimate (rendered with a ~) beats showing nothing."""
    matches = _PARAM_COUNT_RE.findall(model_id.lower())
    if not matches:
        return 0
    params = max(float(m) for m in matches) * 1e9
    quant = (quantization or "").lower()
    bytes_per_param = 0.58  # unknown quant: assume ~4-bit, the local default
    for prefix, value in sorted(_BYTES_PER_PARAM.items(), key=lambda kv: -len(kv[0])):
        if quant.startswith(prefix):
            bytes_per_param = value
            break
    return int(params * bytes_per_param)


def _lmstudio_extras(base_url: str) -> dict[str, dict[str, Any]]:
    """LM Studio's enhanced model list (``/api/v0/models``): quantization,
    loaded/not-loaded state, max context. Empty for any other server."""
    payload = _get_json(f"{_server_root(base_url)}/api/v0/models")
    if not isinstance(payload, dict):
        return {}
    extras: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []) or []:
        if isinstance(item, dict) and item.get("id"):
            extras[str(item["id"])] = item
    return extras


def list_models(base_url: str) -> list[dict[str, Any]]:
    """Fetch the models the active provider actually serves, shaped for the UI.
    Returns [] if the endpoint is unreachable rather than raising."""
    if not base_url:
        return []
    payload = _get_json(f"{base_url.rstrip('/')}/models", timeout=6)
    if not isinstance(payload, dict):
        return []
    extras = _lmstudio_extras(base_url)
    models: list[dict[str, Any]] = []
    for item in payload.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or ""
        if not model_id:
            continue
        entry: dict[str, Any] = {
            "type": "llm",
            "modelKey": model_id,
            "displayName": model_id,
            "maxContextLength": int(item.get("context_length") or 0),
        }
        extra = extras.get(model_id)
        if extra:
            # VLMs chat like LLMs; only embeddings are excluded from role picks.
            entry["type"] = "embedding" if extra.get("type") == "embedding" else "llm"
            if extra.get("quantization"):
                entry["quantization"] = str(extra["quantization"])
            if extra.get("state"):
                entry["state"] = str(extra["state"])
            if extra.get("max_context_length"):
                entry["maxContextLength"] = int(extra["max_context_length"] or 0)
        # Real size when the server reports one (llama.cpp meta.size); else an
        # estimate from the id's parameter count + quantization.
        meta = item.get("meta")
        real_size = int(meta.get("size") or 0) if isinstance(meta, dict) else 0
        if real_size > 0:
            entry["sizeBytes"] = real_size
            entry["sizeIsEstimate"] = False
        else:
            estimate = estimate_size_bytes(model_id, entry.get("quantization"))
            if estimate > 0:
                entry["sizeBytes"] = estimate
                entry["sizeIsEstimate"] = True
        models.append(entry)
    return models


def probe_total_slots(base_url: str) -> int | None:
    """Ask a llama.cpp server how many parallel generation slots it has
    (``/props`` at the server root reports ``total_slots``). Returns None when
    the server isn't llama.cpp or doesn't expose the endpoint — parallelism is
    then simply unknown, not zero."""
    if not base_url:
        return None
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        # A proxy answering unknown routes with 200 + [] / "ok" is not llama.cpp.
        return None
    slots = payload.get("total_slots")
    if isinstance(slots, bool) or not isinstance(slots, (int, float)):
        return None
    return int(slots) if slots > 0 else None


def server_status(base_url: str) -> dict[str, Any]:
    """Probe the provider's ``/v1/models`` for reachability — no CLI, works for
    any OpenAI-compatible server."""
    http_ok = False
    count = 0
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
            http_ok = True
            count = len(payload.get("data", []))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        http_ok = False
    return {
        "running": http_ok,
        "status": "reachable" if http_ok else "unreachable",
        "base_url": base_url,
        "http_ok": http_ok,
        "models_count": count,
        # A one-slot server serializes concurrent workers; surface it so the UI
        # can warn instead of letting "parallel" silently mean "queued".
        "total_slots": probe_total_slots(base_url) if http_ok else None,
    }
