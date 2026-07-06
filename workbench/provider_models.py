"""Provider-agnostic model discovery: read the model list and reachability
straight from an OpenAI-compatible ``/v1/models`` endpoint (llama.cpp, OpenAI,
any compatible server). Replaces the old LM Studio ``lms`` CLI coupling."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def list_models(base_url: str) -> list[dict[str, Any]]:
    """Fetch the models the active provider actually serves, shaped for the UI.
    Returns [] if the endpoint is unreachable rather than raising."""
    if not base_url:
        return []
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return []
    models: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        model_id = item.get("id") or ""
        if not model_id:
            continue
        models.append(
            {
                "type": "llm",
                "modelKey": model_id,
                "displayName": model_id,
                "sizeBytes": 0,
                "maxContextLength": int(item.get("context_length") or 0),
            }
        )
    return models


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
        "loaded_models": [],
    }
