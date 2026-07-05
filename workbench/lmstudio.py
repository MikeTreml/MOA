from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    ok: bool
    stdout: str
    stderr: str


def run_lms(args: list[str], timeout: int = 60) -> CommandResult:
    try:
        result = subprocess.run(
            ["lms", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(result.returncode == 0, result.stdout, result.stderr)
    except Exception as exc:
        return CommandResult(False, "", str(exc))


def list_models() -> list[dict[str, Any]]:
    result = run_lms(["ls", "--json"], timeout=30)
    if not result.ok:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def loaded_models() -> list[dict[str, Any]]:
    result = run_lms(["ps", "--json"], timeout=15)
    if not result.ok:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def server_status(base_url: str = "http://127.0.0.1:1234/v1") -> dict[str, Any]:
    status_result = run_lms(["server", "status"], timeout=10)
    http_ok = False
    models_count = 0
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
            http_ok = True
            models_count = len(payload.get("data", []))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        http_ok = False
    return {
        "running": "running" in status_result.stdout.lower() or http_ok,
        "status": status_result.stdout.strip() or status_result.stderr.strip(),
        "base_url": base_url,
        "http_ok": http_ok,
        "models_count": models_count,
        "loaded_models": loaded_models(),
    }


def load_models(model_keys: list[str], ttl_seconds: int = 3600) -> list[dict[str, Any]]:
    results = []
    for model_key in dict.fromkeys(model_keys):
        result = run_lms(
            ["load", model_key, "--ttl", str(ttl_seconds), "--yes"],
            timeout=180,
        )
        results.append(
            {
                "model": model_key,
                "ok": result.ok,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return results

