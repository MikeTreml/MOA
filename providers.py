import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterable

import openai
from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Per-request wall-clock ceiling so a hung server can't stall a call for the
# library default of 10 minutes. max_retries=0 disables the OpenAI client's own
# hidden retries so they don't stack multiplicatively with ours.
DEFAULT_TIMEOUT = float(os.environ.get("MOA_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("MOA_MAX_RETRIES", "6"))

# Exceptions worth retrying (transient) vs. not (a retry can never help).
# Matched by class name so this stays correct across openai library versions.
_RETRYABLE_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "APIError",
}
_NON_RETRYABLE_NAMES = {
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "UnprocessableEntityError",
    "ConflictError",
}


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in _NON_RETRYABLE_NAMES:
        return False
    if isinstance(exc, (ValueError, NotImplementedError, TypeError)):
        return False
    if name in _RETRYABLE_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError))


# Sync clients are safe to reuse (thread-safe httpx pool), so cache them by
# resolved config to avoid paying TLS/pool setup — and leaking a pool — on every
# call. Async clients are NOT cached (they bind to an event loop) and are closed
# by their callers instead.
_sync_client_cache: dict[tuple[str, str | None, str], openai.OpenAI] = {}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str | None
    api_key: str = field(repr=False)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _normalize_provider(provider: str | None) -> str:
    raw = provider or os.environ.get("MOA_PROVIDER") or os.environ.get("MOA_BACKEND")
    normalized = (raw or "lmstudio").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "lmstudio": "lmstudio",
        "lm": "lmstudio",
        "local": "lmstudio",
        "together": "together",
        "togetherai": "together",
        "openai": "openai",
        "openaicompatible": "openai-compatible",
        "compatible": "openai-compatible",
        "custom": "openai-compatible",
        "omp": "omp",
        "atomic": "atomic",
        "atomicagents": "atomic",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MOA_PROVIDER={raw!r}. Use lmstudio, together, openai, omp, "
            "openai-compatible, or atomic."
        ) from exc


def get_provider_config(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    provider_name = _normalize_provider(provider)

    if provider_name == "lmstudio":
        return ProviderConfig(
            provider=provider_name,
            base_url=base_url or _first_env(
                "MOA_BASE_URL",
                "LM_STUDIO_BASE_URL",
                "LMSTUDIO_BASE_URL",
                default=LM_STUDIO_BASE_URL,
            ),
            api_key=api_key or _first_env(
                "MOA_API_KEY",
                "LM_STUDIO_API_KEY",
                "LMSTUDIO_API_KEY",
                default="lm-studio",
            ),
        )

    if provider_name == "together":
        return ProviderConfig(
            provider=provider_name,
            base_url=base_url or _first_env("MOA_BASE_URL", default=TOGETHER_BASE_URL),
            api_key=api_key or _first_env("MOA_API_KEY", "TOGETHER_API_KEY", default=""),
        )

    if provider_name == "openai":
        return ProviderConfig(
            provider=provider_name,
            base_url=base_url or _first_env("MOA_BASE_URL", "OPENAI_BASE_URL"),
            api_key=api_key or _first_env("MOA_API_KEY", "OPENAI_API_KEY", default=""),
        )

    if provider_name in {"omp", "openai-compatible"}:
        resolved_base_url = base_url or _first_env("MOA_BASE_URL", "OMP_BASE_URL", "OPENAI_BASE_URL")
        if not resolved_base_url:
            raise ValueError(
                f"MOA_PROVIDER={provider_name} requires MOA_BASE_URL or OMP_BASE_URL."
            )
        return ProviderConfig(
            provider=provider_name,
            base_url=resolved_base_url,
            api_key=api_key or _first_env(
                "MOA_API_KEY",
                "OMP_API_KEY",
                "OPENAI_API_KEY",
                default="openai-compatible",
            ),
        )

    return ProviderConfig(provider="atomic", base_url=None, api_key="")


def get_default_model(provider: str | None = None) -> str:
    configured_model = os.environ.get("MOA_MODEL")
    if configured_model:
        return configured_model

    provider_name = _normalize_provider(provider)
    if provider_name == "together":
        return "Qwen/Qwen2.5-72B-Instruct-Turbo"
    if provider_name == "openai":
        return "gpt-4o-mini"
    if provider_name == "omp":
        return "gpt-4o-mini"
    return "local-model"


def get_default_reference_models(provider: str | None = None) -> list[str]:
    configured_models = os.environ.get("MOA_REFERENCE_MODELS")
    if configured_models:
        return parse_model_list(configured_models)

    provider_name = _normalize_provider(provider)
    if provider_name == "together":
        return [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/WizardLM-2-8x22B",
        ]

    default_model = get_default_model(provider_name)
    return [default_model, default_model, default_model]


def get_default_image_model(provider: str | None = None) -> str:
    configured = os.environ.get("MOA_IMAGE_MODEL")
    if configured:
        return configured
    provider_name = _normalize_provider(provider)
    if provider_name == "openai":
        return "dall-e-3"
    if provider_name == "together":
        return "black-forest-labs/FLUX.1-schnell-Free"
    return ""  # local / openai-compatible: caller supplies the model id


def parse_model_list(models: str | Iterable[str]) -> list[str]:
    if isinstance(models, str):
        return [model.strip() for model in models.split(",") if model.strip()]
    return [str(model).strip() for model in models if str(model).strip()]


_ATOMIC_MESSAGE = (
    "Atomic Agents provider placeholder: wire your Atomic Agents graph in "
    "atomic_agents_adapter.py, or use MOA_PROVIDER=lmstudio/omp/openai-compatible."
)


def _sync_client_from_config(config: ProviderConfig) -> openai.OpenAI:
    if config.provider == "atomic":
        raise NotImplementedError(_ATOMIC_MESSAGE)
    key = (config.provider, config.base_url, config.api_key)
    client = _sync_client_cache.get(key)
    if client is None:
        client = openai.OpenAI(
            api_key=config.api_key or "empty",
            base_url=config.base_url,
            timeout=DEFAULT_TIMEOUT,
            max_retries=0,
        )
        _sync_client_cache[key] = client
    return client


def _async_client_from_config(config: ProviderConfig) -> openai.AsyncOpenAI:
    if config.provider == "atomic":
        raise NotImplementedError(_ATOMIC_MESSAGE)
    return openai.AsyncOpenAI(
        api_key=config.api_key or "empty",
        base_url=config.base_url,
        timeout=DEFAULT_TIMEOUT,
        max_retries=0,
    )


def create_client(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> openai.OpenAI:
    return _sync_client_from_config(
        get_provider_config(provider, base_url=base_url, api_key=api_key)
    )


def create_async_client(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> openai.AsyncOpenAI:
    return _async_client_from_config(
        get_provider_config(provider, base_url=base_url, api_key=api_key)
    )


def generate_chat_completion(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    provider: str | None = None,
    response_format=None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    config = get_provider_config(provider, base_url=base_url, api_key=api_key)
    client = _sync_client_from_config(config)
    if DEBUG:
        logger.debug(f"Sending {len(messages)} message(s) to {model} via {config.provider}.")
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature > 1e-4 else 0,
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


async def generate_chat_completion_async(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    provider: str | None = None,
    response_format=None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    config = get_provider_config(provider, base_url=base_url, api_key=api_key)
    client = _async_client_from_config(config)
    if DEBUG:
        logger.debug(f"Sending {len(messages)} async message(s) to {model} via {config.provider}.")
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature > 1e-4 else 0,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    try:
        return await client.chat.completions.create(**kwargs)
    finally:
        # Non-streaming: the response is fully materialized before we return, so
        # closing the per-call client here frees its connection pool instead of
        # leaking one on every fan-out call.
        await client.close()


async def stream_chat_completion_async(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Yield content deltas from an OpenAI-compatible streaming completion.

    Fully async so parallel workers and the SSE event loop are never blocked by
    a synchronous stream iterator."""
    config = get_provider_config(provider, base_url=base_url, api_key=api_key)
    client = _async_client_from_config(config)
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if temperature > 1e-4 else 0,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            yield (getattr(delta, "content", None) or "") if delta is not None else ""
    finally:
        # Runs when the stream is exhausted or the consumer stops early
        # (GeneratorExit), so the per-call client is always closed.
        await client.close()


def generate_image(
    prompt,
    model: str | None = None,
    n: int = 1,
    size: str = "1024x1024",
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Generate image(s) via an OpenAI-compatible images endpoint.

    Returns a list of {"b64_json", "url"} dicts (either may be None depending on
    what the server returns). Prefers b64 so the caller can embed the image
    directly; falls back to plain args for servers that reject response_format.
    Raises NotImplementedError for the atomic placeholder, like the chat paths.
    """
    config = get_provider_config(provider, base_url=base_url, api_key=api_key)
    client = _sync_client_from_config(config)  # NotImplementedError for atomic
    chosen = model or get_default_image_model(config.provider)
    kwargs: dict = {"prompt": prompt, "n": n, "size": size}
    if chosen:
        kwargs["model"] = chosen
    try:
        response = client.images.generate(response_format="b64_json", **kwargs)
    except Exception:
        # Some servers reject response_format (or b64) — retry with plain args.
        response = client.images.generate(**kwargs)
    images = []
    for item in getattr(response, "data", None) or []:
        images.append({"b64_json": getattr(item, "b64_json", None), "url": getattr(item, "url", None)})
    return images


def get_completion_text(response) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return (getattr(message, "content", None) or "").strip() if message is not None else ""


def stream_text_chunks(stream):
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        yield (getattr(delta, "content", None) or "") if delta is not None else ""


def generate_text_with_retries(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    provider: str | None = None,
    max_retries: int | None = None,
):
    """Call the model, retrying only *transient* failures with exponential
    backoff + jitter. Non-retryable errors (bad model name, auth, malformed
    request) raise immediately instead of burning ~60s of doomed retries, and
    exhausting all attempts re-raises the last error rather than silently
    returning None for a caller to trip over."""
    attempts = MAX_RETRIES if max_retries is None else max(1, max_retries)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = generate_chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider,
            )
            return get_completion_text(response)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == attempts - 1:
                logger.error(f"Giving up on {model}: {exc}")
                raise
            delay = min(2 ** attempt, 32) + random.uniform(0, 1)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")
            logger.warning(
                f"Transient error from {model} ({exc}); "
                f"retry {attempt + 1}/{attempts} in {delay:.1f}s"
            )
            time.sleep(delay)
    if last_exc is not None:  # pragma: no cover - loop always returns or raises
        raise last_exc
    return ""
