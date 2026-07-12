from __future__ import annotations

from .schemas import ProviderPreset


def provider_presets() -> list[ProviderPreset]:
    return [
        ProviderPreset(
            id="lmstudio-local",
            name="LM Studio",
            provider="openai-compatible",
            base_url="http://127.0.0.1:1234/v1",
            worker_models=["qwen-3b", "qwen-3b"],
            aggregator_model="qwen-3b",
            evaluator_model="qwen-3b",
            env_keys=["MOA_BASE_URL to point at your server"],
            notes=[
                "LM Studio's OpenAI-compatible server. It can serve several models at once (JIT loading) — what a real mixture needs.",
                "Model names are the ids it reports at /v1/models. Enable parallel requests in LM Studio's server settings or workers will queue.",
            ],
        ),
        ProviderPreset(
            id="llamacpp-local",
            name="llama.cpp",
            provider="openai-compatible",
            base_url="http://127.0.0.1:1235/v1",
            worker_models=["qwen-3b", "qwen-3b"],
            aggregator_model="qwen-3b",
            evaluator_model="qwen-3b",
            env_keys=["MOA_BASE_URL to point at your server"],
            notes=[
                "An OpenAI-compatible llama-server. One model per process — fine for self-ensembles, not multi-model mixtures.",
                "Start it with --parallel N (N ≥ your worker count) or concurrent workers will queue and run serially.",
            ],
        ),
        ProviderPreset(
            id="omp-chatgpt-claude",
            name="OMP ChatGPT + Claude",
            provider="omp",
            base_url="http://127.0.0.1:4141/v1",
            worker_models=["gpt-4o-mini", "claude-3-5-haiku-latest"],
            aggregator_model="gpt-4o",
            evaluator_model="claude-3-5-sonnet-latest",
            env_keys=["OMP_API_KEY", "MOA_BASE_URL or OMP_BASE_URL optional"],
            notes=["Model IDs are placeholders; match them to the names exposed by your OMP proxy."],
        ),
        ProviderPreset(
            id="openai-compatible-proxy",
            name="OpenAI-Compatible Proxy",
            provider="openai-compatible",
            base_url="http://127.0.0.1:8000/v1",
            worker_models=["model-id-from-proxy"],
            aggregator_model="model-id-from-proxy",
            evaluator_model="model-id-from-proxy",
            env_keys=["MOA_API_KEY", "MOA_BASE_URL"],
            notes=["Use this for any non-OMP proxy that implements /v1/chat/completions."],
        ),
        ProviderPreset(
            id="together-api",
            name="Together API",
            provider="together",
            base_url="https://api.together.xyz/v1",
            worker_models=[
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "microsoft/WizardLM-2-8x22B",
            ],
            aggregator_model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            evaluator_model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            env_keys=["TOGETHER_API_KEY or MOA_API_KEY"],
            notes=["Keeps compatibility with the original Together-backed MoA flow."],
        ),
        ProviderPreset(
            id="atomic-placeholder",
            name="Atomic Agents Placeholder",
            provider="atomic",
            base_url="",
            worker_models=["atomic-worker"],
            aggregator_model="atomic-orchestrator",
            evaluator_model="atomic-evaluator",
            env_keys=[],
            notes=["Placeholder only until atomic_agents_adapter.py is wired to an Atomic Agents graph."],
        ),
    ]
