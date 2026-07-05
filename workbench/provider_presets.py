from __future__ import annotations

from .schemas import ProviderPreset


def provider_presets() -> list[ProviderPreset]:
    return [
        ProviderPreset(
            id="lmstudio-local",
            name="LM Studio Local",
            provider="lmstudio",
            base_url="http://127.0.0.1:1234/v1",
            worker_models=["llama-3.2-1b-instruct", "llama-3.2-1b-instruct"],
            aggregator_model="llama-3.2-1b-instruct",
            evaluator_model="llama-3.2-1b-instruct",
            env_keys=["LM_STUDIO_API_KEY optional"],
            notes=["Uses the local LM Studio OpenAI-compatible server and lms model planner."],
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
