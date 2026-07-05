from __future__ import annotations

from collections.abc import Iterable

from .schemas import ModelInfo, ModelPlan


GIB = 1024**3


def _coerce_model(raw) -> ModelInfo | None:
    try:
        model = raw if isinstance(raw, ModelInfo) else ModelInfo.model_validate(raw)
    except Exception:
        return None
    if model.type != "llm" or model.size_bytes <= 0:
        return None
    return model


def _strength_score(model: ModelInfo) -> float:
    size_gb = model.size_bytes / GIB
    context_score = min(model.max_context_length / 131072, 2.5)
    tool_bonus = 3 if model.trained_for_tool_use else 0
    return size_gb + context_score + tool_bonus


def _worker_score(model: ModelInfo) -> tuple[int, float, int]:
    tool = 0 if model.trained_for_tool_use else 1
    context = -model.max_context_length
    return (tool, model.size_bytes / GIB, context)


def _unique_models(models: Iterable[ModelInfo]) -> list[ModelInfo]:
    seen: set[str] = set()
    result: list[ModelInfo] = []
    for model in models:
        if model.model_key in seen:
            continue
        seen.add(model.model_key)
        result.append(model)
    return result


def plan_models(raw_models, memory_cap_gb: float = 80) -> ModelPlan:
    cap_bytes = int(memory_cap_gb * GIB)
    models = _unique_models(
        model for raw in raw_models for model in [_coerce_model(raw)] if model is not None
    )
    if not models:
        return ModelPlan(memory_cap_gb=memory_cap_gb, notes=["No LLM models found."])

    sorted_small = sorted(models, key=lambda model: model.size_bytes)
    smallest_two_size = sum(model.size_bytes for model in sorted_small[:2])
    aggregator_candidates = [
        model for model in models if model.size_bytes + smallest_two_size <= cap_bytes
    ]
    if not aggregator_candidates:
        aggregator_candidates = [model for model in models if model.size_bytes <= cap_bytes]
    if not aggregator_candidates:
        return ModelPlan(
            memory_cap_gb=memory_cap_gb,
            notes=["No model fits within the memory cap."],
        )

    aggregator = max(aggregator_candidates, key=_strength_score)
    evaluator = aggregator
    selected_by_key = {aggregator.model_key: aggregator}
    remaining_bytes = cap_bytes - aggregator.size_bytes

    worker_models: list[ModelInfo] = []
    for model in sorted(models, key=_worker_score):
        if model.model_key in selected_by_key:
            continue
        if model.size_bytes <= remaining_bytes:
            worker_models.append(model)
            selected_by_key[model.model_key] = model
            remaining_bytes -= model.size_bytes
        if len(worker_models) == 3:
            break

    if not worker_models:
        worker_models = [aggregator]

    total = sum(model.size_bytes for model in selected_by_key.values())
    notes = []
    if len(worker_models) < 2:
        notes.append("Only one worker model fits; parallelism will use one model.")

    return ModelPlan(
        memory_cap_gb=memory_cap_gb,
        worker_models=[model.model_key for model in worker_models],
        aggregator_model=aggregator.model_key,
        evaluator_model=evaluator.model_key,
        total_size_bytes=total,
        selected_models=list(selected_by_key.values()),
        notes=notes,
    )

