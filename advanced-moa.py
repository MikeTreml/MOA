# Advanced Mixture-of-Agents example – 3 layers
import asyncio
import os
from providers import (
    generate_chat_completion,
    generate_chat_completion_async,
    get_completion_text,
    get_default_model,
    get_default_reference_models,
    stream_text_chunks,
)

provider = os.environ.get("MOA_PROVIDER", "lmstudio")
user_prompt = os.environ.get("MOA_PROMPT", "What are 3 fun things to do in SF?")
reference_models = get_default_reference_models(provider)
aggregator_model = get_default_model(provider)
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
# A meaningful MoA needs at least one reference layer plus the aggregator, so 2
# is the floor. Without this clamp MOA_LAYERS=1 silently ran 2 effective layers.
layers = max(int(os.environ.get("MOA_LAYERS", "3")), 2)


def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


async def run_llm(model, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    response = None
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            response = await generate_chat_completion_async(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                provider=provider,
            )
            print("Model: ", model)
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return get_completion_text(response) if response else ""


async def main():
    """Run the main loop of the MOA process."""
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    for _ in range(1, layers - 1):
        results = await asyncio.gather(
            *[run_llm(model, prev_response=results) for model in reference_models]
        )

    finalStream = generate_chat_completion(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        streaming=True,
        provider=provider,
    )
    for text in stream_text_chunks(finalStream):
        print(text, end="", flush=True)


asyncio.run(main())
