# Mixture-of-Agents in 50 lines of code
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

provider = os.environ.get("MOA_PROVIDER", "openai-compatible")
user_prompt = os.environ.get("MOA_PROMPT", "What are 3 fun things to do in SF?")
reference_models = get_default_reference_models(provider)
aggregator_model = get_default_model(provider)
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

async def run_llm(model):
    """Run a single LLM call with a reference model."""
    response = None
    for sleep_time in [1, 2, 4]:
        try:
            response = await generate_chat_completion_async(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=512,
                provider=provider,
            )
            break
        except Exception as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return get_completion_text(response) if response else ""

async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])
    finalStream = generate_chat_completion(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])},
            {"role": "user", "content": user_prompt},
        ],
        streaming=True,
        provider=provider,
    )
    for text in stream_text_chunks(finalStream):
        print(text, end="", flush=True)

asyncio.run(main())
