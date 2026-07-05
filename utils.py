import os
import copy

from loguru import logger
from providers import generate_chat_completion, generate_text_with_retries


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_provider(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    provider=None,
):
    if streaming:
        return generate_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            provider=provider,
        )

    return generate_text_with_retries(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
):
    return generate_provider(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
    )


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    response = generate_chat_completion(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        streaming=True,
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    return generate_provider(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="openai",
    )


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=None,
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):
    references = references or []

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
