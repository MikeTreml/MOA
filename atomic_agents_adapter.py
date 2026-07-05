"""
Placeholder for an Atomic Agents-backed MoA provider.

The rest of this fork can already call LM Studio and OpenAI-compatible
endpoints directly. Use this module if you want an Atomic Agents graph to
own model routing, tool use, memory, or provider-specific agent behavior.
"""


def generate_atomic_completion(*args, **kwargs):
    raise NotImplementedError(
        "Atomic Agents adapter placeholder: install/configure Atomic Agents and "
        "implement generate_atomic_completion() for this workspace."
    )
