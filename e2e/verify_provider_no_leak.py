"""E2E: run the real hybrid workflow against the fake upstream and prove every
async client is created AND closed (the providers.py leak fix), with real HTTP
streaming. Exit 0 on PASS.

    python e2e/verify_provider_no_leak.py

Not part of the unittest suite (needs network + the sandbox's SSL). The verify_
prefix keeps it out of the test_*.py discovery pattern.
"""
from __future__ import annotations

import asyncio
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "e2e"))

import _fake_upstream  # noqa: E402

PORT = 8771


async def run():
    import providers
    from workbench.schemas import Profile, RunRequest
    from workbench.workflow import run_workflow

    created, closed = [], []
    orig = providers._async_client_from_config

    def tracking(config):
        client = orig(config)
        created.append(client)
        real = client.close

        async def wrapped():
            closed.append(client)
            return await real()

        client.close = wrapped
        return client

    providers._async_client_from_config = tracking

    tokens, steps = [], []
    profile = Profile(
        name="Fake", provider="openai-compatible", base_url=f"http://127.0.0.1:{PORT}/v1",
        workflow="hybrid", worker_models=["fa", "fb"], aggregator_model="fg",
        evaluator_model="fe", max_iterations=1,
    )
    record = await run_workflow(
        RunRequest(prompt="Build a thing", profile=profile),
        on_step=lambda s: steps.append(s.stage),
        on_event=lambda k, p: tokens.append(p["text"]) if k == "token" else None,
    )
    return record, created, closed, tokens, steps


def main():
    _fake_upstream.start(PORT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record, created, closed, tokens, steps = asyncio.run(run())
    unclosed = [w for w in caught if "unclosed" in str(w.message).lower()]

    print("stages          :", steps)
    print("streamed tokens :", "".join(tokens)[:60])
    print("final_output    :", repr(record.final_output)[:60])
    print("async clients   : created=%d closed=%d" % (len(created), len(closed)))
    print("unclosed warns  :", len(unclosed))
    ok = (
        len(steps) >= 4 and record.final_output and "".join(tokens)
        and len(created) >= 4 and len(closed) == len(created) and not unclosed
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
