"""Shared test settings: no traces sent, no real telemetry, fresh circuit breakers."""

import pytest

from careflow import config, llm_client, reliability, tracing


@pytest.fixture(autouse=True)
def isolated_tracing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_CONFIGURED", False)
    monkeypatch.setattr(config, "TELEMETRY_PATH", tmp_path / "telemetry.jsonl")
    tracing.set_client(None)
    llm_client.LLM_BREAKER.reset()
    reliability.EHR_BREAKER.reset()
    yield
    tracing.set_client(None)
    llm_client.LLM_BREAKER.reset()
    reliability.EHR_BREAKER.reset()