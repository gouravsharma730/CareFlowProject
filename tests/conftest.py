"""Shared test settings: tests never send traces and never touch real telemetry."""

import pytest

from careflow import config, tracing


@pytest.fixture(autouse=True)
def isolated_tracing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_CONFIGURED", False)
    monkeypatch.setattr(config, "TELEMETRY_PATH", tmp_path / "telemetry.jsonl")
    tracing.set_client(None)
    yield
    tracing.set_client(None)