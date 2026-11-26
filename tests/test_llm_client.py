"""Tests for the LLM client's retry behaviour, with a fake provider."""

from types import SimpleNamespace

import litellm
import pytest

from careflow import llm_client


def fake_success():
    message = SimpleNamespace(content="ok")
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Skip the real sleeping so tests run instantly."""
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)


def test_temporary_errors_are_retried_then_succeed(monkeypatch):
    attempts = []

    def flaky_completion(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise litellm.ServiceUnavailableError(
                message="busy", llm_provider="gemini", model="m"
            )
        return fake_success()

    monkeypatch.setattr(llm_client, "completion", flaky_completion)
    assert llm_client.call_llm([{"role": "user", "content": "hi"}]) == "ok"
    assert len(attempts) == 3


def test_permanent_errors_are_not_retried(monkeypatch):
    attempts = []

    def bad_key_completion(**kwargs):
        attempts.append(1)
        raise litellm.AuthenticationError(
            message="bad key", llm_provider="gemini", model="m"
        )

    monkeypatch.setattr(llm_client, "completion", bad_key_completion)
    with pytest.raises(litellm.AuthenticationError):
        llm_client.call_llm([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1


def test_gives_up_after_max_retries(monkeypatch):
    attempts = []

    def always_busy(**kwargs):
        attempts.append(1)
        raise litellm.ServiceUnavailableError(
            message="busy", llm_provider="gemini", model="m"
        )

    monkeypatch.setattr(llm_client, "completion", always_busy)
    with pytest.raises(litellm.ServiceUnavailableError):
        llm_client.call_llm([{"role": "user", "content": "hi"}])
    assert len(attempts) == llm_client.config.LLM_MAX_RETRIES + 1