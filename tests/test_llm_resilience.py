"""LLM circuit breaker, fallback model, and safe behaviour when the LLM is down."""

import json
from types import SimpleNamespace

import litellm
import pytest

from careflow import config, llm_client, rag, team, workflow
from careflow.llm_client import LLMUnavailable
from test_team import FakeEHR, ScriptedLoop, intake_for, record


def ok_response(text="ok"):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def busy(**kwargs):
    raise litellm.ServiceUnavailableError(message="busy", llm_provider="gemini", model="m")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)


def test_breaker_opens_after_failed_calls_then_fails_fast(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "completion", lambda **kw: (calls.append(1), busy())[1])
    for _ in range(config.BREAKER_FAILURE_THRESHOLD):
        with pytest.raises(LLMUnavailable):
            llm_client.call_llm([{"role": "user", "content": "hi"}])
    assert llm_client.LLM_BREAKER.state == "open"
    before = len(calls)
    with pytest.raises(LLMUnavailable, match="circuit open"):
        llm_client.call_llm([{"role": "user", "content": "hi"}])
    assert len(calls) == before  # the provider was not called at all


def test_fallback_model_is_off_by_default(monkeypatch):
    assert config.LLM_FALLBACK_MODEL == ""
    monkeypatch.setattr(llm_client, "completion", busy)
    with pytest.raises(LLMUnavailable):
        llm_client.call_llm([{"role": "user", "content": "hi"}])


def test_fallback_model_answers_and_is_flagged(monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "gemini/backup-model")
    models = []

    def main_down_backup_up(**kwargs):
        models.append(kwargs["model"])
        if kwargs["model"] == config.LLM_MODEL:
            busy()
        return ok_response("from backup")

    monkeypatch.setattr(llm_client, "completion", main_down_backup_up)
    before = llm_client.CALL_STATS["fallbacks"]
    assert llm_client.call_llm([{"role": "user", "content": "hi"}]) == "from backup"
    assert models[-1] == "gemini/backup-model"
    assert llm_client.CALL_STATS["fallbacks"] == before + 1
    events = [json.loads(l) for l in config.TELEMETRY_PATH.read_text().splitlines()]
    backup = [e for e in events if e.get("model") == "gemini/backup-model"]
    assert backup and backup[-1]["used_fallback"] is True and backup[-1]["level"] == "WARNING"


def test_permanent_errors_do_not_trip_the_breaker_or_use_the_fallback(monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "gemini/backup-model")

    def bad_key(**kwargs):
        raise litellm.AuthenticationError(message="bad key", llm_provider="gemini", model="m")

    monkeypatch.setattr(llm_client, "completion", bad_key)
    with pytest.raises(litellm.AuthenticationError):
        llm_client.call_llm([{"role": "user", "content": "hi"}])
    assert llm_client.LLM_BREAKER.state == "closed"


def test_policy_answer_says_dont_know_when_the_llm_is_down():
    def down(messages, json_mode=False):
        raise LLMUnavailable("down")

    chunks = [{"chunk_id": "a#001", "doc_title": "T", "section": "S", "text": "x"}]
    result = rag.answer_question("Any question?", retriever=lambda q, k: chunks, llm=down)
    assert result.answer == rag.DONT_KNOW
    assert result.failure_reason == "the language model is unavailable"


@pytest.fixture
def team_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(config, "OPERATIONS_DB_PATH", tmp_path / "ops.sqlite")
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(workflow, "extract_patient_request", intake_for)
    monkeypatch.setattr(team, "open_ehr", FakeEHR)


def test_team_hands_over_to_staff_when_the_agent_llm_is_down(team_env, monkeypatch):
    def down(*args, **kwargs):
        raise LLMUnavailable("down")

    monkeypatch.setattr(team, "run_tool_loop", down)
    graph = team.build_team_graph(workflow.open_checkpointer())
    out = workflow.start_request(graph, "t-down", record("What will an MRI cost me?"))
    assert out["status"] == "done" and out["stop_reason"] == "handoff_llm_unavailable"


def test_team_never_approves_unchecked_when_the_reviewer_llm_is_down(team_env, monkeypatch):
    def down(messages, json_mode=False):
        raise LLMUnavailable("down")

    monkeypatch.setattr(team, "call_llm", down)
    monkeypatch.setattr(team, "run_tool_loop", ScriptedLoop(("Your plan is active.", [])))
    graph = team.build_team_graph(workflow.open_checkpointer())
    out = workflow.start_request(graph, "t-rev-down", record("Is my plan active?"))
    assert out["stop_reason"] == "handoff_llm_unavailable"


def test_workflow_hands_over_when_intake_llm_is_down(team_env, monkeypatch):
    def down(record):
        raise LLMUnavailable("down")

    monkeypatch.setattr(workflow, "extract_patient_request", down)
    graph = workflow.build_graph(workflow.open_checkpointer())
    out = workflow.start_request(graph, "t-intake-down", record("Is my plan active?"))
    assert out["stop_reason"] == "handoff_intake_failed"