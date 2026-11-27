"""Tracing tests with a real Langfuse client exporting to memory (no network)."""

import json
from types import SimpleNamespace

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from careflow import config, llm_client, team, tracing, workflow
from test_team import PATIENT, FakeEHR, ScriptedLoop, intake_for, record


@pytest.fixture(scope="module")
def langfuse_in_memory():
    """One Langfuse client for the module: Langfuse keeps one client per key."""
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test",
                      host="http://127.0.0.1:9", span_exporter=exporter)
    return client, exporter


@pytest.fixture
def spans(langfuse_in_memory, tmp_path, monkeypatch):
    client, exporter = langfuse_in_memory
    exporter.clear()
    tracing.set_client(client)
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(config, "OPERATIONS_DB_PATH", tmp_path / "ops.sqlite")
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(workflow, "extract_patient_request", intake_for)
    monkeypatch.setattr(team, "open_ehr", FakeEHR)
    monkeypatch.setattr(team, "call_llm",
                        lambda m, json_mode=False: '{"verdict": "approve", "issues": []}')
    return exporter


def _run_insurance_request(monkeypatch):
    loop = ScriptedLoop(("Your estimated cost is $110.",
                         [("calculate_copay", {"procedure_code": "ORTH-MRI"})]))
    monkeypatch.setattr(team, "run_tool_loop", loop)
    graph = team.build_team_graph(workflow.open_checkpointer())
    return workflow.start_request(graph, "t-trace", record("What will an MRI cost me?"))


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_one_trace_per_request_with_nested_agent_and_tool_spans(spans, monkeypatch):
    out = _run_insurance_request(monkeypatch)
    assert out["stop_reason"] == "answered"
    named = _by_name(spans)
    root, agent, tool = named["careflow-team"], named["insurance_agent"], named["tool.calculate_copay"]
    assert root.parent is None
    assert len({s.context.trace_id for s in spans.get_finished_spans()}) == 1  # one trace
    assert agent.parent.span_id == root.context.span_id
    assert tool.parent.span_id == agent.context.span_id
    assert agent.attributes["langfuse.observation.type"] == "agent"
    assert tool.attributes["langfuse.observation.type"] == "tool"
    for name in ("intake", "supervisor", "safety_review", "finalize"):
        assert name in named


def test_traces_carry_session_and_hashed_user_ids(spans, monkeypatch):
    _run_insurance_request(monkeypatch)
    root = _by_name(spans)["careflow-team"]
    assert root.attributes["session.id"] == "t-trace"
    assert root.attributes["user.id"] == tracing.pseudonym(PATIENT)
    assert root.attributes["user.id"] != PATIENT


def test_no_patient_reference_or_message_is_sent_to_the_tracer(spans, monkeypatch):
    _run_insurance_request(monkeypatch)
    everything = json.dumps([dict(s.attributes) for s in spans.get_finished_spans()])
    assert PATIENT not in everything
    assert "What will an MRI cost me?" not in everything


def test_llm_calls_are_generations_with_model_tokens_and_cost(spans, monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
    response = SimpleNamespace(usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(llm_client, "completion", lambda **kwargs: response)
    llm_client.call_llm([{"role": "user", "content": "hi"}])
    tracing.flush()
    generation = _by_name(spans)["llm.completion"]
    attributes = generation.attributes
    assert attributes["langfuse.observation.type"] == "generation"
    assert attributes["langfuse.observation.model.name"] == config.LLM_MODEL
    assert json.loads(attributes["langfuse.observation.usage_details"])["input"] == 1000
    assert json.loads(attributes["langfuse.observation.cost_details"])["total"] > 0


def test_failed_llm_call_is_marked_as_an_error(spans, monkeypatch):
    def broken(**kwargs):
        raise ValueError("bad request")

    monkeypatch.setattr(llm_client, "completion", broken)
    with pytest.raises(ValueError):
        llm_client.call_llm([{"role": "user", "content": "hi"}])
    tracing.flush()
    assert _by_name(spans)["llm.completion"].attributes["langfuse.observation.level"] == "ERROR"


def test_human_approval_pause_is_not_recorded_as_an_error(spans, monkeypatch):
    graph = team.build_team_graph(workflow.open_checkpointer())
    out = workflow.start_request(graph, "t-pause", record("Should I stop my tablets?"))
    assert out["status"] == "paused"
    approval = _by_name(spans)["human_approval"]
    assert approval.attributes.get("langfuse.observation.level") != "ERROR"


def test_every_span_is_also_written_to_the_local_telemetry_log(spans, monkeypatch):
    _run_insurance_request(monkeypatch)
    events = [json.loads(line) for line in
              config.TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()]
    tool = next(e for e in events if e["name"] == "tool.calculate_copay")
    assert tool["agent"] == "insurance_agent" and tool["thread_id"] == "t-trace"
    assert all(PATIENT not in json.dumps(e) for e in events)


def test_tracing_off_without_keys_still_writes_telemetry(monkeypatch):
    tracing.set_client(None)
    assert not tracing.langfuse_enabled()
    with tracing.span("demo", as_type="tool", tool="x"):
        pass
    assert "demo" in config.TELEMETRY_PATH.read_text(encoding="utf-8")