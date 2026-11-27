"""The KPI report aggregates telemetry events correctly."""

from evals.kpi_report import compute

EVENTS = [
    {"thread_id": "t1", "agent": "insurance_agent", "name": "insurance_agent", "type": "agent",
     "duration_ms": 900.0, "level": "DEFAULT"},
    {"thread_id": "t1", "agent": "insurance_agent", "name": "llm.completion", "type": "generation",
     "duration_ms": 700.0, "level": "DEFAULT", "tokens_in": 1000, "tokens_out": 100,
     "cost_usd": 0.00055},
    {"thread_id": "t1", "agent": "insurance_agent", "name": "tool.calculate_copay", "type": "tool",
     "duration_ms": 20.0, "level": "WARNING", "used_fallback": True},
    {"thread_id": "t1", "agent": None, "name": "careflow-team", "type": "chain",
     "duration_ms": 1500.0, "level": "DEFAULT"},
]


def test_kpis_per_agent_and_per_run():
    report = compute(EVENTS)
    agent = report["per_agent"]["insurance_agent"]
    assert agent["llm_calls"] == 1 and agent["tokens_in"] == 1000 and agent["tokens_out"] == 100
    assert agent["cost_usd"] == 0.00055
    assert agent["tool_error_rate"] == 1.0 and agent["fallback_rate"] == 1.0
    run = report["per_run"]["t1"]
    assert run["latency_ms"] == 1500.0 and run["tokens"] == 1100 and run["fallbacks"] == 1
    assert report["overall"]["requests"] == 1 and report["overall"]["tool_fallback_rate"] == 1.0


def test_empty_log_gives_an_empty_report():
    assert compute([])["overall"]["requests"] == 0