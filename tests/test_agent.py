"""Tests for the agent loop, using a fake LLM (no network calls)."""

import json
from types import SimpleNamespace

from careflow import agent, config


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def reply(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def install_fake_llm(monkeypatch, replies):
    """Return the replies in order (the last one repeats forever)."""
    seen_messages = []

    def fake_chat(messages, tools):
        seen_messages.append([dict(m) for m in messages])
        return replies[min(len(seen_messages) - 1, len(replies) - 1)]

    monkeypatch.setattr(agent, "chat_with_tools", fake_chat)
    return seen_messages


def test_answer_without_tools_ends_after_one_step(monkeypatch):
    install_fake_llm(monkeypatch, [reply("Hello! How can I help?")])
    result = agent.run_agent("hi")
    assert result.stop_reason == agent.STOP_FINAL
    assert result.steps == 1
    assert result.tool_calls == []


def test_tool_is_executed_and_result_goes_back_to_the_model(monkeypatch):
    call = tool_call(
        "check_insurance_eligibility", json.dumps({"patient_ref": "MHP-P-00000"})
    )
    seen = install_fake_llm(monkeypatch, [reply(None, [call]), reply("You are active.")])
    result = agent.run_agent("Is MHP-P-00000 insured?")
    assert result.stop_reason == agent.STOP_FINAL
    assert result.steps == 2
    assert result.tool_calls[0]["tool"] == "check_insurance_eligibility"
    assert result.tool_calls[0]["result"]["success"] is True
    tool_messages = [m for m in seen[1] if m["role"] == "tool"]
    assert len(tool_messages) == 1


def test_loop_stops_at_the_step_limit(monkeypatch):
    call = tool_call("find_appointment_slots", json.dumps({"specialty": "Cardiology"}))
    seen = install_fake_llm(monkeypatch, [reply(None, [call])])  # never answers
    result = agent.run_agent("find slots forever")
    assert result.stop_reason == agent.STOP_MAX_STEPS
    assert result.steps == config.MAX_AGENT_STEPS
    assert len(seen) == config.MAX_AGENT_STEPS
    assert result.final_answer == agent.MAX_STEPS_ANSWER


def test_invalid_json_arguments_become_a_tool_error(monkeypatch):
    call = tool_call("calculate_copay", "{not json")
    install_fake_llm(monkeypatch, [reply(None, [call]), reply("Sorry, could not do that.")])
    result = agent.run_agent("copay please")
    assert result.tool_calls[0]["result"]["error_code"] == "INVALID_ARGUMENT"
    assert result.stop_reason == agent.STOP_FINAL


def test_unknown_tool_does_not_crash_the_agent(monkeypatch):
    call = tool_call("do_everything", "{}")
    install_fake_llm(monkeypatch, [reply(None, [call]), reply("I cannot do that.")])
    result = agent.run_agent("do everything")
    assert result.tool_calls[0]["result"]["error_code"] == "UNKNOWN_TOOL"
    assert result.stop_reason == agent.STOP_FINAL