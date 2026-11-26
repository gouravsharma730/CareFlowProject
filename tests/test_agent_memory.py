"""Tests for how the agent uses long-term memory (fake LLM, temporary database)."""

import json
from types import SimpleNamespace

from careflow import agent, config
from careflow.memory_store import MemoryStore

PATIENT = "MHP-P-00000"


def use_temporary_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")


def tool_call(name, arguments):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def reply(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def install_fake_llm(monkeypatch, replies):
    seen = []

    def fake_chat(messages, tools):
        seen.append([dict(m) for m in messages])
        return replies[min(len(seen) - 1, len(replies) - 1)]

    monkeypatch.setattr(agent, "chat_with_tools", fake_chat)
    return seen


def test_saved_preferences_are_loaded_into_the_prompt(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    MemoryStore().remember_fact(PATIENT, "reminder_channel", "SMS in Telugu")
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    agent.run_agent("Remind me about my visit", patient_ref=PATIENT, session_id="s2")
    system_prompt = seen[0][0]["content"]
    assert "SMS in Telugu" in system_prompt
    assert f"Patient reference: {PATIENT}" in seen[0][1]["content"]


def test_another_patients_memory_is_not_loaded(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    MemoryStore().remember_fact(PATIENT, "reminder_channel", "SMS in Telugu")
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    agent.run_agent("Hello", patient_ref="MHP-P-00001", session_id="s9")
    assert "SMS in Telugu" not in seen[0][0]["content"]


def test_no_patient_means_no_memory_in_the_prompt(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    MemoryStore().remember_fact(PATIENT, "reminder_channel", "SMS in Telugu")
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    agent.run_agent("Hello")
    assert "SMS in Telugu" not in seen[0][0]["content"]


def test_human_review_is_recorded_as_an_event_without_clinical_detail(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    call = tool_call(
        "flag_for_human",
        {"patient_ref": PATIENT, "reason": "asks about doubling beta blocker dose",
         "urgency": "Urgent"},
    )
    install_fake_llm(monkeypatch, [reply(None, [call]), reply("A staff member will follow up.")])
    agent.run_agent("Should I double my dose?", patient_ref=PATIENT, session_id="s1")
    events = MemoryStore().recent_events(PATIENT)
    assert events[0]["event_type"] == "human_review_requested"
    assert events[0]["session_id"] == "s1"
    assert "beta blocker" not in events[0]["summary"]


def test_failed_tool_call_records_no_event(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    call = tool_call(
        "flag_for_human", {"patient_ref": PATIENT, "reason": "x", "urgency": "ASAP"}
    )
    install_fake_llm(monkeypatch, [reply(None, [call]), reply("Sorry.")])
    agent.run_agent("help", patient_ref=PATIENT, session_id="s1")
    assert MemoryStore().recent_events(PATIENT) == []


def test_preference_saved_through_the_agent_carries_the_session_id(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    call = tool_call(
        "save_patient_preference",
        {"patient_ref": PATIENT, "key": "language", "value": "Telugu"},
    )
    install_fake_llm(monkeypatch, [reply(None, [call]), reply("Noted.")])
    agent.run_agent("Please use Telugu", patient_ref=PATIENT, session_id="s1")
    assert MemoryStore().get_facts(PATIENT) == {"language": "Telugu"}