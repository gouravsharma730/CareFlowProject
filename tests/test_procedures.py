"""Tests for procedural memory (stored, versioned, matched by code)."""

import json
from types import SimpleNamespace

from careflow import agent, config, procedures
from careflow.memory_store import MemoryStore
from careflow.procedures import ProcedureDefinition, find_procedure, format_procedure


def use_temporary_memory(monkeypatch, tmp_path):
    path = tmp_path / "memory.sqlite"
    monkeypatch.setattr(config, "MEMORY_DB_PATH", path)
    return path


def reply(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def install_fake_llm(monkeypatch, replies):
    seen = []

    def fake_chat(messages, tools):
        seen.append([dict(m) for m in messages])
        return replies[min(len(seen) - 1, len(replies) - 1)]

    monkeypatch.setattr(agent, "chat_with_tools", fake_chat)
    return seen


def test_unrelated_request_matches_nothing_and_opens_no_database(monkeypatch, tmp_path):
    path = use_temporary_memory(monkeypatch, tmp_path)
    assert find_procedure("What is my copay for a cardiology visit?") is None
    assert not path.exists()


def test_mri_request_matches_and_the_procedure_is_stored(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    found = find_procedure("I need to book an MRI for my knee")
    assert found["name"] == "mri_booking_preauth"
    assert found["version"] == 1
    assert len(found["steps"]) == 6
    assert MemoryStore().get_procedure("mri_booking_preauth")["version"] == 1


def test_matching_again_does_not_create_a_new_version(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    find_procedure("book an MRI")
    assert find_procedure("another MRI please")["version"] == 1


def test_changed_definition_is_saved_as_a_new_version(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    find_procedure("book an MRI")
    changed = ProcedureDefinition(
        name="mri_booking_preauth",
        patterns=(r"\bmri\b",),
        steps=("Only one step now.",),
        source="test",
    )
    monkeypatch.setattr(procedures, "DEFINITIONS", (changed,))
    found = find_procedure("book an MRI")
    assert found["version"] == 2
    assert found["steps"] == ["Only one step now."]


def test_steps_are_numbered_in_order():
    text = format_procedure({"name": "demo", "version": 3, "steps": ["first", "second"]})
    assert "demo v3" in text
    assert text.index("1. first") < text.index("2. second")


def test_agent_receives_the_matching_procedure(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    result = agent.run_agent("I need to book an MRI for my knee")
    assert "physical therapy" in seen[0][0]["content"]
    assert result.procedure == "mri_booking_preauth v1"


def test_agent_gets_no_procedure_for_other_requests(monkeypatch, tmp_path):
    use_temporary_memory(monkeypatch, tmp_path)
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    result = agent.run_agent("What is my copay for a cardiology visit?")
    assert "physical therapy" not in seen[0][0]["content"]
    assert result.procedure is None