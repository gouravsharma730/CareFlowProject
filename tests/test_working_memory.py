"""Tests for working memory (trimmed conversation history)."""

import json
from types import SimpleNamespace

from careflow import agent, config
from careflow.memory_store import MemoryStore
from careflow.working_memory import WorkingMemory

PATIENT = "MHP-P-00000"


def reply(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def install_fake_llm(monkeypatch, replies):
    seen = []

    def fake_chat(messages, tools):
        seen.append([dict(m) for m in messages])
        return replies[min(len(seen) - 1, len(replies) - 1)]

    monkeypatch.setattr(agent, "chat_with_tools", fake_chat)
    return seen


def test_only_the_last_turns_are_kept():
    memory = WorkingMemory(max_turns=3, max_chars=10_000)
    for number in range(10):
        memory.add(f"question {number}", f"answer {number}")
    assert len(memory) == 3
    assert memory.trimmed_turns == 7
    texts = [message["content"] for message in memory.messages()]
    assert texts == [
        "question 7", "answer 7", "question 8", "answer 8", "question 9", "answer 9",
    ]


def test_character_budget_drops_the_oldest_turns_first():
    memory = WorkingMemory(max_turns=100, max_chars=30)
    memory.add("a" * 10, "b" * 10)
    memory.add("c" * 10, "d" * 10)
    memory.add("e" * 5, "f" * 5)
    texts = [message["content"] for message in memory.messages()]
    assert "a" * 10 not in texts
    assert texts[-2:] == ["e" * 5, "f" * 5]


def test_the_latest_turn_is_always_kept():
    memory = WorkingMemory(max_turns=1, max_chars=5)
    memory.add("a very long question", "a very long answer")
    assert len(memory) == 1


def test_messages_alternate_user_and_assistant():
    memory = WorkingMemory(max_turns=4, max_chars=1000)
    memory.add("q1", "a1")
    memory.add("q2", "a2")
    roles = [message["role"] for message in memory.messages()]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_agent_sends_earlier_turns_to_the_model(monkeypatch):
    seen = install_fake_llm(monkeypatch, [reply("first answer"), reply("second answer")])
    history = WorkingMemory()
    agent.run_agent("first question", history=history)
    agent.run_agent("second question", history=history)
    second_call = [message["content"] for message in seen[1]]
    assert "first question" in second_call
    assert "first answer" in second_call


def test_run_without_history_behaves_as_before(monkeypatch):
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    agent.run_agent("hello")
    assert [message["role"] for message in seen[0]] == ["system", "user"]


def test_unfinished_run_is_not_added_to_history(monkeypatch):
    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(
            name="find_appointment_slots",
            arguments=json.dumps({"specialty": "Cardiology"}),
        ),
    )
    install_fake_llm(monkeypatch, [reply(None, [call])])  # never answers
    history = WorkingMemory()
    result = agent.run_agent("find slots", history=history)
    assert result.stop_reason == agent.STOP_MAX_STEPS
    assert len(history) == 0


def test_early_preference_survives_a_long_conversation(monkeypatch, tmp_path):
    """T9: a fact from turn 1 is still available after 16+ turns of filler."""
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")
    MemoryStore().remember_fact(PATIENT, "reminder_channel", "SMS in Telugu")  # turn 1
    seen = install_fake_llm(monkeypatch, [reply("ok")])
    history = WorkingMemory()

    for number in range(16):
        agent.run_agent(f"filler message {number}", patient_ref=PATIENT, history=history)

    last_call = seen[-1]
    contents = " ".join(message["content"] for message in last_call)
    assert len(history) <= config.MAX_HISTORY_TURNS          # budget is enforced
    assert "filler message 0" not in contents                # the early turn was trimmed
    assert "SMS in Telugu" in last_call[0]["content"]        # but the fact is still known