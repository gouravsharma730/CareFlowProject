"""Tests for the save_patient_preference tool."""

import sqlite3

import pytest

from careflow import config
from careflow.memory_store import MemoryStore
from careflow.tools import execute_tool

PATIENT = "MHP-P-00000"


@pytest.fixture(autouse=True)
def temporary_memory(tmp_path, monkeypatch):
    """Use a throwaway database so the real memory file is never touched."""
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")


def save(arguments, session_id="s1"):
    return execute_tool("save_patient_preference", arguments, session_id=session_id)


def test_preference_is_saved_to_long_term_memory():
    result = save({"patient_ref": PATIENT, "key": "reminder_channel", "value": "SMS in Telugu"})
    assert result["status"] == "saved"
    assert MemoryStore().get_facts(PATIENT) == {"reminder_channel": "SMS in Telugu"}


def test_changed_preference_replaces_the_old_one():
    save({"patient_ref": PATIENT, "key": "language", "value": "Telugu"})
    save({"patient_ref": PATIENT, "key": "language", "value": "Hindi"})
    assert MemoryStore().get_facts(PATIENT) == {"language": "Hindi"}


def test_session_id_comes_from_the_system_not_the_model():
    save(
        {"patient_ref": PATIENT, "key": "language", "value": "Telugu", "session_id": "evil"},
        session_id="real-session",
    )
    with sqlite3.connect(config.MEMORY_DB_PATH) as connection:
        stored = connection.execute("SELECT session_id FROM facts").fetchone()[0]
    assert stored == "real-session"


def test_unknown_key_is_rejected():
    result = save({"patient_ref": PATIENT, "key": "diagnosis", "value": "anything"})
    assert result["error_code"] == "INVALID_ARGUMENT"
    assert MemoryStore().get_facts(PATIENT) == {}


def test_empty_and_overlong_values_are_rejected():
    empty = save({"patient_ref": PATIENT, "key": "language", "value": "  "})
    too_long = save(
        {"patient_ref": PATIENT, "key": "language",
         "value": "x" * (config.MAX_PREFERENCE_VALUE_LENGTH + 1)}
    )
    assert empty["error_code"] == "INVALID_ARGUMENT"
    assert too_long["error_code"] == "INVALID_ARGUMENT"


def test_unknown_patient_is_rejected():
    result = save({"patient_ref": "NOPE", "key": "language", "value": "Telugu"})
    assert result["error_code"] == "PATIENT_NOT_FOUND"