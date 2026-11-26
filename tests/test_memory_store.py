"""Tests for the long-term memory store."""

import pytest

from careflow.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.sqlite")


def test_fact_is_saved_and_read_back(store):
    store.remember_fact("P-A", "reminder_channel", "SMS in Telugu", session_id="s1")
    assert store.get_facts("P-A") == {"reminder_channel": "SMS in Telugu"}


def test_changed_fact_replaces_the_old_value(store):
    store.remember_fact("P-A", "reminder_channel", "SMS in Telugu")
    store.remember_fact("P-A", "reminder_channel", "email in English")
    assert store.get_facts("P-A") == {"reminder_channel": "email in English"}


def test_events_come_back_newest_first_with_session_and_time(store):
    store.add_event("P-A", "referral_created", "Cardiology referral", session_id="s1")
    store.add_event("P-A", "appointment_booked", "Booked slot", session_id="s1")
    events = store.recent_events("P-A")
    assert [event["event_type"] for event in events] == [
        "appointment_booked",
        "referral_created",
    ]
    assert events[0]["session_id"] == "s1"
    assert events[0]["occurred_at"]


def test_memory_survives_a_restart(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = MemoryStore(path)
    first.remember_fact("P-A", "language", "Telugu")
    first.add_event("P-A", "referral_created", "Cardiology referral")
    del first  # the "process" ends here

    second = MemoryStore(path)  # a brand-new store on the same file
    assert second.get_facts("P-A") == {"language": "Telugu"}
    assert second.recent_events("P-A")[0]["summary"] == "Cardiology referral"


def test_patients_never_see_each_others_memory(store):
    store.remember_fact("P-A", "language", "Telugu")
    store.add_event("P-A", "referral_created", "Cardiology referral")
    assert store.get_facts("P-B") == {}
    assert store.recent_events("P-B") == []


def test_procedure_is_saved_updated_and_versioned(store):
    assert store.get_procedure("mri_booking") is None
    assert store.save_procedure("mri_booking", ["check plan", "request approval"]) == 1
    assert store.save_procedure("mri_booking", ["check plan", "request approval", "book"]) == 2
    procedure = store.get_procedure("mri_booking")
    assert procedure["version"] == 2
    assert procedure["steps"] == ["check plan", "request approval", "book"]


def test_delete_patient_data_removes_only_that_patient(store):
    store.remember_fact("P-A", "language", "Telugu")
    store.remember_fact("P-B", "language", "Hindi")
    store.add_event("P-A", "referral_created", "Cardiology referral")
    store.delete_patient_data("P-A")
    assert store.get_facts("P-A") == {}
    assert store.recent_events("P-A") == []
    assert store.get_facts("P-B") == {"language": "Hindi"}