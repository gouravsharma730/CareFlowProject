"""Tests for intake extraction, using a fake LLM (no network calls)."""

import json

import pytest

from careflow import config, intake
from careflow.intake import IntakeError, extract_patient_request

RECORD = {
    "patient_ref": "P-REAL-REF",
    "channel": "sms",
    "raw_text": "need a heart doctor",
    "insurance_id_stated": "PLAN-123",
}

GOOD_REPLY = json.dumps(
    {
        "patient_ref": "WRONG-REF-FROM-LLM",
        "channel": "wrong",
        "raw_text": "wrong text",
        "specialty": "Cardiology",
        "insurance_id": "PLAN-123",
        "urgency": "Routine",
        "seeks_clinical_advice": False,
        "requires_human_review": False,
        "missing_fields": [],
    }
)


def install_fake_llm(monkeypatch, replies):
    """Replace call_llm with a fake that returns the given replies in order."""
    calls = []

    def fake_call_llm(messages, json_mode=False):
        calls.append(list(messages))
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(intake, "call_llm", fake_call_llm)
    return calls


def test_valid_reply_succeeds_in_one_call(monkeypatch):
    calls = install_fake_llm(monkeypatch, [GOOD_REPLY])
    result = extract_patient_request(RECORD)
    assert len(calls) == 1
    assert result.specialty == "Cardiology"


def test_code_copies_identity_fields_not_the_llm(monkeypatch):
    install_fake_llm(monkeypatch, [GOOD_REPLY])
    result = extract_patient_request(RECORD)
    assert result.patient_ref == "P-REAL-REF"
    assert result.channel == "sms"
    assert result.raw_text == "need a heart doctor"


def test_broken_json_is_repaired(monkeypatch):
    calls = install_fake_llm(monkeypatch, ["this is not json", GOOD_REPLY])
    result = extract_patient_request(RECORD)
    assert len(calls) == 2
    assert result.urgency == "Routine"


def test_invalid_value_is_sent_back_for_repair(monkeypatch):
    bad = json.dumps({**json.loads(GOOD_REPLY), "urgency": "Whenever"})
    calls = install_fake_llm(monkeypatch, [bad, GOOD_REPLY])
    extract_patient_request(RECORD)
    repair_message = calls[1][-1]["content"]
    assert "invalid" in repair_message.lower()


def test_repair_loop_is_bounded(monkeypatch):
    calls = install_fake_llm(monkeypatch, ["still not json"])
    with pytest.raises(IntakeError):
        extract_patient_request(RECORD)
    assert len(calls) == config.MAX_REPAIR_ATTEMPTS + 1