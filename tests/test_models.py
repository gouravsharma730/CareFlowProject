"""Tests for the typed PatientRequest model."""

import pytest
from pydantic import ValidationError

from careflow.models import PatientRequest

VALID = {
    "patient_ref": "P-1",
    "channel": "sms",
    "raw_text": "need a heart doctor",
    "urgency": "Routine",
    "seeks_clinical_advice": False,
    "requires_human_review": False,
}


def test_valid_request_builds_with_defaults():
    request = PatientRequest.model_validate(VALID)
    assert request.specialty is None
    assert request.insurance_id is None
    assert request.missing_fields == []


def test_invalid_urgency_is_rejected():
    with pytest.raises(ValidationError):
        PatientRequest.model_validate({**VALID, "urgency": "Whenever"})


def test_unknown_specialty_is_rejected():
    with pytest.raises(ValidationError):
        PatientRequest.model_validate({**VALID, "specialty": "Astrology"})


def test_required_boolean_cannot_be_missing():
    data = dict(VALID)
    del data["seeks_clinical_advice"]
    with pytest.raises(ValidationError):
        PatientRequest.model_validate(data)