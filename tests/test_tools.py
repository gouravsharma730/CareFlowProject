"""Tests for the CareFlow tools, using the committed mock tables."""

import json

from careflow import config
from careflow.tools import TOOLS, execute_tool, tool_schemas

PATIENT = "MHP-P-00000"


def test_eligibility_returns_plan_details():
    result = execute_tool("check_insurance_eligibility", {"patient_ref": PATIENT})
    assert result["success"] is True
    assert result["plan_code"] == "MERIDIAN-GOLD"
    assert result["specialist_copay_usd"] == 25.0


def test_unknown_patient_returns_structured_error():
    result = execute_tool("check_insurance_eligibility", {"patient_ref": "NOPE"})
    assert result["success"] is False
    assert result["error_code"] == "PATIENT_NOT_FOUND"


def test_slot_limit_is_capped_in_code():
    result = execute_tool("find_appointment_slots", {"specialty": "Cardiology", "limit": 999})
    assert result["success"] is True
    assert 0 < len(result["slots"]) <= config.MAX_SLOTS_RETURNED


def test_slots_are_earliest_first():
    result = execute_tool("find_appointment_slots", {"specialty": "Cardiology", "limit": 5})
    times = [slot["starts_at"] for slot in result["slots"]]
    assert times == sorted(times)


def test_invalid_mode_is_rejected():
    result = execute_tool("find_appointment_slots", {"specialty": "Cardiology", "mode": "x"})
    assert result["error_code"] == "INVALID_ARGUMENT"


def test_unknown_specialty_is_rejected():
    result = execute_tool("find_appointment_slots", {"specialty": "Astrology"})
    assert result["error_code"] == "UNKNOWN_SPECIALTY"


def test_copay_for_consultation():
    result = execute_tool(
        "calculate_copay", {"patient_ref": PATIENT, "procedure_code": "CARD-CONS"}
    )
    assert result["success"] is True
    assert result["estimated_out_of_pocket_usd"] == 25.0


def test_preauthorization_rule_is_read():
    result = execute_tool(
        "check_preauthorization", {"patient_ref": PATIENT, "procedure_code": "CARD-CONS"}
    )
    assert result["success"] is True
    assert result["covered"] is True
    assert result["preauth_required"] is False


def test_flag_for_human_rejects_bad_urgency_and_empty_reason():
    bad_urgency = execute_tool(
        "flag_for_human", {"patient_ref": PATIENT, "reason": "dose question", "urgency": "ASAP"}
    )
    assert bad_urgency["error_code"] == "INVALID_ARGUMENT"
    no_reason = execute_tool("flag_for_human", {"patient_ref": PATIENT, "reason": "  "})
    assert no_reason["error_code"] == "INVALID_ARGUMENT"


def test_flag_for_human_succeeds():
    result = execute_tool(
        "flag_for_human",
        {"patient_ref": PATIENT, "reason": "asks about dose", "urgency": "Urgent"},
    )
    assert result["status"] == "flagged_for_human_review"


def test_unknown_tool_and_bad_arguments_never_crash():
    assert execute_tool("do_everything", {})["error_code"] == "UNKNOWN_TOOL"
    missing = execute_tool("calculate_copay", {"patient_ref": PATIENT})
    assert missing["error_code"] == "INVALID_ARGUMENT"
    extra = execute_tool(
        "calculate_copay", {"patient_ref": PATIENT, "procedure_code": "X", "zzz": 1}
    )
    assert extra["error_code"] == "INVALID_ARGUMENT"


def test_every_tool_has_a_readable_spec():
    schemas = tool_schemas()
    assert len(schemas) == len(TOOLS) == 5
    for schema in schemas:
        function = schema["function"]
        assert "_" in function["name"]  # verb_noun style
        assert len(function["description"]) > 80  # says what and when to use it
        assert function["parameters"]["required"]
        json.dumps(schema)  # must be JSON-safe