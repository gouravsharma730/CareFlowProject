"""Tests for the MCP server, through a real MCP client and server process.

No LLM and no network: the server is started as a child process and
spoken to over stdio, exactly as the agents use it.
"""

import json

import pytest

from careflow.mcp_client import CareFlowMCP
from careflow.tools import get_referral_status

EXPECTED_TOOLS = {
    "check_insurance_eligibility", "check_preauthorization", "calculate_copay",
    "find_appointment_slots", "get_referral_status", "flag_for_human",
}


@pytest.fixture(scope="module")
def ehr():
    with CareFlowMCP() as client:
        yield client


def test_server_exposes_the_ehr_tools(ehr):
    assert {tool.name for tool in ehr.list_tools()} == EXPECTED_TOOLS


def test_tool_specs_are_typed_and_described(ehr):
    for schema in ehr.tool_schemas():
        function = schema["function"]
        assert len(function["description"]) > 60, function["name"]
        assert function["parameters"]["type"] == "object"
    eligibility = next(s for s in ehr.tool_schemas()
                       if s["function"]["name"] == "check_insurance_eligibility")
    patient = eligibility["function"]["parameters"]["properties"]["patient_ref"]
    assert patient["pattern"] == r"^MHP-P-\d{5}$"


def test_tool_schemas_can_be_limited_to_one_agents_tools(ehr):
    names = {s["function"]["name"] for s in ehr.tool_schemas({"get_referral_status"})}
    assert names == {"get_referral_status"}


def test_tool_call_returns_structured_data(ehr):
    result = ehr.call_tool("check_insurance_eligibility", {"patient_ref": "MHP-P-00000"})
    assert result["success"] is True
    assert result["plan_code"] == "MERIDIAN-GOLD"


def test_invalid_arguments_come_back_as_errors_not_crashes(ehr):
    result = ehr.call_tool("check_insurance_eligibility", {"patient_ref": "../../etc/passwd"})
    assert result["success"] is False and result["error_code"] == "INVALID_ARGUMENT"
    unknown = ehr.call_tool("delete_all_records", {})
    assert unknown["success"] is False


def test_business_errors_pass_through(ehr):
    result = ehr.call_tool("get_referral_status",
                           {"patient_ref": "MHP-P-00353", "referral_id": "MHP-R-99999"})
    assert result == {"success": False, "error_code": "REFERRAL_NOT_FOUND",
                      "message": "No referral MHP-R-99999 for patient MHP-P-00353."}


def test_policy_resources_can_be_read(ehr):
    index = json.loads(ehr.read_resource("policy://index"))
    assert {entry["slug"] for entry in index} >= {"clinical-escalation", "preauth-matrix"}
    sop = ehr.read_resource("policy://clinical-escalation")
    assert "Exact Deflection Script" in sop
    bands = json.loads(ehr.read_resource("careflow://urgency-bands"))
    assert {band["urgency"] for band in bands} >= {"Routine", "Urgent"}


def test_unknown_policy_slug_is_refused(ehr):
    with pytest.raises(Exception):
        ehr.read_resource("policy://..%2F..%2F.env")


def test_prompt_template_is_served(ehr):
    text = ehr.get_prompt("clinical_deflection", {"patient_message": "Should I stop my pills?"})
    assert "not permitted to provide medical advice" in text
    assert "Should I stop my pills?" in text


# ---- the new referral tool itself (no MCP needed) ----

def test_referral_status_lists_a_patients_referrals():
    result = get_referral_status("MHP-P-00353")
    assert result["success"] and result["referrals"][0]["referral_id"] == "MHP-R-00000"
    assert result["referrals"][0]["sla_due_on"] == "2026-05-18"


def test_referral_status_errors():
    assert get_referral_status("MHP-P-99999")["error_code"] == "PATIENT_NOT_FOUND"
    assert get_referral_status("MHP-P-00000")["referrals"] == []