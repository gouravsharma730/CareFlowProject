"""Tests for the agent team (offline: fake MCP server, fake LLM calls)."""

import json

import pytest

from careflow import config, team, workflow
from careflow.agent_loop import LoopResult
from careflow.escalations import EscalationQueue
from careflow.models import PatientRequest

PATIENT = "MHP-P-00353"
SOP = "Clinical Escalation SOP: never give medical advice. Exact Deflection Script ..."
BANDS = json.dumps([{"urgency": "Routine", "referral_sla_business_days": "5"}])


class FakeEHR:
    """Stands in for the MCP server. Records every call and resource read."""

    log: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tool_schemas(self, allowed):
        FakeEHR.log.append(("schemas", sorted(allowed)))
        return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in allowed]

    def call_tool(self, name, arguments):
        FakeEHR.log.append(("call", name, arguments))
        if name == "get_referral_status":
            return {"success": True, "referrals": [
                {"referral_id": "MHP-R-00000", "status": "accepted", "sla_due_on": "2026-05-18"}]}
        if name == "calculate_copay":
            return {"success": True, "estimated_out_of_pocket_usd": 110.0}
        return {"success": True}

    def read_resource(self, uri):
        FakeEHR.log.append(("resource", uri))
        return SOP if uri == "policy://clinical-escalation" else BANDS


class ScriptedLoop:
    """Fake specialist: each call returns the next draft and runs the given tool calls."""

    def __init__(self, *turns):
        self.turns = list(turns)  # each turn: (draft, [(tool, args), ...])
        self.users = []

    def __call__(self, system, user, schemas, execute, max_steps):
        self.users.append(user)
        draft, calls = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        made = [{"tool": t, "arguments": a, "result": execute(t, a)} for t, a in calls]
        return LoopResult(draft, "final_answer", 1, made)


def intake_for(record):
    text = record["raw_text"].lower()
    clinical = "should i" in text
    return PatientRequest(
        patient_ref=record["patient_ref"], channel=record["channel"], raw_text=record["raw_text"],
        urgency="Emergent" if "can't breathe" in text else "Routine",
        seeks_clinical_advice=clinical, requires_human_review=clinical,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(config, "OPERATIONS_DB_PATH", tmp_path / "ops.sqlite")
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(workflow, "extract_patient_request", intake_for)
    monkeypatch.setattr(team, "open_ehr", FakeEHR)
    FakeEHR.log = []
    reviews = []

    def reviewer(messages, json_mode=False):
        reviews.append(messages)
        return json.dumps({"verdict": "approve", "issues": []})

    monkeypatch.setattr(team, "call_llm", reviewer)
    return reviews


def graph():
    return team.build_team_graph(workflow.open_checkpointer())


def record(text):
    return {"patient_ref": PATIENT, "channel": "sms", "raw_text": text,
            "insurance_id_stated": None}


# ---- supervisor ----

def test_supervisor_routes_clear_cases_by_rule():
    assert team.choose_agent("What will my MRI cost?") == ("insurance", "rule")
    assert team.choose_agent("Where is my referral?") == ("referral", "rule")


def test_supervisor_llm_choice_is_validated_against_legal_routes():
    assert team.choose_agent("hello there", llm=lambda m, json_mode: '{"agent": "referral"}') \
        == ("referral", "llm")
    assert team.choose_agent("hello there", llm=lambda m, json_mode: '{"agent": "billing_bot"}') \
        == ("insurance", "default")
    assert team.choose_agent("hello there", llm=lambda m, json_mode: "not json") \
        == ("insurance", "default")


# ---- boundaries ----

def test_agents_can_only_use_their_own_tools():
    execute = team.make_executor("referral", PATIENT, FakeEHR())
    refused = execute("calculate_copay", {"patient_ref": PATIENT, "procedure_code": "ORTH-MRI"})
    assert refused["error_code"] == "TOOL_NOT_ALLOWED"
    assert team.ALLOWED_TOOLS["insurance"].isdisjoint(team.ALLOWED_TOOLS["referral"])


def test_agents_cannot_read_another_patients_record():
    FakeEHR.log = []
    execute = team.make_executor("referral", PATIENT, FakeEHR())
    execute("get_referral_status", {"patient_ref": "MHP-P-00001"})
    assert FakeEHR.log[-1] == ("call", "get_referral_status", {"patient_ref": PATIENT})


def test_nodes_cannot_write_keys_they_do_not_own():
    sneaky = team.bounded("referral_agent", lambda state: {"final_answer": "hacked"})
    with pytest.raises(RuntimeError, match="does not own"):
        sneaky({})


# ---- team runs ----

def test_insurance_question_is_answered_after_review(env, monkeypatch):
    loop = ScriptedLoop(("Your estimated cost is $110.", [("calculate_copay",
                         {"procedure_code": "ORTH-MRI"})]))
    monkeypatch.setattr(team, "run_tool_loop", loop)
    out = workflow.start_request(graph(), "t-ins", record("What will an MRI cost me?"))
    assert out["status"] == "done" and out["stop_reason"] == "answered"
    assert out["final_answer"] == "Your estimated cost is $110."
    assert ("resource", "policy://clinical-escalation") in FakeEHR.log  # reviewer read the SOP
    assert len(env) == 1  # one LLM review


def test_referral_question_goes_to_the_referral_agent(env, monkeypatch):
    loop = ScriptedLoop(("Referral MHP-R-00000 is accepted; due by 2026-05-18.",
                         [("get_referral_status", {})]))
    monkeypatch.setattr(team, "run_tool_loop", loop)
    out = workflow.start_request(graph(), "t-ref", record("Where is my referral?"))
    assert out["stop_reason"] == "answered"
    assert ("resource", "careflow://urgency-bands") in FakeEHR.log
    assert ("schemas", sorted(team.REFERRAL_TOOLS)) in FakeEHR.log


def test_specialist_sees_only_its_slice_of_state(env, monkeypatch):
    loop = ScriptedLoop(("Referral MHP-R-00000 is accepted.", [("get_referral_status", {})]))
    monkeypatch.setattr(team, "run_tool_loop", loop)
    workflow.start_request(graph(), "t-slice", record("Where is my referral?"))
    assert "insurance_result" not in loop.users[0] and "audit" not in loop.users[0]


def test_reviewer_catches_dosage_advice_and_sends_it_back(env, monkeypatch):
    loop = ScriptedLoop(
        ("Your MRI is covered. Also take 50 mg of ibuprofen.", []),
        ("Your estimated cost is $110.", [("calculate_copay", {"procedure_code": "ORTH-MRI"})]),
    )
    monkeypatch.setattr(team, "run_tool_loop", loop)
    out = workflow.start_request(graph(), "t-critic", record("What will an MRI cost me?"))
    assert out["final_answer"] == "Your estimated cost is $110."
    reviews = [a for a in out["audit"] if a["event"] == "review"]
    assert reviews[0]["verdict"] == "revise" and "dose" in reviews[0]["issues"][0]
    assert "rejected your previous draft" in loop.users[1]  # feedback reached the agent


def test_reviewer_catches_an_invented_figure(env, monkeypatch):
    loop = ScriptedLoop(
        ("Your estimated cost is $999.", [("calculate_copay", {"procedure_code": "ORTH-MRI"})]),
        ("Your estimated cost is $110.", [("calculate_copay", {"procedure_code": "ORTH-MRI"})]),
    )
    monkeypatch.setattr(team, "run_tool_loop", loop)
    out = workflow.start_request(graph(), "t-fig", record("What will an MRI cost me?"))
    first = next(a for a in out["audit"] if a["event"] == "review")
    assert "999" in first["issues"][0]
    assert out["final_answer"] == "Your estimated cost is $110."


def test_llm_reviewer_can_also_send_work_back(env, monkeypatch):
    verdicts = iter(['{"verdict": "revise", "issues": ["Answers a should-I question"]}',
                     '{"verdict": "approve", "issues": []}'])
    monkeypatch.setattr(team, "call_llm", lambda m, json_mode=False: next(verdicts))
    monkeypatch.setattr(team, "run_tool_loop", ScriptedLoop(("Your plan is active.", [])))
    out = workflow.start_request(graph(), "t-llm", record("Is my plan active?"))
    assert out["stop_reason"] == "answered"
    assert [a["verdict"] for a in out["audit"] if a["event"] == "review"] == ["revise", "approve"]


def test_revision_budget_ends_with_a_human_handoff(env, monkeypatch):
    monkeypatch.setattr(team, "run_tool_loop", ScriptedLoop(("Take 2 tablets daily.", [])))
    out = workflow.start_request(graph(), "t-budget", record("What will an MRI cost me?"))
    assert out["stop_reason"] == "handoff_revision_budget_spent"
    reviews = [a for a in out["audit"] if a["event"] == "review"]
    assert len(reviews) == config.MAX_REVISIONS + 1


def test_clinical_request_pauses_and_resumes_after_restart(env):
    g = graph()
    paused = workflow.start_request(g, "t-clin", record("Should I stop my tablets?"))
    assert paused["status"] == "paused"
    assert paused["interrupt"]["type"] == "clinical_escalation_approval"
    g.checkpointer.conn.close()  # the process dies here
    done = workflow.resume_request(graph(), "t-clin", {
        "decision": "approve", "approver": "Nurse Rao", "reason": "Medication question"})
    assert done["stop_reason"] == "escalated"
    assert len(EscalationQueue().for_thread("t-clin")) == 1


def test_clinical_request_can_be_rejected(env):
    g = graph()
    workflow.start_request(g, "t-clin-r", record("Should I see a doctor?"))
    done = workflow.resume_request(graph(), "t-clin-r", {
        "decision": "reject", "approver": "Nurse Rao", "reason": "Handled by phone"})
    assert done["stop_reason"] == "escalation_rejected"
    assert EscalationQueue().for_thread("t-clin-r") == []


def test_emergency_is_redirected(env):
    out = workflow.start_request(graph(), "t-em", record("I can't breathe"))
    assert out["stop_reason"] == "emergency_redirect"


def test_deterministic_checks():
    assert team.deterministic_issues("Your plan is active.", "{}", "") == []
    assert team.deterministic_issues("Stop taking your tablets.", "{}", "")
    assert team.deterministic_issues("It costs $40.", '{"x": 40}', "") == []
    assert team.deterministic_issues("It costs $41.", '{"x": 40}', "")