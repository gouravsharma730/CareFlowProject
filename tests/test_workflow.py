"""Tests for the LangGraph workflow (offline: fake intake and fake agent).

The restart tests close the SQLite connection and build a brand-new graph
on the same checkpoint file, which is what a new process does.
"""

import json

import pytest

from careflow import config
from careflow import workflow
from careflow.agent import AgentResult
from careflow.escalations import EscalationQueue
from careflow.intake import IntakeError
from careflow.models import PatientRequest

PATIENT = "MHP-P-00000"  # exists in the mock patient table


def fake_intake(record):
    text = record["raw_text"].lower()
    if "garbled" in text:
        raise IntakeError("could not parse")
    clinical = "should i" in text or "dose" in text
    return PatientRequest(
        patient_ref=record["patient_ref"], channel=record["channel"],
        raw_text=record["raw_text"], specialty="Cardiology", insurance_id=None,
        urgency="Emergent" if "can't breathe" in text else "Routine",
        seeks_clinical_advice=clinical, requires_human_review=clinical,
    )


class FakeAgent:
    def __init__(self):
        self.calls = []

    def __call__(self, message, patient_ref=None, session_id=None, history=None):
        self.calls.append({"message": message, "patient_ref": patient_ref,
                           "session_id": session_id})
        return AgentResult(final_answer="Your plan is active.", stop_reason="final_answer",
                           steps=1, tool_calls=[])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "memory.sqlite")
    monkeypatch.setattr(config, "OPERATIONS_DB_PATH", tmp_path / "ops.sqlite")
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(workflow, "extract_patient_request", fake_intake)
    agent = FakeAgent()
    monkeypatch.setattr(workflow, "run_agent", agent)
    return agent


def new_graph():
    """A fresh graph on the checkpoint file, as a new process would build it."""
    return workflow.build_graph(workflow.open_checkpointer())


def record(text, patient=PATIENT):
    return {"patient_ref": patient, "channel": "sms", "raw_text": text,
            "insurance_id_stated": None}


APPROVE = {"decision": "approve", "approver": "Nurse Rao", "reason": "Medication question"}
REJECT = {"decision": "reject", "approver": "Nurse Rao", "reason": "Not clinical"}


# ---- routing is pure code ----

@pytest.mark.parametrize("urgency, clinical, expected", [
    ("Emergent", False, "emergency"),
    ("Emergent", True, "emergency"),
    ("Routine", True, "clinical"),
    ("Priority", False, "admin"),
])
def test_decide_route(urgency, clinical, expected):
    request = PatientRequest(patient_ref="p", channel="sms", raw_text="x", urgency=urgency,
                             seeks_clinical_advice=clinical, requires_human_review=False)
    assert workflow.decide_route(request) == expected


def test_routers_read_only_state():
    assert workflow.route_after_triage({"route": "clinical"}) == "human_approval"
    assert workflow.route_after_triage({"route": "admin"}) == "eligibility"
    assert workflow.route_after_approval({"approval": {"decision": "reject"}}) == "close_rejected"
    assert workflow.route_after_approval({"approval": {"decision": "edit"}}) == "escalate"
    assert workflow.route_after_approval({"approval_attempts": 1}) == "human_approval"
    assert workflow.route_after_approval(
        {"approval_attempts": config.MAX_APPROVAL_ATTEMPTS}) == "hold"


# ---- paths through the graph ----

def test_admin_request_does_not_pause_and_uses_the_agent(env):
    out = workflow.start_request(new_graph(), "t-admin", record("Is my insurance active?"))
    assert out["status"] == "done"
    assert out["stop_reason"] == "assist_final_answer"
    assert env.calls[0]["patient_ref"] == PATIENT
    assert env.calls[0]["session_id"] == "t-admin"
    assert "eligibility_status" in env.calls[0]["message"]  # eligibility passed on


def test_clinical_request_pauses_with_a_clear_question(env):
    out = workflow.start_request(new_graph(), "t-c", record("Should I double my dose?"))
    assert out["status"] == "paused"
    question = out["interrupt"]
    assert question["type"] == "clinical_escalation_approval"
    assert question["patient_ref"] == PATIENT
    assert question["allowed_decisions"] == ["approve", "reject", "edit"]
    assert EscalationQueue().for_thread("t-c") == []  # nothing happened yet


def test_approve_after_restart_escalates_exactly_once(env):
    graph = new_graph()
    workflow.start_request(graph, "t-restart", record("Should I stop my tablets?"))
    graph.checkpointer.conn.close()  # the "process" dies here

    out = workflow.resume_request(new_graph(), "t-restart", APPROVE)
    assert out["status"] == "done" and out["stop_reason"] == "escalated"
    rows = EscalationQueue().for_thread("t-restart")
    assert len(rows) == 1 and rows[0]["approved_by"] == "Nurse Rao"
    decisions = [a for a in out["audit"] if a["event"] == "approval_decision"]
    assert decisions[0]["approver"] == "Nurse Rao" and decisions[0]["at"]


def test_reject_ends_with_no_side_effect(env):
    graph = new_graph()
    workflow.start_request(graph, "t-reject", record("Should I see someone?"))
    out = workflow.resume_request(new_graph(), "t-reject", REJECT)
    assert out["stop_reason"] == "escalation_rejected"
    assert EscalationQueue().for_thread("t-reject") == []


def test_edit_changes_the_urgency_of_the_escalation(env):
    graph = new_graph()
    workflow.start_request(graph, "t-edit", record("Should I come in today?"))
    edit = {**APPROVE, "decision": "edit", "urgency": "Urgent"}
    workflow.resume_request(graph, "t-edit", edit)
    assert EscalationQueue().for_thread("t-edit")[0]["urgency"] == "Urgent"


def test_invalid_replies_are_asked_again_then_put_on_hold(env):
    graph = new_graph()
    workflow.start_request(graph, "t-bad", record("Should I take more?"))
    out = None
    for attempt in range(1, config.MAX_APPROVAL_ATTEMPTS + 1):
        out = workflow.resume_request(graph, "t-bad", {"decision": "maybe"})
        if attempt < config.MAX_APPROVAL_ATTEMPTS:
            assert out["status"] == "paused"
            assert out["interrupt"]["previous_error"]
            assert out["interrupt"]["attempt"] == attempt + 1
    assert out["status"] == "done" and out["stop_reason"] == "approval_on_hold"
    assert EscalationQueue().for_thread("t-bad") == []


def test_emergency_is_redirected_without_pausing(env):
    out = workflow.start_request(new_graph(), "t-em", record("I can't breathe"))
    assert out["status"] == "done" and out["stop_reason"] == "emergency_redirect"
    assert "emergency" in out["final_answer"].lower()


def test_intake_failure_hands_off_to_staff(env):
    out = workflow.start_request(new_graph(), "t-garbled", record("garbled ###"))
    assert out["stop_reason"] == "handoff_intake_failed"


def test_unknown_patient_is_handed_off_after_eligibility(env):
    out = workflow.start_request(new_graph(), "t-unknown",
                                 record("Is my plan active?", patient="MHP-P-99999"))
    assert out["stop_reason"] == "handoff_eligibility_failed"
    assert env.calls == []


# ---- guards ----

def test_resume_and_start_are_checked(env):
    graph = new_graph()
    workflow.start_request(graph, "t-done", record("Is my plan active?"))
    with pytest.raises(ValueError):
        workflow.resume_request(graph, "t-done", APPROVE)   # not paused
    with pytest.raises(ValueError):
        workflow.start_request(graph, "t-done", record("again"))  # thread reused


def test_escalation_queue_is_idempotent_per_thread(env):
    queue = EscalationQueue()
    first = queue.escalate("t-x", PATIENT, "Routine", "reason", "Nurse Rao")
    second = queue.escalate("t-x", PATIENT, "Routine", "reason", "Nurse Rao")
    assert first["newly_created"] and not second["newly_created"]
    assert len(queue.for_thread("t-x")) == 1


def test_recursion_limit_comes_from_config():
    assert workflow._run_config("t")["recursion_limit"] == config.GRAPH_RECURSION_LIMIT


def test_command_line_start_and_resume(env, capsys):
    workflow.main(["start", "--thread", "t-cli", "--patient", PATIENT,
                   "--text", "Should I change my dose?"])
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "paused" and "pid" in paused
    workflow.main(["resume", "--thread", "t-cli", "--decision", "approve",
                   "--approver", "Nurse Rao", "--reason", "Dose question"])
    done = json.loads(capsys.readouterr().out)
    assert done["stop_reason"] == "escalated"