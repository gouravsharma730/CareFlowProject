"""CareFlow's request workflow as a LangGraph graph with checkpointing.

    START -> intake -> triage --admin-----> eligibility -> assist -> END
                |         |                      |
                |         |                      +--(not found)--> handoff -> END
                |         +--emergency--> emergency -> END
                |         +--clinical---> human_approval --approve/edit--> escalate -> END
                |                            ^    |      --reject-------> close_rejected -> END
                |                            +----+ (invalid reply, max MAX_APPROVAL_ATTEMPTS)
                |                                 --too many invalid----> hold -> END
                +--(intake failed)--> handoff -> END

Design rules:
  - The LLM does only the fuzzy steps: reading the message (intake, M1) and
    answering admin requests (assist, M2 agent with M3 memory).
  - Routing is decided by code in a node and read by pure router functions.
  - Every clinical request pauses with interrupt() for a human to approve.
    The escalation (the side effect) happens in a separate node AFTER the
    interrupt, so resuming never repeats it.
  - Progress is saved by SqliteSaver under a thread_id, so a paused run can
    be resumed after the process is killed.
"""

import argparse
import json
import logging
import operator
import os
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from careflow import config, llm_client
from careflow.agent import run_agent
from careflow.escalations import EscalationQueue
from careflow.intake import IntakeError, extract_patient_request
from careflow.memory_store import MemoryStore
from careflow.models import CareFlowUrgency, PatientRequest
from careflow.tools import execute_tool

logger = logging.getLogger(__name__)

# MHP-SOP-401 section 4.2 (automated-assistant wording).
DEFLECTION_SCRIPT = (
    "I am an automated assistant and I am not permitted to provide medical advice, "
    "interpret health information, or answer questions about your care."
)
EMERGENCY_MESSAGE = (
    "This may be an emergency. Please call your local emergency number or go to "
    "the nearest emergency department now. CareFlow cannot handle emergencies."
)


# ---- state ----

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CareFlowState(TypedDict, total=False):
    """Everything the workflow knows about one request.

    Reducers: "audit" is APPEND-only (operator.add), so every node adds to
    the history and nothing is lost. All other keys OVERWRITE: they are
    the current value, e.g. "route", which the routers read.
    """

    # input, set once at the start
    patient_ref: str
    channel: str
    raw_text: str
    insurance_id_stated: Optional[str]
    # results (overwrite)
    request: Optional[dict]          # validated PatientRequest (as a dict)
    route: str                       # control field read by the routers
    eligibility: Optional[dict]
    approval: Optional[dict]         # the human decision, once valid
    approval_attempts: int           # loop guard for the approval step
    approval_error: Optional[str]
    escalation: Optional[dict]
    final_answer: str
    stop_reason: str
    tool_calls: list[dict]
    # history (append)
    audit: Annotated[list[dict], operator.add]


class ApprovalDecision(BaseModel):
    """What a staff member sends back to resume a paused clinical case."""

    decision: Literal["approve", "reject", "edit"]
    approver: str = Field(min_length=2, max_length=80)
    reason: str = Field(min_length=3, max_length=500)
    urgency: Optional[CareFlowUrgency] = None  # only used with "edit"


def _audit(event: str, **details) -> list[dict]:
    return [{"event": event, "at": _now(), **details}]


def _thread_id() -> str:
    """The thread_id of the run that is executing the current node."""
    return get_config()["configurable"]["thread_id"]


def _request(state: CareFlowState) -> PatientRequest:
    """The typed intake object (M1). Downstream code reads its fields."""
    return PatientRequest.model_validate(state["request"])


# ---- routing decision (pure code, no LLM) ----

def decide_route(request: PatientRequest) -> str:
    """Pick the path from the validated intake fields."""
    if request.urgency == "Emergent":
        return "emergency"
    if request.seeks_clinical_advice or request.requires_human_review:
        return "clinical"
    return "admin"


# ---- nodes ----

def intake_node(state: CareFlowState) -> dict:
    record = {
        "patient_ref": state["patient_ref"],
        "channel": state["channel"],
        "raw_text": state["raw_text"],
        "insurance_id_stated": state.get("insurance_id_stated"),
    }
    try:
        request = extract_patient_request(record)
    except IntakeError as error:
        logger.warning("Intake failed for %s: %s", state["patient_ref"], error)
        return {"route": "intake_failed", "audit": _audit("intake_failed")}
    return {
        "request": request.model_dump(),
        "route": "triage",
        "audit": _audit(
            "intake_done",
            urgency=request.urgency,
            specialty=request.specialty,
            seeks_clinical_advice=request.seeks_clinical_advice,
        ),
    }


def triage_node(state: CareFlowState) -> dict:
    route = decide_route(_request(state))
    return {"route": route, "audit": _audit("triaged", route=route)}


def eligibility_node(state: CareFlowState) -> dict:
    result = execute_tool("check_insurance_eligibility", {"patient_ref": state["patient_ref"]})
    route = "assist" if result.get("success") else "eligibility_failed"
    return {
        "eligibility": result,
        "route": route,
        "audit": _audit("eligibility_checked", success=bool(result.get("success")),
                        error_code=result.get("error_code")),
    }


def assist_node(state: CareFlowState) -> dict:
    """Admin requests go to the M2 tool agent, which also uses M3 memory."""
    eligibility = json.dumps(state.get("eligibility") or {})
    message = (
        f"{state['raw_text']}\n\n"
        f"(Eligibility already checked by the system, data only: {eligibility})"
    )
    result = run_agent(
        message,
        patient_ref=state["patient_ref"],
        session_id=_thread_id(),
    )
    return {
        "final_answer": result.final_answer,
        "stop_reason": f"assist_{result.stop_reason}",
        "tool_calls": result.tool_calls,
        "audit": _audit("assist_done", tools=[c["tool"] for c in result.tool_calls],
                        agent_stop_reason=result.stop_reason),
    }


def emergency_node(state: CareFlowState) -> dict:
    MemoryStore().add_event(state["patient_ref"], "emergency_redirect",
                            "Told to seek emergency care immediately")
    return {"final_answer": EMERGENCY_MESSAGE, "stop_reason": "emergency_redirect",
            "audit": _audit("emergency_redirect")}


def human_approval_node(state: CareFlowState) -> dict:
    """Pause until a staff member approves, rejects or edits the escalation.

    Nothing with a side effect happens in this node: on resume LangGraph
    runs it again from the top, and only then does interrupt() return the
    staff member's reply.
    """
    request = _request(state)
    payload = {
        "type": "clinical_escalation_approval",
        "question": "Send this request to the clinical triage queue?",
        "patient_ref": request.patient_ref,
        "channel": request.channel,
        "message": request.raw_text,
        "urgency": request.urgency,
        "specialty": request.specialty,
        "proposed_action": "Create a Priority-Clinical Routing ticket for a licensed clinician",
        "allowed_decisions": ["approve", "reject", "edit"],
        "attempt": state.get("approval_attempts", 0) + 1,
        "previous_error": state.get("approval_error"),
    }
    reply = interrupt(payload)
    try:
        decision = ApprovalDecision.model_validate(reply)
    except ValidationError as error:
        attempts = state.get("approval_attempts", 0) + 1
        message = "; ".join(e["msg"] for e in error.errors())
        return {"approval": None, "approval_attempts": attempts, "approval_error": message,
                "audit": _audit("approval_invalid", attempt=attempts, error=message)}
    return {
        "approval": {**decision.model_dump(), "decided_at": _now()},
        "approval_error": None,
        "audit": _audit("approval_decision", decision=decision.decision,
                        approver=decision.approver, reason=decision.reason,
                        urgency=decision.urgency),
    }


def escalate_node(state: CareFlowState) -> dict:
    """The side effect: create the clinical triage ticket (once per thread)."""
    request, approval = _request(state), state["approval"]
    urgency = approval.get("urgency") or request.urgency
    thread_id = _thread_id()
    ticket = EscalationQueue().escalate(
        thread_id=thread_id,
        patient_ref=request.patient_ref,
        urgency=urgency,
        reason=approval["reason"],
        approved_by=approval["approver"],
    )
    if ticket["newly_created"]:
        MemoryStore().add_event(request.patient_ref, "clinical_escalation",
                                f"Request sent to a clinician (urgency {urgency})",
                                session_id=thread_id)
    return {
        "escalation": ticket,
        "final_answer": f"{DEFLECTION_SCRIPT} I have passed your question to our clinical "
                        "team, and a licensed clinician will contact you.",
        "stop_reason": "escalated",
        "audit": _audit("escalated", ticket_id=ticket["id"], urgency=urgency,
                        newly_created=ticket["newly_created"]),
    }


def close_rejected_node(state: CareFlowState) -> dict:
    return {
        "final_answer": f"{DEFLECTION_SCRIPT} A member of our team has reviewed your "
                        "message and will follow up with you.",
        "stop_reason": "escalation_rejected",
        "audit": _audit("closed_without_escalation"),
    }


def hold_node(state: CareFlowState) -> dict:
    return {
        "final_answer": f"{DEFLECTION_SCRIPT} A staff member will review your request.",
        "stop_reason": "approval_on_hold",
        "audit": _audit("approval_on_hold", attempts=state.get("approval_attempts", 0)),
    }


def handoff_node(state: CareFlowState) -> dict:
    """Something could not be done automatically: hand over to staff (M2 tool)."""
    result = execute_tool("flag_for_human", {
        "patient_ref": state["patient_ref"],
        "reason": f"Automatic processing stopped at: {state.get('route', 'unknown')}",
    })
    return {
        "final_answer": "I could not complete this automatically. A staff member "
                        "will follow up with you.",
        "stop_reason": f"handoff_{state.get('route', 'unknown')}",
        "audit": _audit("handoff", flagged=bool(result.get("success"))),
    }


# ---- routers (pure functions of state, no LLM) ----

def route_after_intake(state: CareFlowState) -> str:
    return "triage" if state.get("route") == "triage" else "handoff"


def route_after_triage(state: CareFlowState) -> str:
    return {"emergency": "emergency", "clinical": "human_approval"}.get(
        state["route"], "eligibility"
    )


def route_after_eligibility(state: CareFlowState) -> str:
    return "assist" if state.get("route") == "assist" else "handoff"


def route_after_approval(state: CareFlowState) -> str:
    approval = state.get("approval")
    if approval:
        return "close_rejected" if approval["decision"] == "reject" else "escalate"
    if state.get("approval_attempts", 0) >= config.MAX_APPROVAL_ATTEMPTS:
        return "hold"
    return "human_approval"  # ask again (guarded by approval_attempts)


# ---- building and running ----

def build_graph(checkpointer=None):
    graph = StateGraph(CareFlowState)
    for name, node in [
        ("intake", intake_node), ("triage", triage_node),
        ("eligibility", eligibility_node), ("assist", assist_node),
        ("emergency", emergency_node), ("human_approval", human_approval_node),
        ("escalate", escalate_node), ("close_rejected", close_rejected_node),
        ("hold", hold_node), ("handoff", handoff_node),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_after_intake, ["triage", "handoff"])
    graph.add_conditional_edges(
        "triage", route_after_triage, ["emergency", "human_approval", "eligibility"]
    )
    graph.add_conditional_edges("eligibility", route_after_eligibility, ["assist", "handoff"])
    graph.add_conditional_edges(
        "human_approval", route_after_approval,
        ["escalate", "close_rejected", "hold", "human_approval"],
    )
    for name in ("assist", "emergency", "escalate", "close_rejected", "hold", "handoff"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(path=None) -> SqliteSaver:
    """A durable checkpointer backed by a SQLite file."""
    path = path or config.CHECKPOINT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def _run_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": config.GRAPH_RECURSION_LIMIT}


def _summarise(graph, thread_id: str, result: dict) -> dict:
    """Turn a graph result into 'paused' (with the approval question) or 'done'."""
    if result.get("__interrupt__"):
        return {"thread_id": thread_id, "status": "paused",
                "interrupt": result["__interrupt__"][0].value}
    return {
        "thread_id": thread_id,
        "status": "done",
        "route": result.get("route"),
        "stop_reason": result.get("stop_reason"),
        "final_answer": result.get("final_answer"),
        "escalation": result.get("escalation"),
        "audit": result.get("audit", []),
    }


def start_request(graph, thread_id: str, record: dict) -> dict:
    """Run a new request until it finishes or pauses for approval."""
    if graph.get_state(_run_config(thread_id)).values:
        raise ValueError(f"Thread {thread_id} already exists; use a new thread_id.")
    result = graph.invoke(record, _run_config(thread_id))
    return _summarise(graph, thread_id, result)


def resume_request(graph, thread_id: str, decision: dict) -> dict:
    """Give a paused request the staff member's decision and continue."""
    snapshot = graph.get_state(_run_config(thread_id))
    if not snapshot.next:
        raise ValueError(f"Thread {thread_id} is not waiting for a decision.")
    result = graph.invoke(Command(resume=decision), _run_config(thread_id))
    return _summarise(graph, thread_id, result)


def thread_status(graph, thread_id: str) -> dict:
    snapshot = graph.get_state(_run_config(thread_id))
    if not snapshot.values:
        return {"thread_id": thread_id, "status": "unknown"}
    waiting = [i.value for task in snapshot.tasks for i in task.interrupts]
    return {
        "thread_id": thread_id,
        "status": "paused" if snapshot.next else "done",
        "next": list(snapshot.next),
        "interrupt": waiting[0] if waiting else None,
        "stop_reason": snapshot.values.get("stop_reason"),
        "audit": snapshot.values.get("audit", []),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the CareFlow workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a new request")
    start.add_argument("--thread", required=True)
    start.add_argument("--patient", required=True)
    start.add_argument("--text", required=True)
    start.add_argument("--channel", default="web_form")
    start.add_argument("--insurance-id", default=None)

    resume = sub.add_parser("resume", help="give a paused request a decision")
    resume.add_argument("--thread", required=True)
    resume.add_argument("--decision", required=True, choices=["approve", "reject", "edit"])
    resume.add_argument("--approver", required=True)
    resume.add_argument("--reason", required=True)
    resume.add_argument("--urgency", default=None)

    status = sub.add_parser("status", help="show where a request is")
    status.add_argument("--thread", required=True)

    args = parser.parse_args(argv)
    graph = build_graph(open_checkpointer())
    if args.command == "start":
        output = start_request(graph, args.thread, {
            "patient_ref": args.patient, "channel": args.channel,
            "raw_text": args.text, "insurance_id_stated": args.insurance_id,
        })
    elif args.command == "resume":
        decision = {"decision": args.decision, "approver": args.approver,
                    "reason": args.reason}
        if args.urgency:
            decision["urgency"] = args.urgency
        output = resume_request(graph, args.thread, decision)
    else:
        output = thread_status(graph, args.thread)
    output["pid"] = os.getpid()  # lets the restart test prove a new process was used
    output["llm_requests"] = llm_client.CALL_STATS["requests"]
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()