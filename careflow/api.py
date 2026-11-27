"""CareFlow as a web service (FastAPI).

Endpoints
  GET  /health                     liveness check
  POST /chat                       a patient message through the agent team
  POST /approvals/{thread_id}      staff decision on a paused clinical request

Status codes
  200  answered (or a hand-off / emergency message)
  202  paused: waiting for a clinician to approve the escalation
  401  approvals: missing or wrong staff token
  404  approvals: unknown thread
  409  approvals: the thread is not waiting for a decision
  422  guardrail rejection (error "blocked_by_guardrail", with reason and flags)
       or a malformed request (error "invalid_request"); never a 500
  503  the service could not complete the request right now

Run:
    python -m careflow.api
    (or: uvicorn careflow.api:app --host 127.0.0.1 --port 8000)
"""

import hmac
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from careflow import config, team, workflow
from careflow.models import CareFlowUrgency

logger = logging.getLogger(__name__)

HOLDING_REPLY = (f"{workflow.DEFLECTION_SCRIPT} I have passed your question to our clinical "
                 "team for review, and a licensed clinician will contact you.")


class ChatRequest(BaseModel):
    patient_ref: str = Field(pattern=r"^MHP-P-\d{5}$", description="Patient reference.")
    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    channel: Literal["web_form", "sms", "email", "phone_transcript", "app"] = "web_form"
    insurance_id: Optional[str] = Field(default=None, max_length=40)
    thread_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{3,80}$",
                                     description="Optional; a new one is created if omitted.")


class ChatResponse(BaseModel):
    thread_id: str
    status: str                  # answered, pending_clinician_review, emergency_redirect, ...
    reply: str
    stop_reason: Optional[str] = None
    needs_clinician: bool = False


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    approver: str = Field(min_length=2, max_length=80)
    reason: str = Field(min_length=3, max_length=500)
    urgency: Optional[CareFlowUrgency] = None


def _error(status: int, error: str, reason: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "reason": reason, **extra})


def create_app(graph=None) -> FastAPI:
    """Build the app. Tests pass their own graph; the service builds the real one."""
    lock = threading.Lock()  # one graph run at a time (SQLite checkpointer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph = graph or team.build_team_graph(workflow.open_checkpointer())
        yield

    app = FastAPI(title="CareFlow", version="1.0",
                  description="AI care-coordination assistant (synthetic data only).",
                  lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Report what is wrong without echoing the submitted values back.
        problems = [{"field": ".".join(str(p) for p in e.get("loc", [])[1:]) or "body",
                     "problem": e.get("msg", "invalid")} for e in exc.errors()]
        return _error(422, "invalid_request", "The request body is missing or invalid.",
                      problems=problems)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "careflow", "model": config.LLM_MODEL,
                "tracing": config.LANGFUSE_CONFIGURED, "approvals_enabled": bool(config.STAFF_API_TOKEN)}

    @app.post("/chat", response_model=ChatResponse,
              responses={202: {"model": ChatResponse}, 422: {"description": "Blocked or invalid"}})
    def chat(body: ChatRequest):
        thread_id = body.thread_id or f"api-{uuid.uuid4().hex[:16]}"
        record = {"patient_ref": body.patient_ref, "channel": body.channel,
                  "raw_text": body.message, "insurance_id_stated": body.insurance_id}
        try:
            with lock:
                result = workflow.start_request(app.state.graph, thread_id, record)
                state = app.state.graph.get_state(
                    {"configurable": {"thread_id": thread_id}}).values
        except ValueError as error:  # e.g. the thread id is already used
            return _error(409, "thread_exists", str(error))
        except Exception:
            logger.exception("Chat request failed (thread %s)", thread_id)
            return _error(503, "temporarily_unavailable",
                          "The assistant could not complete this request. Please try again later.")

        if result.get("stop_reason") == "blocked_input":
            guard = state.get("guard") or {}
            return _error(422, "blocked_by_guardrail", guard.get("reason", "Request blocked."),
                          flags=guard.get("flags", []), thread_id=thread_id)
        if result["status"] == "paused":
            response = ChatResponse(thread_id=thread_id, status="pending_clinician_review",
                                    reply=HOLDING_REPLY, needs_clinician=True)
            return JSONResponse(status_code=202, content=response.model_dump())
        stop = result.get("stop_reason") or "done"
        return ChatResponse(thread_id=thread_id, reply=result.get("final_answer") or "",
                            status="answered" if stop == "answered" else stop, stop_reason=stop)

    @app.post("/approvals/{thread_id}")
    def approve(thread_id: str, body: ApprovalRequest,
                x_staff_token: Optional[str] = Header(default=None)):
        if not config.STAFF_API_TOKEN:
            return _error(503, "approvals_disabled",
                          "Set CAREFLOW_STAFF_API_TOKEN to enable staff approvals.")
        if not x_staff_token or not hmac.compare_digest(x_staff_token, config.STAFF_API_TOKEN):
            return _error(401, "unauthorized", "A valid X-Staff-Token header is required.")
        run_config = {"configurable": {"thread_id": thread_id}}
        try:
            with lock:
                snapshot = app.state.graph.get_state(run_config)
                if not snapshot.values:
                    return _error(404, "unknown_thread", f"No request with thread id {thread_id}.")
                if not snapshot.next:
                    return _error(409, "not_waiting", "This request is not waiting for a decision.")
                result = workflow.resume_request(app.state.graph, thread_id,
                                                 body.model_dump(exclude_none=True))
        except Exception:
            logger.exception("Approval failed (thread %s)", thread_id)
            return _error(503, "temporarily_unavailable", "The decision could not be recorded.")
        return {"thread_id": thread_id, "status": result["status"],
                "stop_reason": result.get("stop_reason"), "reply": result.get("final_answer")}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("careflow.api:app", host=config.API_HOST, port=config.API_PORT)