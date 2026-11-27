"""Tracing for CareFlow: Langfuse spans plus a local telemetry log.

One request = one trace. Inside it:
    request (chain)
      -> one span per graph node; agents are typed "agent"
           -> one span per tool call (type "tool")
           -> one generation per LLM call (model, tokens, cost)

Every trace carries a session id (the workflow thread_id) and a user id.
The user id is a salted hash of the patient reference: no names, patient
references, messages, prompts or replies are ever sent to the tracer, only
metadata such as routes, tool names, error codes, counts and timings.

Langfuse is used when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(and CAREFLOW_TRACING is not "false"). Whether or not Langfuse is on, each
finished span is also appended to CAREFLOW_TELEMETRY_PATH, which
evals/kpi_report.py turns into the KPI table.

Call flush() before a script or process ends so traces are sent.
"""

import atexit
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone

from careflow import config

logger = logging.getLogger(__name__)

_current_agent = contextvars.ContextVar("careflow_agent", default=None)
_current_thread = contextvars.ContextVar("careflow_thread", default=None)
_client = None
_client_lock = threading.Lock()
_write_lock = threading.Lock()

# Metadata keys copied into the local telemetry log (never free text).
TELEMETRY_KEYS = ("route", "verdict", "error_code", "used_fallback", "tool", "tools",
                  "retries", "fallbacks", "breaker_state", "attempts", "decided_by",
                  "answerable", "retrieved", "revisions")


def langfuse_enabled() -> bool:
    return config.TRACING_ENABLED and (config.LANGFUSE_CONFIGURED or _client is not None)


def get_client():
    """The Langfuse client, created on first use."""
    global _client
    with _client_lock:
        if _client is None:
            from langfuse import Langfuse  # only imported when tracing is on

            # Keys come from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env.
            host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            _client = Langfuse(host=host, environment=config.TRACING_ENVIRONMENT)
            atexit.register(flush)
        return _client


def set_client(client) -> None:
    """Use a specific Langfuse client (tests pass one with an in-memory exporter)."""
    global _client
    _client = client


def flush() -> None:
    """Send any buffered traces now (call at the end of scripts and requests)."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.warning("Could not flush traces", exc_info=True)


def pseudonym(patient_ref: str | None) -> str | None:
    """A stable, non-reversible id for a patient, safe to send to the tracer."""
    if not patient_ref:
        return None
    digest = hashlib.sha256(f"{config.TRACE_USER_SALT}:{patient_ref}".encode()).hexdigest()
    return f"pt-{digest[:12]}"


def _write_event(event: dict) -> None:
    try:
        config.TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, open(config.TELEMETRY_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, default=str) + "\n")
    except OSError:
        logger.warning("Could not write telemetry", exc_info=True)


class Observation:
    """Handle given to the code inside a span, to attach results to it."""

    def __init__(self, metadata: dict):
        self.metadata = dict(metadata)
        self.usage: dict[str, int] = {}
        self.cost: float | None = None
        self.model: str | None = None
        self.level = "DEFAULT"
        self.status: str | None = None

    def update(self, **metadata) -> None:
        self.metadata.update(metadata)

    def warn(self, status: str) -> None:
        self.level, self.status = "WARNING", status

    def record_llm_response(self, response, model: str) -> None:
        """Take token usage from a LiteLLM response and price it."""
        self.model = model
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        self.usage = {"input": prompt, "output": completion, "total": prompt + completion}
        try:
            import litellm

            input_cost, output_cost = litellm.cost_per_token(
                model=model, prompt_tokens=prompt, completion_tokens=completion)
            self.cost = round(input_cost + output_cost, 8)
        except Exception:
            self.cost = None  # model not in LiteLLM's price list


def _is_pause(error: BaseException) -> bool:
    """LangGraph's interrupt() works by raising; that is a pause, not a failure."""
    return type(error).__name__ in ("GraphInterrupt", "NodeInterrupt")


@contextmanager
def span(name: str, as_type: str = "span", model: str | None = None, **metadata):
    """Record one step. as_type: span, agent, tool, generation, retriever, chain."""
    handle = Observation(metadata)
    agent_token = _current_agent.set(name) if as_type == "agent" else None
    start = time.perf_counter()
    lf_context = (get_client().start_as_current_observation(
        name=name, as_type=as_type, model=model, metadata=metadata)
        if langfuse_enabled() else nullcontext())
    try:
        with lf_context as observation:
            try:
                yield handle
            except BaseException as error:
                if _is_pause(error):
                    handle.status = "paused for human approval"
                else:
                    handle.level, handle.status = "ERROR", type(error).__name__
                raise
            finally:
                if observation is not None:
                    update = {"metadata": handle.metadata, "level": handle.level,
                              "status_message": handle.status}
                    if as_type == "generation":
                        update.update(model=handle.model or model,
                                      usage_details=handle.usage or None,
                                      cost_details={"total": handle.cost}
                                      if handle.cost is not None else None)
                    observation.update(**{k: v for k, v in update.items() if v is not None})
    finally:
        event = {
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "thread_id": _current_thread.get(),
            "agent": name if as_type == "agent" else _current_agent.get(),
            "name": name,
            "type": as_type,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            "level": handle.level,
            "status": handle.status,
        }
        if as_type == "generation":
            event.update(model=handle.model or model, tokens_in=handle.usage.get("input", 0),
                         tokens_out=handle.usage.get("output", 0), cost_usd=handle.cost)
        event.update({k: handle.metadata[k] for k in TELEMETRY_KEYS if k in handle.metadata})
        _write_event(event)
        if agent_token is not None:
            _current_agent.reset(agent_token)


@contextmanager
def request_trace(name: str, thread_id: str, patient_ref: str | None, **metadata):
    """The root of one request's trace, carrying session and (hashed) user ids."""
    thread_token = _current_thread.set(thread_id)
    try:
        with span(name, as_type="chain", **metadata) as root:
            if langfuse_enabled():
                from langfuse import propagate_attributes

                with propagate_attributes(user_id=pseudonym(patient_ref), session_id=thread_id,
                                          trace_name=name, tags=["careflow"]):
                    yield root
            else:
                yield root
    finally:
        _current_thread.reset(thread_token)
        flush()


def traced_node(name: str, function, as_type: str = "span"):
    """Wrap a graph node so each run of it becomes a span."""

    def wrapper(state):
        with span(name, as_type=as_type) as observation:
            update = function(state)
            if isinstance(update, dict):
                observation.update(**{k: update[k] for k in ("route", "stop_reason")
                                      if isinstance(update.get(k), str)})
                audit = (update.get("audit") or [{}])[-1]
                details = {**audit.get("ehr", {}), **audit}
                observation.update(**{k: details[k] for k in TELEMETRY_KEYS
                                      if k in details and k != "route"})
                if details.get("fallbacks"):
                    observation.update(used_fallback=True)
                    observation.warn("clinic system degraded")
            return update

    wrapper.__name__ = name
    return wrapper