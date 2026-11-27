"""Reliability for calls to the clinic system (the MCP server).

ResilientEHR wraps an MCP session and adds, for every tool call:
  - retries with exponential backoff, ONLY for temporary failures
    (server unreachable, timeout). Business answers such as
    PATIENT_NOT_FOUND are real results and are never retried. Write
    tools are never retried, so a hand-off cannot be recorded twice.
  - a circuit breaker shared by the whole process: after
    BREAKER_FAILURE_THRESHOLD failed calls in a row (each already retried)
    it OPENS and fails fast;
    after BREAKER_RESET_SECONDS it goes HALF-OPEN and lets one trial call
    through; a success CLOSES it again, a failure re-opens it.
  - a flagged fallback: when the system cannot be reached the caller gets
    {"success": False, "error_code": "EHR_UNAVAILABLE", "used_fallback": True}
    (or, only if CAREFLOW_EHR_STALE_FALLBACK=true, the last good answer
    marked "stale"). Never a crash and never an invented answer.

Timeouts are applied by the MCP client (CAREFLOW_EHR_CALL_TIMEOUT_SECONDS).
"""

import logging
import threading
import time

from careflow import config

logger = logging.getLogger(__name__)

TRANSIENT_ERROR_CODES = {"MCP_UNAVAILABLE", "EHR_TIMEOUT"}
WRITE_TOOLS = {"flag_for_human"}  # never retried, never served from cache


class EHRUnavailable(Exception):
    """The clinic system could not be reached (used for resource reads)."""


class CircuitBreaker:
    """Closed -> open after repeated failures -> half-open trial -> closed."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, failure_threshold: int | None = None,
                 reset_seconds: float | None = None, clock=time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold or config.BREAKER_FAILURE_THRESHOLD
        self.reset_seconds = reset_seconds if reset_seconds is not None else config.BREAKER_RESET_SECONDS
        self.clock = clock
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self.times_opened = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> str:
        if self._state == self.OPEN and self.clock() - self._opened_at >= self.reset_seconds:
            self._state = self.HALF_OPEN
            logger.info("Circuit %s half-open: allowing a trial call", self.name)
        return self._state

    def allow(self) -> bool:
        """May a call go through right now?"""
        with self._lock:
            return self._current_state() != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state != self.CLOSED:
                logger.info("Circuit %s closed again", self.name)
            self._state, self._failures = self.CLOSED, 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            state = self._current_state()
            if state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                if state != self.OPEN:
                    self.times_opened += 1
                    logger.warning("Circuit %s opened after %d failure(s)", self.name, self._failures)
                self._state, self._opened_at = self.OPEN, self.clock()

    def reset(self) -> None:
        with self._lock:
            self._state, self._failures, self._opened_at = self.CLOSED, 0, 0.0


# One breaker for the clinic system, shared by every agent in this process.
EHR_BREAKER = CircuitBreaker("clinic-ehr")


class ResilientEHR:
    """An MCP session with retries, a circuit breaker and a flagged fallback."""

    def __init__(self, ehr, breaker: CircuitBreaker | None = None, sleep=time.sleep):
        self.ehr = ehr
        self.breaker = breaker or EHR_BREAKER
        self.sleep = sleep
        self.cache: dict = {}
        self.stats = {"calls": 0, "attempts": 0, "retries": 0, "fallbacks": 0,
                      "fast_failed": 0}

    # tool_schemas and anything else pass straight through
    def __getattr__(self, name):
        return getattr(self.ehr, name)

    def _fallback(self, name: str, arguments: dict, reason: str) -> dict:
        self.stats["fallbacks"] += 1
        key = (name, tuple(sorted(arguments.items())))
        if config.EHR_STALE_FALLBACK and name not in WRITE_TOOLS and key in self.cache:
            logger.warning("Serving a stale %s result (%s)", name, reason)
            return {**self.cache[key], "used_fallback": True, "stale": True,
                    "fallback_reason": reason}
        logger.warning("Clinic system unavailable for %s (%s)", name, reason)
        return {"success": False, "error_code": "EHR_UNAVAILABLE", "used_fallback": True,
                "fallback_reason": reason,
                "message": "The clinic system is temporarily unavailable. "
                           "A staff member will follow up."}

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call one tool with retries; the breaker counts failed CALLS, not attempts.

        Retries absorb short flakiness. Only a call that still fails after
        all its retries counts toward opening the breaker, which is meant
        for real outages. A failed half-open trial re-opens it at once.
        """
        self.stats["calls"] += 1
        retries = 0 if name in WRITE_TOOLS else config.EHR_MAX_RETRIES
        for attempt in range(retries + 1):
            if not self.breaker.allow():
                self.stats["fast_failed"] += 1
                return self._fallback(name, arguments, "circuit_open")
            trial = self.breaker.state == CircuitBreaker.HALF_OPEN
            self.stats["attempts"] += 1
            result = self.ehr.call_tool(name, arguments)
            if result.get("error_code") not in TRANSIENT_ERROR_CODES:
                self.breaker.record_success()  # the system answered (even "not found")
                if result.get("success") and name not in WRITE_TOOLS:
                    self.cache[(name, tuple(sorted(arguments.items())))] = result
                return result
            if trial:
                break  # the trial call failed: re-open without more retries
            if attempt < retries:
                self.stats["retries"] += 1
                wait = config.EHR_RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning("Temporary clinic-system error on %s (%s); retry %d in %.1fs",
                               name, result["error_code"], attempt + 1, wait)
                self.sleep(wait)
        self.breaker.record_failure()
        return self._fallback(name, arguments, "retries_exhausted")

    def read_resource(self, uri: str) -> str:
        """Read a resource with the same retry and breaker rules."""
        last_error = None
        for attempt in range(config.EHR_MAX_RETRIES + 1):
            if not self.breaker.allow():
                raise EHRUnavailable(f"circuit open, cannot read {uri}")
            trial = self.breaker.state == CircuitBreaker.HALF_OPEN
            try:
                text = self.ehr.read_resource(uri)
            except Exception as error:
                last_error = error
                if trial:
                    break
                if attempt < config.EHR_MAX_RETRIES:
                    self.sleep(config.EHR_RETRY_BASE_SECONDS * (2 ** attempt))
                continue
            self.breaker.record_success()
            return text
        self.breaker.record_failure()
        raise EHRUnavailable(f"cannot read {uri}: {type(last_error).__name__}")

    def degradation(self) -> dict:
        """Tags for the audit log and traces: what went wrong, if anything."""
        return {**self.stats, "breaker_state": self.breaker.state}