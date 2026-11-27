"""Tests for the circuit breaker and the resilient clinic-system client."""

import pytest

from careflow import config
from careflow.reliability import CircuitBreaker, EHRUnavailable, ResilientEHR

DOWN = {"success": False, "error_code": "MCP_UNAVAILABLE", "message": "down"}
OK = {"success": True, "plan_code": "MERIDIAN-GOLD"}
NOT_FOUND = {"success": False, "error_code": "PATIENT_NOT_FOUND", "message": "no"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ScriptedEHR:
    """Returns the scripted results in order (the last one repeats)."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def call_tool(self, name, arguments):
        self.calls += 1
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def read_resource(self, uri):
        self.calls += 1
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if result is DOWN:
            raise ConnectionError("down")
        return "SOP text"


@pytest.fixture
def breaker():
    return CircuitBreaker("test", failure_threshold=3, reset_seconds=30, clock=Clock())


def resilient(ehr, breaker):
    return ResilientEHR(ehr, breaker=breaker, sleep=lambda seconds: None)


# ---- circuit breaker ----

def test_breaker_opens_after_repeated_failures_then_half_opens(breaker):
    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow()
    breaker.clock.now = 31
    assert breaker.state == "half_open" and breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.times_opened == 1


def test_failed_trial_call_reopens_the_breaker(breaker):
    for _ in range(3):
        breaker.record_failure()
    breaker.clock.now = 31
    assert breaker.state == "half_open"
    breaker.record_failure()
    assert breaker.state == "open" and breaker.times_opened == 2


def test_a_success_resets_the_failure_count(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"


# ---- resilient client ----

def test_temporary_failures_are_retried(breaker):
    ehr = ScriptedEHR(DOWN, DOWN, OK)
    result = resilient(ehr, breaker).call_tool("check_insurance_eligibility", {"patient_ref": "p"})
    assert result == OK and ehr.calls == 3


def test_retry_waits_grow_exponentially(breaker):
    waits = []
    client = ResilientEHR(ScriptedEHR(DOWN, DOWN, OK), breaker=breaker, sleep=waits.append)
    client.call_tool("check_insurance_eligibility", {})
    assert waits == [config.EHR_RETRY_BASE_SECONDS, config.EHR_RETRY_BASE_SECONDS * 2]


def test_business_errors_are_not_retried(breaker):
    ehr = ScriptedEHR(NOT_FOUND)
    assert resilient(ehr, breaker).call_tool("check_insurance_eligibility", {}) == NOT_FOUND
    assert ehr.calls == 1 and breaker.state == "closed"


def test_write_tools_are_never_retried(breaker):
    ehr = ScriptedEHR(DOWN, OK)
    result = resilient(ehr, breaker).call_tool("flag_for_human", {"reason": "x"})
    assert result["error_code"] == "EHR_UNAVAILABLE" and ehr.calls == 1


def test_exhausted_retries_give_a_flagged_fallback(breaker):
    result = resilient(ScriptedEHR(DOWN), breaker).call_tool("check_insurance_eligibility", {})
    assert result["success"] is False
    assert result["error_code"] == "EHR_UNAVAILABLE" and result["used_fallback"] is True


def test_flaky_attempts_alone_do_not_open_the_breaker(breaker):
    client = resilient(ScriptedEHR(DOWN, DOWN, OK, DOWN, DOWN, OK, DOWN, DOWN, OK), breaker)
    for _ in range(3):
        assert client.call_tool("check_insurance_eligibility", {}) == OK
    assert breaker.state == "closed" and breaker.times_opened == 0


def test_open_breaker_fails_fast_without_calling_the_system(breaker):
    client = resilient(ScriptedEHR(DOWN), breaker)
    for _ in range(3):  # three calls, each failing after its retries, open it
        client.call_tool("check_insurance_eligibility", {})
    assert breaker.state == "open"
    calls_before = client.ehr.calls
    result = client.call_tool("check_insurance_eligibility", {})
    assert result["fallback_reason"] == "circuit_open"
    assert client.ehr.calls == calls_before and client.stats["fast_failed"] == 1


def test_stale_cache_fallback_is_off_by_default_and_flagged_when_on(breaker, monkeypatch):
    client = resilient(ScriptedEHR(OK, DOWN), breaker)
    client.call_tool("check_insurance_eligibility", {"patient_ref": "p"})
    assert "plan_code" not in client.call_tool("check_insurance_eligibility", {"patient_ref": "p"})

    monkeypatch.setattr(config, "EHR_STALE_FALLBACK", True)
    breaker.reset()
    client = resilient(ScriptedEHR(OK, DOWN), breaker)
    client.call_tool("check_insurance_eligibility", {"patient_ref": "p"})
    stale = client.call_tool("check_insurance_eligibility", {"patient_ref": "p"})
    assert stale["plan_code"] == "MERIDIAN-GOLD"
    assert stale["used_fallback"] is True and stale["stale"] is True


def test_resource_reads_are_retried_then_raise(breaker):
    assert resilient(ScriptedEHR(DOWN, "ok"), breaker).read_resource("policy://x") == "SOP text"
    breaker.reset()
    with pytest.raises(EHRUnavailable):
        resilient(ScriptedEHR(DOWN), breaker).read_resource("policy://x")


def test_degradation_tags(breaker):
    client = resilient(ScriptedEHR(DOWN, OK), breaker)
    client.call_tool("check_insurance_eligibility", {})
    tags = client.degradation()
    assert tags["retries"] == 1 and tags["breaker_state"] == "closed" and tags["fallbacks"] == 0


def test_failed_half_open_trial_is_not_retried(breaker):
    client = resilient(ScriptedEHR(DOWN), breaker)
    for _ in range(3):
        client.call_tool("check_insurance_eligibility", {})
    breaker.clock.now = 31  # cooldown over: half-open
    calls_before = client.ehr.calls
    client.call_tool("check_insurance_eligibility", {})
    assert client.ehr.calls == calls_before + 1  # one trial, no retries
    assert breaker.state == "open"