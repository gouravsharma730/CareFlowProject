"""The fault-injection harness is deterministic and hardening never hides failures."""

import logging

from evals.fault_injection import READ_TOOLS, build_workload, run_scenario


def _expected(workload):
    return [READ_TOOLS[tool](arguments) for tool, arguments in workload]


def test_same_seed_gives_the_same_numbers():
    logging.disable(logging.CRITICAL)
    workload = build_workload(size=60)
    expected = _expected(workload)
    first = run_scenario("outage", True, workload, expected)
    second = run_scenario("outage", True, workload, expected)
    logging.disable(logging.NOTSET)
    assert first == second


def test_hardening_helps_and_never_returns_a_wrong_or_unflagged_result():
    logging.disable(logging.CRITICAL)
    workload = build_workload(size=120)
    expected = _expected(workload)
    before = run_scenario("flaky", False, workload, expected)
    after = run_scenario("flaky", True, workload, expected)
    logging.disable(logging.NOTSET)
    assert after["success_rate"] > before["success_rate"]
    assert after["wrong_answers"] == 0 and after["failed_without_flag"] == 0