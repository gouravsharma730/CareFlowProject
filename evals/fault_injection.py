"""Fault injection: success rate before and after hardening (M7.C6).

The same seeded workload of clinic-system calls is run twice through the
same seeded faults:

  before        a plain client: one attempt per call, no retry, no breaker
  retries_only  ResilientEHR with the breaker disabled (for comparison)
  after         ResilientEHR: retries with backoff on temporary errors, a
                circuit breaker, and a flagged fallback

A cooldown sweep (3 seeds x 4 cooldowns) is saved too; it is how the
default CAREFLOW_BREAKER_RESET_SECONDS was chosen.

Faults are injected per attempt between CareFlow and the clinic system
(the real tool functions that the MCP server wraps):
  flaky    25% of attempts fail (unreachable or timeout)
  outage   10% of attempts fail, and the system is fully down for calls
           100-159 (about a minute of simulated time)

A simulated clock (1 s per request, plus any backoff waits) drives the
circuit breaker, so the run takes seconds and makes no LLM calls.
Same seed, same numbers.

Run:
    python -m evals.fault_injection

Results are saved to evals/results/fault_injection.json.
"""

import json
import logging
import random
from datetime import datetime, timezone

from careflow import config, tools
from careflow.mock_data import load_table
from careflow.reliability import CircuitBreaker, ResilientEHR

RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "fault_injection.json"
SEED = 7
CALLS = 300

SCENARIOS = {
    "flaky": {"fault_rate": 0.25, "outage": None},
    "outage": {"fault_rate": 0.10, "outage": (100, 160)},
}

READ_TOOLS = {
    "check_insurance_eligibility": lambda a: tools.check_insurance_eligibility(a["patient_ref"]),
    "check_preauthorization": lambda a: tools.check_preauthorization(a["patient_ref"], a["procedure_code"]),
    "calculate_copay": lambda a: tools.calculate_copay(a["patient_ref"], a["procedure_code"]),
    "get_referral_status": lambda a: tools.get_referral_status(a["patient_ref"]),
}


def build_workload(seed: int = SEED, size: int = CALLS) -> list[tuple[str, dict]]:
    """Realistic read calls for real patients and procedures."""
    rng = random.Random(seed)
    patients = [p["patient_ref"] for p in load_table("patients")][:100]
    procedures = ["ORTH-MRI", "CARD-ECHO", "GAST-COLO", "NEUR-MRI", "CARD-CONS", "DERM-EXCI"]
    workload = []
    for _ in range(size):
        tool = rng.choice(list(READ_TOOLS))
        arguments = {"patient_ref": rng.choice(patients)}
        if tool in ("check_preauthorization", "calculate_copay"):
            arguments["procedure_code"] = rng.choice(procedures)
        workload.append((tool, arguments))
    return workload


class SimulatedClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FaultyEHR:
    """The clinic system with seeded faults injected on every attempt."""

    def __init__(self, fault_rate: float, outage: tuple[int, int] | None, seed: int):
        self.rng = random.Random(seed)
        self.fault_rate = fault_rate
        self.outage = outage
        self.request_index = 0
        self.attempts = 0

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.attempts += 1
        roll = self.rng.random()
        down = self.outage and self.outage[0] <= self.request_index < self.outage[1]
        if down or roll < self.fault_rate:
            code = "EHR_TIMEOUT" if roll < self.fault_rate * 0.2 else "MCP_UNAVAILABLE"
            return {"success": False, "error_code": code, "message": "injected fault"}
        return READ_TOOLS[name](arguments)


def run_scenario(name: str, hardened: bool, workload, expected, seed: int = SEED,
                 reset_seconds: float | None = None, use_breaker: bool = True) -> dict:
    spec = SCENARIOS[name]
    clock = SimulatedClock()
    system = FaultyEHR(spec["fault_rate"], spec["outage"], seed=seed)
    breaker = CircuitBreaker(f"{name}-test", clock=clock, reset_seconds=reset_seconds,
                             failure_threshold=None if use_breaker else 10**9)
    client = ResilientEHR(system, breaker=breaker, sleep=clock.sleep) if hardened else system

    correct = wrong = flagged_failures = unflagged_failures = 0
    for index, ((tool, arguments), truth) in enumerate(zip(workload, expected)):
        system.request_index = index
        clock.now += 1.0  # one request per simulated second
        result = client.call_tool(tool, arguments)
        if result == truth:
            correct += 1
        elif result.get("success"):
            wrong += 1  # a successful-looking but different answer: must never happen
        elif result.get("used_fallback"):
            flagged_failures += 1
        else:
            unflagged_failures += 1
    report = {
        "success_rate": round(correct / len(workload), 3),
        "correct": correct,
        "wrong_answers": wrong,
        "failed_with_used_fallback_flag": flagged_failures,
        "failed_without_flag": unflagged_failures,
        "attempts_sent_to_clinic_system": system.attempts,
    }
    if hardened:
        report.update({
            "retries": client.stats["retries"],
            "fast_failed_by_open_breaker": client.stats["fast_failed"],
            "breaker_opened": breaker.times_opened,
        })
    return report


def cooldown_sweep(seeds=(7, 11, 23), cooldowns=(5, 10, 15, 30)) -> list[dict]:
    """Outage success rate and load for several breaker cooldowns and seeds."""
    rows = []
    for seed in seeds:
        workload = build_workload(seed)
        expected = [READ_TOOLS[tool](arguments) for tool, arguments in workload]
        for cooldown in cooldowns:
            r = run_scenario("outage", True, workload, expected, seed, reset_seconds=cooldown)
            rows.append({"seed": seed, "cooldown_seconds": cooldown,
                         "success_rate": r["success_rate"],
                         "attempts": r["attempts_sent_to_clinic_system"]})
        r = run_scenario("outage", True, workload, expected, seed, use_breaker=False)
        rows.append({"seed": seed, "cooldown_seconds": None, "success_rate": r["success_rate"],
                     "attempts": r["attempts_sent_to_clinic_system"], "note": "no breaker"})
    return rows


def main() -> dict:
    logging.basicConfig(level=logging.ERROR)  # the run triggers many expected warnings
    workload = build_workload()
    expected = [READ_TOOLS[tool](arguments) for tool, arguments in workload]
    results = {
        name: {"before": run_scenario(name, False, workload, expected),
               "retries_only": run_scenario(name, True, workload, expected, use_breaker=False),
               "after": run_scenario(name, True, workload, expected)}
        for name in SCENARIOS
    }
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "calls_per_scenario": CALLS,
        "scenarios": SCENARIOS,
        "settings": {
            "max_retries": config.EHR_MAX_RETRIES,
            "retry_base_seconds": config.EHR_RETRY_BASE_SECONDS,
            "breaker_failure_threshold": config.BREAKER_FAILURE_THRESHOLD,
            "breaker_reset_seconds": config.BREAKER_RESET_SECONDS,
            "stale_fallback": config.EHR_STALE_FALLBACK,
        },
        "results": results,
        "cooldown_sweep": cooldown_sweep(),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'scenario':10}{'':14}{'success':>9}{'wrong':>7}{'flagged':>9}"
          f"{'unflagged':>11}{'attempts':>10}{'fast-fail':>11}{'opened':>8}")
    for name, pair in results.items():
        for label in ("before", "retries_only", "after"):
            r = pair[label]
            print(f"{name:10}{label:14}{r['success_rate']:>9}{r['wrong_answers']:>7}"
                  f"{r['failed_with_used_fallback_flag']:>9}{r['failed_without_flag']:>11}"
                  f"{r['attempts_sent_to_clinic_system']:>10}"
                  f"{r.get('fast_failed_by_open_breaker', '-'):>11}{r.get('breaker_opened', '-'):>8}")
    print("\nBreaker cooldown sweep (outage scenario):")
    for row in report["cooldown_sweep"]:
        label = row.get("note") or f"cooldown {row['cooldown_seconds']}s"
        print(f"  seed {row['seed']:>2}  {label:14} success {row['success_rate']:<6}"
              f" attempts {row['attempts']}")
    print(f"\nSaved to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    main()