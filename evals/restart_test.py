"""Pause-kill-resume test for the workflow (rubric Section 7, M5.C4/C5).

Every step runs `python -m careflow.workflow ...` in a NEW process, so
nothing survives in memory between steps; only the SQLite checkpoint
file and the mock clinic database carry state across.

  1. A clinical request pauses and returns an approval question.
  2. A new process resumes it with "approve": it completes and creates
     exactly one escalation.
  3. Resuming the same thread again is refused, and still one escalation.
  4. A second clinical request is resumed with "reject": no escalation.
  5. A low-risk request completes without pausing.

Needs the Gemini key (real intake and agent): about 6-10 LLM calls in all.
Each step is a new process, so the test also waits
CAREFLOW_LLM_MIN_SECONDS_BETWEEN_CALLS between steps to respect the
free-tier per-minute limit. Run:
    python -m evals.restart_test

Results are saved to evals/results/restart_test.json.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from careflow import config
from careflow.escalations import EscalationQueue

RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "restart_test.json"
PATIENT = "MHP-P-00000"
CLINICAL_TEXT = ("I've felt dizzy since I started the new blood pressure tablets. "
                 "Should I stop taking them?")
LOW_RISK_TEXT = "Can you check whether my insurance is active?"


def cli(*args: str) -> tuple[int, dict | str]:
    """Run the workflow command line in a fresh process."""
    completed = subprocess.run(
        [sys.executable, "-m", "careflow.workflow", *args],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )
    try:
        return completed.returncode, json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.returncode, (completed.stderr or completed.stdout)[-500:]


def main() -> None:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    approve_thread, reject_thread, low_thread = (f"{run}-{x}" for x in ("approve", "reject", "lowrisk"))
    queue = EscalationQueue()
    steps, checks = [], {}

    def step(name, *args):
        if steps:
            time.sleep(config.LLM_MIN_SECONDS_BETWEEN_CALLS)  # free-tier pacing
        code, output = cli(*args)
        steps.append({"step": name, "exit_code": code, "output": output})
        print(f"{name}: exit={code} status={output.get('status') if isinstance(output, dict) else 'ERROR'}"
              f" pid={output.get('pid') if isinstance(output, dict) else '-'}")
        return code, output

    _, paused = step("1_start_clinical", "start", "--thread", approve_thread,
                     "--patient", PATIENT, "--channel", "sms", "--text", CLINICAL_TEXT)
    checks["clinical_request_paused"] = isinstance(paused, dict) and paused.get("status") == "paused"
    checks["interrupt_explains_the_approval"] = isinstance(paused, dict) and bool(
        (paused.get("interrupt") or {}).get("proposed_action"))
    checks["no_escalation_before_approval"] = queue.for_thread(approve_thread) == []

    _, resumed = step("2_resume_approve_new_process", "resume", "--thread", approve_thread,
                      "--decision", "approve", "--approver", "Nurse Rao",
                      "--reason", "Medication side-effect question needs a clinician")
    checks["resumed_in_a_different_process"] = (
        isinstance(paused, dict) and isinstance(resumed, dict)
        and paused.get("pid") != resumed.get("pid"))
    checks["approved_run_completed"] = isinstance(resumed, dict) and resumed.get("stop_reason") == "escalated"
    checks["exactly_one_escalation_after_approve"] = len(queue.for_thread(approve_thread)) == 1

    code, _ = step("3_resume_again_is_refused", "resume", "--thread", approve_thread,
                   "--decision", "approve", "--approver", "Nurse Rao", "--reason", "Second try")
    checks["second_resume_refused"] = code != 0
    checks["still_exactly_one_escalation"] = len(queue.for_thread(approve_thread)) == 1

    step("4a_start_clinical_for_reject", "start", "--thread", reject_thread,
         "--patient", PATIENT, "--channel", "sms", "--text", CLINICAL_TEXT)
    _, rejected = step("4b_resume_reject_new_process", "resume", "--thread", reject_thread,
                       "--decision", "reject", "--approver", "Nurse Rao",
                       "--reason", "Already handled by phone")
    checks["rejected_run_completed"] = isinstance(rejected, dict) and rejected.get("stop_reason") == "escalation_rejected"
    checks["no_escalation_after_reject"] = queue.for_thread(reject_thread) == []

    _, low = step("5_low_risk_request", "start", "--thread", low_thread,
                  "--patient", PATIENT, "--channel", "web_form", "--text", LOW_RISK_TEXT)
    checks["low_risk_did_not_pause"] = isinstance(low, dict) and low.get("status") == "done"

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_db": str(config.CHECKPOINT_DB_PATH.relative_to(config.PROJECT_ROOT)),
        "threads": {"approve": approve_thread, "reject": reject_thread, "low_risk": low_thread},
        "checks": checks,
        "all_passed": all(checks.values()),
        "llm_requests_total": sum(
            s["output"].get("llm_requests", 0) for s in steps if isinstance(s["output"], dict)),
        "steps": steps,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\nChecks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nLLM requests used: {report['llm_requests_total']}")
    print(f"All passed: {report['all_passed']}\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()