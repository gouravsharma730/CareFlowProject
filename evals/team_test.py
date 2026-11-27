"""End-to-end run of the agent team, every step in a NEW process.

  1. Insurance question  -> supervisor -> Insurance agent (MCP tools) -> reviewer -> answer
  2. Referral question   -> supervisor -> Referral agent (MCP tools)  -> reviewer -> answer
  3. Clinical question   -> reviewer pauses for a human (team-level interrupt)
  4. A new process resumes it with "approve" -> exactly one escalation

Evidence for M6.C1-C6: which agent ran, which tools it used (all through
the MCP server), the reviewer's verdicts, the MCP resources read, and the
pause and resume across processes.

Needs the Gemini key: about 8-14 LLM calls, paced by
CAREFLOW_LLM_MIN_SECONDS_BETWEEN_CALLS. Run:
    python -m evals.team_test

Results are saved to evals/results/team_test.json.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from careflow import config
from careflow.escalations import EscalationQueue
from careflow.team import INSURANCE_TOOLS, REFERRAL_TOOLS

RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "team_test.json"

CASES = {
    "insurance": ("MHP-P-00003", "Does an MRI of my knee need pre-approval on my plan, "
                                 "and roughly what would I pay?"),
    "referral": ("MHP-P-00085", "Hi, any update on my referral? When should I hear back?"),
    "clinical": ("MHP-P-00000", "I've had headaches since starting my new tablets. "
                                "Should I stop taking them?"),
}


def cli(*args: str) -> tuple[int, dict | str]:
    completed = subprocess.run(
        [sys.executable, "-m", "careflow.team", *args],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=900,
    )
    try:
        return completed.returncode, json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.returncode, (completed.stderr or completed.stdout)[-800:]


def events(output, name: str) -> list[dict]:
    if not isinstance(output, dict):
        return []
    return [a for a in output.get("audit", []) if a.get("event") == name]


def main() -> None:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    steps, checks = [], {}

    def step(name, *args):
        if steps:
            time.sleep(config.LLM_MIN_SECONDS_BETWEEN_CALLS)  # free-tier pacing
        code, output = cli(*args)
        steps.append({"step": name, "exit_code": code, "output": output})
        ok = isinstance(output, dict)
        print(f"{name}: exit={code} status={output.get('status') if ok else 'ERROR'} "
              f"stop={output.get('stop_reason') if ok else '-'}")
        return output

    for agent, allowed in (("insurance", INSURANCE_TOOLS), ("referral", REFERRAL_TOOLS)):
        patient, text = CASES[agent]
        out = step(f"{agent}_question", "start", "--thread", f"{run}-{agent}",
                   "--patient", patient, "--text", text)
        routed = events(out, "supervisor_routed")
        drafts = events(out, "draft_written")
        tools_used = {tool for d in drafts for tool in d.get("tools", [])}
        checks[f"{agent}_routed_to_{agent}_agent"] = bool(routed) and routed[0]["route"] == f"{agent}_agent"
        checks[f"{agent}_agent_used_tools"] = bool(tools_used)
        checks[f"{agent}_agent_used_only_its_own_tools"] = tools_used <= allowed
        checks[f"{agent}_reviewed_before_answer"] = bool(events(out, "review"))
        checks[f"{agent}_answered"] = isinstance(out, dict) and out.get("stop_reason") == "answered"

    patient, text = CASES["clinical"]
    thread = f"{run}-clinical"
    paused = step("clinical_question", "start", "--thread", thread,
                  "--patient", patient, "--channel", "sms", "--text", text)
    checks["clinical_paused_for_human"] = isinstance(paused, dict) and paused.get("status") == "paused"
    checks["no_escalation_before_approval"] = EscalationQueue().for_thread(thread) == []
    resumed = step("clinical_resume_approve_new_process", "resume", "--thread", thread,
                   "--decision", "approve", "--approver", "Nurse Rao",
                   "--reason", "Medication side-effect question needs a clinician")
    checks["resumed_in_a_different_process"] = (
        isinstance(paused, dict) and isinstance(resumed, dict)
        and paused.get("pid") != resumed.get("pid"))
    checks["clinical_escalated_exactly_once"] = (
        isinstance(resumed, dict) and resumed.get("stop_reason") == "escalated"
        and len(EscalationQueue().for_thread(thread)) == 1)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": config.LLM_MODEL,
        "checks": checks,
        "all_passed": all(checks.values()),
        "llm_requests_total": sum(s["output"].get("llm_requests", 0)
                                  for s in steps if isinstance(s["output"], dict)),
        "steps": steps,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\nChecks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    for s in steps:
        if isinstance(s["output"], dict) and s["output"].get("final_answer"):
            print(f"\n[{s['step']}] {s['output']['final_answer']}")
    print(f"\nLLM requests used: {report['llm_requests_total']}")
    print(f"All passed: {report['all_passed']}\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()