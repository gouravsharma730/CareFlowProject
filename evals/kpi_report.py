"""KPI view: tokens, cost, latency, errors and fallbacks per agent and per run (M7.C4).

Reads the local telemetry log written by careflow/tracing.py (one line per
span; the same spans go to Langfuse when it is configured) and computes:

  per agent  runs, latency p50/p95, LLM calls, tokens in/out, cost,
             tool calls, tool error rate, fallback rate
  per run    one row per request thread: agents used, LLM calls, tokens,
             cost, total latency, errors, fallbacks
  overall    totals and rates

Costs are LiteLLM list prices for the configured model (on a free tier
the real charge is $0).

Run after any live run (for example python -m evals.team_test):
    python -m evals.kpi_report

Results are saved to evals/results/kpi_report.json.
"""

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from careflow import config

RESULTS_PATH = config.PROJECT_ROOT / "evals" / "results" / "kpi_report.json"


def load_events(path=None) -> list[dict]:
    path = path or config.TELEMETRY_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 1)


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def compute(events: list[dict]) -> dict:
    agents = defaultdict(lambda: {"runs": 0, "latencies": [], "llm_calls": 0, "tokens_in": 0,
                                  "tokens_out": 0, "cost_usd": 0.0, "tool_calls": 0,
                                  "tool_errors": 0, "fallbacks": 0})
    runs = defaultdict(lambda: {"agents": set(), "llm_calls": 0, "tokens": 0, "cost_usd": 0.0,
                                "latency_ms": 0.0, "errors": 0, "fallbacks": 0, "steps": 0})
    for event in events:
        agent = event.get("agent") or "(no agent)"
        run = runs[event.get("thread_id") or "(outside a request)"]
        if event["type"] == "agent":
            agents[agent]["runs"] += 1
            agents[agent]["latencies"].append(event["duration_ms"])
            run["agents"].add(agent)
        elif event["type"] == "generation":
            a = agents[agent]
            a["llm_calls"] += 1
            a["tokens_in"] += event.get("tokens_in") or 0
            a["tokens_out"] += event.get("tokens_out") or 0
            a["cost_usd"] += event.get("cost_usd") or 0.0
            run["llm_calls"] += 1
            run["tokens"] += (event.get("tokens_in") or 0) + (event.get("tokens_out") or 0)
            run["cost_usd"] += event.get("cost_usd") or 0.0
        elif event["type"] == "tool":
            agents[agent]["tool_calls"] += 1
            if event.get("level") in ("WARNING", "ERROR"):
                agents[agent]["tool_errors"] += 1
            if event.get("used_fallback"):
                agents[agent]["fallbacks"] += 1
                run["fallbacks"] += 1
        elif event["type"] == "chain":
            run["latency_ms"] += event["duration_ms"]
            run["steps"] += 1
        if event.get("level") == "ERROR":
            run["errors"] += 1

    agent_rows = {
        name: {
            "runs": a["runs"],
            "latency_p50_ms": _percentile(a["latencies"], 0.5),
            "latency_p95_ms": _percentile(a["latencies"], 0.95),
            "llm_calls": a["llm_calls"],
            "tokens_in": a["tokens_in"],
            "tokens_out": a["tokens_out"],
            "cost_usd": round(a["cost_usd"], 6),
            "tool_calls": a["tool_calls"],
            "tool_error_rate": _rate(a["tool_errors"], a["tool_calls"]),
            "fallback_rate": _rate(a["fallbacks"], a["tool_calls"]),
        }
        for name, a in sorted(agents.items())
    }
    run_rows = {
        thread: {**{k: v for k, v in r.items() if k != "agents"},
                 "agents": sorted(r["agents"]), "cost_usd": round(r["cost_usd"], 6),
                 "latency_ms": round(r["latency_ms"], 1)}
        for thread, r in runs.items() if r["steps"]
    }
    total_tools = sum(a["tool_calls"] for a in agents.values())
    overall = {
        "requests": len(run_rows),
        "llm_calls": sum(a["llm_calls"] for a in agents.values()),
        "tokens": sum(a["tokens_in"] + a["tokens_out"] for a in agents.values()),
        "cost_usd": round(sum(a["cost_usd"] for a in agents.values()), 6),
        "mean_request_latency_ms": round(statistics.mean(
            r["latency_ms"] for r in run_rows.values()), 1) if run_rows else None,
        "requests_with_errors_rate": _rate(
            sum(1 for r in run_rows.values() if r["errors"]), len(run_rows)),
        "tool_fallback_rate": _rate(sum(a["fallbacks"] for a in agents.values()), total_tools),
    }
    return {"overall": overall, "per_agent": agent_rows, "per_run": run_rows}


def main() -> dict:
    events = load_events()
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(config.TELEMETRY_PATH),
        "events": len(events),
        "model": config.LLM_MODEL,
        "cost_note": "LiteLLM list price for the model; the free tier is not charged.",
        **compute(events),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Overall: {report['overall']}\n")
    header = (f"{'agent':18}{'runs':>5}{'p50 ms':>9}{'p95 ms':>9}{'LLM':>5}"
              f"{'tok in':>8}{'tok out':>8}{'cost $':>10}{'tools':>6}{'err':>6}{'fallbk':>7}")
    print(header)
    for name, a in report["per_agent"].items():
        print(f"{name:18}{a['runs']:>5}{str(a['latency_p50_ms']):>9}{str(a['latency_p95_ms']):>9}"
              f"{a['llm_calls']:>5}{a['tokens_in']:>8}{a['tokens_out']:>8}{a['cost_usd']:>10.6f}"
              f"{a['tool_calls']:>6}{str(a['tool_error_rate']):>6}{str(a['fallback_rate']):>7}")
    print(f"\n{'run (thread)':34}{'LLM':>5}{'tokens':>8}{'cost $':>10}{'ms':>9}{'err':>5}  agents")
    for thread, r in report["per_run"].items():
        print(f"{thread[:34]:34}{r['llm_calls']:>5}{r['tokens']:>8}{r['cost_usd']:>10.6f}"
              f"{r['latency_ms']:>9}{r['errors']:>5}  {', '.join(r['agents'])}")
    print(f"\nSaved to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    main()