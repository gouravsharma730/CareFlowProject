# Observability and reliability (M7)

Code: `careflow/tracing.py`, `careflow/reliability.py`, `careflow/llm_client.py`.
Tests: `tests/test_tracing.py`, `tests/test_reliability.py`, `tests/test_llm_resilience.py`,
`tests/test_kpi_report.py`, `tests/test_fault_injection.py`.
Evidence: `evals/results/kpi_report.json`, `evals/results/fault_injection.json`,
`evals/results/team_test.json`.

## Tracing (Langfuse)

One request = one trace, with nested spans:

```
careflow-team                     session = thread_id, user = hashed patient id
├── intake            [agent]  └── llm.completion [generation: model, tokens, cost]
├── supervisor        [agent]
├── insurance_agent   [agent]  ├── llm.completion [generation]
│                              ├── tool.check_preauthorization [tool]
│                              └── tool.calculate_copay        [tool]
├── safety_review     [agent]  └── llm.completion [generation]
└── finalize
```

- Every LLM call is a generation with model name, input and output tokens, and
  cost (LiteLLM list price; the free tier is not charged).
- Traces carry a session id (the workflow thread_id; a paused and resumed
  request shares one session) and a user id.
- **Privacy:** the user id is a salted hash (`pt-…`). No names, patient
  references, messages, prompts or replies are sent to the tracer, only
  metadata (routes, tool names, error codes, token counts, timings).
  `tests/test_tracing.py` checks this.
- `flush()` runs at the end of every request and at process exit.
- Agent, service and API code logs with `logging`. The only `print()` calls are
  in command-line demo entry points.
- Tracing turns on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
  (see `.env.example`). Tests use a real Langfuse client that exports to memory,
  so they never send traces.

## KPI view

`python -m evals.kpi_report` reads the local span log (`data/telemetry/events.jsonl`)
and reports per agent and per request: runs, latency p50 and p95, LLM calls,
tokens, cost, tool calls, tool error rate and fallback rate.

Live team run (3 requests, `evals/results/kpi_report.json`):

| Agent | LLM calls | Tokens in / out | Cost (list price) |
|---|---|---|---|
| intake | 3 | 1,499 / 382 | $0.0014 |
| insurance_agent | 2 | 2,303 / 137 | $0.0010 |
| referral_agent | 2 | 2,221 / 88 | $0.0009 |
| safety_review | 2 | 3,301 / 36 | $0.0011 |
| supervisor | 0 (routed by rule) | 0 | $0 |
| **Total** | **9** | **9,967** | **$0.0044** |

Error rate 0, fallback rate 0. Note: most of the agent latency in this run
(about 13 s per specialist) is the deliberate free-tier pacing of 7 s between
LLM calls (`CAREFLOW_LLM_MIN_SECONDS_BETWEEN_CALLS`), not system slowness.

## Reliability of external calls

| Protection | Clinic system (MCP) | LLM (Gemini via LiteLLM) |
|---|---|---|
| Timeout | `CAREFLOW_EHR_CALL_TIMEOUT_SECONDS` (10 s) | `CAREFLOW_LLM_TIMEOUT_SECONDS` (30 s) |
| Retries, exponential backoff, temporary errors only | yes (business answers such as "patient not found" are never retried; write tools are never retried) | yes (a bad API key or model name is never retried) |
| Circuit breaker (closed, open, half-open) | yes, shared per process | yes, shared per process |
| Flagged fallback | `EHR_UNAVAILABLE` with `used_fallback: true`; optional stale cache, off by default | optional backup model `CAREFLOW_LLM_FALLBACK_MODEL`, off by default, flagged `used_fallback` |
| When everything fails | the agent says the system is unavailable | intake, agents and reviewer hand the request to staff; the reviewer never approves an unchecked draft |

The breaker counts failed **calls** (after retries), not individual attempts.
The first fault-injection run showed that counting attempts let ordinary
flakiness keep the breaker open, which dropped success to 0.30.

## Failure injection

`python -m evals.fault_injection`: 300 seeded clinic-system calls through seeded
faults, same seed and same numbers on every run, no LLM calls.

| Scenario | Before (no hardening) | After |
|---|---|---|
| Flaky (25% of attempts fail) | 0.727 success, 82 failures unflagged | **0.967**, 0 wrong answers, every failure flagged |
| Outage (system down for 60 calls) | 0.73, 81 unflagged | **0.793** (best possible 0.80), 38% fewer calls to the dead system than retries alone |

The breaker cooldown (`CAREFLOW_BREAKER_RESET_SECONDS` = 10 s) was chosen from a
sweep over 3 seeds and 4 cooldowns. 30 s blocked too many requests after the
system had recovered. The sweep is saved in the results file.

## Degradation in traces

Agent spans carry the clinic-system retries, fallbacks and breaker state, and
are marked WARNING when a fallback was used. Tool spans carry the error code
and `used_fallback`. Fallback-model generations are marked WARNING.