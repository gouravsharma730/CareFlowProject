# CareFlow Architecture

This document describes how CareFlow is built, how a request flows through it, how it behaves when things fail, how it was evaluated, and what its known gaps are.

> All data in this project is synthetic.

---

## 1. Goals and non-goals

**Goals**
- Answer administrative patient requests (coverage, cost estimates, pre-authorisation, referrals, appointments) accurately and only from real data or policy text.
- Never give medical advice. Route clinical questions to a licensed clinician, with a human approval step.
- Detect emergencies and respond immediately, without an LLM in the path.
- Stay safe and honest when a dependency fails: a flagged "unavailable" message, never a guessed answer.
- Be measurable: every claim about quality is backed by an eval in `evals/results/`.

**Non-goals**
- Diagnosis, triage of symptoms, medication or dosing advice (SOP-401 forbids them).
- Writing to real clinical systems. The clinic system is a mock served over MCP.

---

## 2. Component overview

```
                       ┌─────────────────────────────────────┐
  HTTP client ───────► │ FastAPI  careflow/api.py            │  /chat  /approvals/{id}  /health
  CLI ───────────────► │ careflow/team.py (start|resume|status)
                       └──────────────┬──────────────────────┘
                                      ▼
                ┌──────────── LangGraph agent team (careflow/team.py) ────────────┐
                │ input_guard ─┬─► blocked                                        │
                │              ├─► emergency                                      │
                │              └─► intake ─► supervisor ─┬─► insurance_agent ─┐   │
                │                                        └─► referral_agent ──┤   │
                │                         safety_review (critic) ◄────────────┘   │
                │                              │ revise (max 2) / approve         │
                │                              ▼                                  │
                │                          finalize ─► output_guard ─► reply      │
                │ clinical request ─► human_approval (interrupt) ─► escalate      │
                └──────┬───────────────────┬────────────────────┬─────────────────┘
                       │                   │                    │
           SQLite checkpointer      MCP client (stdio)      LLM client (LiteLLM)
           (careflow_graph.sqlite)        │                     │ pacing, retries,
           escalation queue               ▼                     │ circuit breaker,
           (careflow_ops.sqlite)   careflow-ehr MCP server       │ optional fallback model
                                   6 tools + policy resources   ▼
                                   + SOP prompt             Gemini (flash-lite)
                                          │
                                   ResilientEHR wrapper: retries, breaker, flagged fallback

        Policy retrieval (careflow/rag.py, retriever.py, vector_index.py):
        corpus ─► heading chunks ─► bge-small embeddings (Qdrant, local) + BM25
               ─► reciprocal rank fusion (k=60) ─► cross-encoder rerank ─► top 5
```

| Component | File(s) | Responsibility |
|---|---|---|
| API | `careflow/api.py` | HTTP interface, input validation, status codes, staff-token check |
| Agent team | `careflow/team.py` | Supervisor, specialists, safety critic, bounded state |
| Workflow (M5) | `careflow/workflow.py` | Intake, triage, eligibility, human approval, escalation, CLI |
| Guardrails | `careflow/guardrails.py` | Input, retrieved-document and output checks |
| LLM client | `careflow/llm_client.py` | The single place that calls the model |
| MCP server / client | `careflow/mcp_server.py`, `careflow/mcp_client.py` | Typed clinic tools and resources over stdio |
| Reliability | `careflow/reliability.py` | Circuit breaker, retries, `ResilientEHR` |
| Retrieval / RAG | `careflow/ingest.py`, `vector_index.py`, `retriever.py`, `rag.py` | Grounded, cited policy answers |
| Memory | `careflow/memory_store.py`, `memory_context.py`, `working_memory.py` | Patient preferences and recent history |
| Tracing | `careflow/tracing.py` | Langfuse spans, local telemetry |
| Escalations | `careflow/escalations.py` | Idempotent escalation queue (one per thread) |

---

## 3. Request lifecycle

1. **API validation.** Pydantic checks the patient reference format, message length (max 2,000 characters) and channel. Bad input gets **422 `invalid_request`**, and the submitted values are not echoed back.
2. **Input guard** (no LLM):
   - A weighted injection score of 2 or more → **blocked** (API returns **422 `blocked_by_guardrail`** with reason and flags).
   - Emergency patterns → the **emergency** message straight away.
   - Clinical-request patterns (SOP-401 categories A–E, including self-declared clinicians) → marked as needing a clinician.
3. **Intake.** The LLM extracts a structured `PatientRequest` (validated, with repair attempts).
4. **Supervisor.** Chooses `insurance_agent` or `referral_agent`. A rule decides first; the LLM is used only when the rule is unsure, and its choice is validated.
5. **Specialist.** Runs a bounded tool loop (max 5 steps) over MCP. Tools are limited to that agent's allowed list, and `patient_ref` is overwritten with the requesting patient, so an agent cannot read another patient's data.
6. **Safety review (critic).** Deterministic checks first: clinical advice patterns, numbers not found in tool results, and dollar figures quoted for a plan that does not exist. Then an LLM verdict, using SOP text read over MCP. If there are problems, the draft goes back for revision (max 2). If it still fails, a safe hand-off reply is used.
7. **Finalize and output guard.** Last check for leaked internal text, other patients' IDs or medical advice, and the reply length cap.
8. **Clinical path.** SOP-401 deflection, then `interrupt()` pauses the graph (API returns **202**). A staff member decides through `POST /approvals/{thread_id}` (approve / reject / edit). Approval writes **exactly one** escalation (`thread_id` is UNIQUE).

The state is checkpointed after every node, so a paused request survives a process restart (`evals/restart_test.py`).

---

## 4. Key design decisions

| Decision | Why |
|---|---|
| Guardrails run **before** the LLM | Blocks injections and emergencies at zero cost, independent of model behaviour |
| Boundaries enforced in **code**, not prompts | A prompt can be ignored; an allowed-tool list and a forced `patient_ref` cannot |
| Each node may write only its own state keys (`OWNED_KEYS`) | Stops one agent from overwriting another's results; the audit log is append-only |
| Hop cap 4, revision cap 2, graph recursion limit 25 | No infinite loops between supervisor, specialists and critic |
| MCP for clinic tools | Typed, discoverable tools with timeouts; the server can be replaced by a real EHR adapter |
| Hybrid retrieval + rerank | Best MRR in the retrieval eval (0.956 vs 0.889 for dense only); exact IDs like procedure codes need BM25 |
| Write tools are never retried | Avoids duplicate bookings or escalations |
| Fallbacks are always **flagged** (`used_fallback`, `EHR_UNAVAILABLE`) | The system never presents a guess as a fact |
| Stale-cache fallback **off** by default | Outdated coverage data could mislead a patient about cost |
| Single LLM layer with pacing | One place for retries, breaker, cost and token tracking, and free-tier limits |

---

## 5. Failure modes and how they are handled

| Failure | Detection | Behaviour | Evidence |
|---|---|---|---|
| Gemini rate limit (429) or timeout | LiteLLM exception | Exponential-backoff retries (up to 7), pacing of 7 s between calls | `tests/test_llm_client.py` |
| Gemini down for a long time | LLM circuit breaker opens | `LLMUnavailable` → safe hand-off message; optional fallback model | `tests/test_llm_resilience.py` |
| Clinic system flaky | Transient error codes | Retry reads only (max 2, backoff 0.5 s) | Fault injection: success 0.727 → 0.967 |
| Clinic system outage | EHR circuit breaker (3 failed calls, 10 s cooldown) | Fast-fail with flagged `EHR_UNAVAILABLE`; fewer calls to the failing system (457 → 278 attempts) | Fault injection: 0 unflagged failures |
| MCP server does not start or hangs | Open timeout 60 s, call timeout 10 s | `MCP_UNAVAILABLE` / `EHR_TIMEOUT` error to the agent, then a safe reply | `tests/test_mcp.py` |
| Process restart while waiting for approval | SQLite checkpointer | Resumes from the saved state; no duplicate escalation | `evals/results/restart_test.json` |
| Same approval sent twice | `snapshot.next` empty / UNIQUE thread | 409 `not_waiting`; still one escalation | `tests/test_api.py` |
| LLM returns invalid JSON at intake | Pydantic validation | Repair attempts, then a safe hand-off | `evals/results/intake_eval.json` |
| Policy answer not supported by sources | Citation and number checks | Repair, then "I don't know" | `evals/results/rag_eval.json` |
| Poisoned policy document | `scan_documents` | The chunk is dropped before reaching the LLM | `tests/test_guardrails.py` |
| Agent loops | Hop and revision caps | Stops with a hand-off reply | `tests/test_team.py` |

---

## 6. Security and guardrail coverage

| Threat | Control | Layer |
|---|---|---|
| Prompt injection in the patient message | Weighted pattern score (block at 2 or more) | Input guard |
| Injection hidden in retrieved documents | Document scan + `untrusted_document` tags in the prompt | RAG |
| Medical advice in a reply | Clinical detection on input, critic patterns, output check | Input, critic, output |
| Reading another patient's data | `patient_ref` forced in code; output check for other patient IDs | Executor, output guard |
| Agent using the wrong tools | Allowed-tool list per agent | Executor |
| Unauthorised approvals | `X-Staff-Token`, compared in constant time (`hmac.compare_digest`); endpoint off when no token is set | API |
| Malformed or oversized requests | Pydantic schema, length limits, 422 without echoing input | API |
| Secrets in the repository | `.env` in `.gitignore`; `.env.example` has placeholders only | Repo |
| PII in traces | Hashed user IDs (salted); no message text in metadata | Tracing |

**Guardrail eval** (29 golden cases, `evals/results/guardrail_eval.json`): accuracy 1.0, refusal recall 1.0, refusal precision 1.0.

**Known gaps**
- Pattern-based input checks can miss heavily reworded or non-English attacks. The LLM critic and output guard are the backup layers, and they also run on an LLM.
- The 29-case golden set is small. It covers each SOP-401 category, but not every phrasing.
- No rate limiting or patient authentication on the API (a real deployment would put it behind the clinic's login).

---

## 7. Evaluation summary

| Area | Result | File |
|---|---|---|
| Unit tests | 258 passed (offline) | `tests/` |
| Intake extraction | 20/20 valid | `intake_eval.json` |
| Tool selection | 10/10 | `tool_selection.json` |
| Retrieval (hybrid + rerank) | MRR 0.956, hit@1 0.933, recall@3 1.0 | `retrieval_eval.json` |
| RAG answers | correct, cited, numbers match, judge-grounded: all 1.0 | `rag_eval.json` |
| Agent team | 14/14 checks | `team_test.json` |
| Fault injection | flaky 0.727 → 0.967; outage 0.73 → 0.793, 0 unflagged failures | `fault_injection.json` |
| Guardrails | accuracy, recall, precision 1.0 | `guardrail_eval.json` |
| End-to-end baseline → improved | 28/29 → **29/29**; answer content 0.938 → 1.0; guardrail, outcome, routing, tools 1.0 in both; judge faithful 1.0 → 0.938 (one judge error on e27, explained in `docs/experiments.md`) | `e2e_baseline.json`, `e2e_improved.json` |
| Cost | team test: 9 LLM calls, 9,967 tokens, about $0.0044 at list price | `kpi_report.json` |

The one baseline failure (e12: a cost quoted for a plan that does not exist) was fixed in the system, not in the golden set. The specialist now notes unknown plans, and the critic rejects dollar figures for plans not in the plans table. Re-running the same 29 cases gave 29/29.

---

## 8. Deployment notes

- **Run:** `python -m careflow.api` (or `uvicorn careflow.api:app --host 127.0.0.1 --port 8000`). Build the policy index once with `python -m careflow.vector_index`.
- **Configuration:** everything comes from environment variables (`careflow/config.py`); secrets live only in `.env`.
- **Concurrency:** one graph run at a time (a lock around the SQLite checkpointer). To scale: Postgres checkpointer, a job queue, several workers.
- **Tracing:** set the Langfuse keys to enable traces; otherwise only local telemetry is written.
- **Production changes needed:** patient authentication, HTTPS, API rate limiting, a real EHR adapter behind the MCP server, a paid LLM tier (the free tier allows about 8 calls per minute with the current pacing), and monitoring alerts on breaker state.

---

## 9. Limitations and future work

- Synthetic data and a mock clinic system.
- English only.
- Free-tier model: slow (median about 27 s per end-to-end case because of pacing) and rate-limited.
- Future work: multilingual guardrails, a larger golden set from real (anonymised) traffic, an LLM-based injection classifier as a second input check, and a staff dashboard for the escalation queue.