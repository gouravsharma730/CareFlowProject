"""The single provider-agnostic LLM layer for CareFlow.

All LLM calls in the project go through this file. To switch provider,
change CAREFLOW_LLM_MODEL in .env (no code edit needed).

Protection for every call:
  1. Pacing: CAREFLOW_LLM_MIN_SECONDS_BETWEEN_CALLS spaces requests out,
     so a free tier's requests-per-minute limit is not hit.
  2. Timeout: CAREFLOW_LLM_TIMEOUT_SECONDS per request.
  3. Retries: temporary provider errors (rate limit, overload, timeout,
     connection) are retried with exponential backoff (2s, 4s, 8s, ...).
     Permanent errors (wrong model name, bad API key) are NOT retried.
  4. Circuit breaker: after BREAKER_FAILURE_THRESHOLD calls in a row fail
     (each after its retries), calls fail fast for BREAKER_RESET_SECONDS
     instead of waiting on a provider that is down.
  5. Fallback: if CAREFLOW_LLM_FALLBACK_MODEL is set, a call the main model
     cannot serve is sent to that model and flagged used_fallback. It is
     off by default. Without it, LLMUnavailable is raised and the caller
     hands the request to a human.

Every request is counted in CALL_STATS (useful for daily quotas).
"""

import logging
import threading
import time

import litellm
from litellm import completion

from careflow import config, tracing
from careflow.reliability import CircuitBreaker

# Stop LiteLLM printing its "Provider List / Give Feedback" help text on
# every error. Our own logger still records each error and retry.
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# Only these errors are worth retrying: they usually go away on their own.
TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.APIConnectionError,
)

# Requests sent by this process (retries included) and fallbacks used.
CALL_STATS = {"requests": 0, "retries": 0, "fallbacks": 0}

# One breaker for the main model, shared by every caller in this process.
LLM_BREAKER = CircuitBreaker("llm")


class LLMUnavailable(Exception):
    """The model could not answer (provider down or breaker open, no fallback)."""


_pace_lock = threading.Lock()
_last_request_at = 0.0


def _wait_for_turn() -> None:
    """Keep at least LLM_MIN_SECONDS_BETWEEN_CALLS between two requests."""
    global _last_request_at
    gap = config.LLM_MIN_SECONDS_BETWEEN_CALLS
    with _pace_lock:
        if gap > 0:
            wait = _last_request_at + gap - time.monotonic()
            if wait > 0:
                logger.debug("Pacing LLM calls: waiting %.1fs", wait)
                time.sleep(wait)
        _last_request_at = time.monotonic()
        CALL_STATS["requests"] += 1


def _call_model(model: str, used_fallback: bool, **kwargs):
    """One model, with retries on temporary errors. Raises the last error."""
    kwargs.update(model=model, temperature=config.LLM_TEMPERATURE,
                  timeout=config.LLM_TIMEOUT_SECONDS)
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        _wait_for_turn()
        try:
            with tracing.span("llm.completion", as_type="generation", model=model,
                              attempt=attempt + 1, tools=len(kwargs.get("tools") or []),
                              used_fallback=used_fallback) as gen:
                response = completion(**kwargs)
                gen.record_llm_response(response, model)
                if used_fallback:
                    gen.warn("answered by the fallback model")
            return response
        except TRANSIENT_ERRORS as error:
            if attempt == config.LLM_MAX_RETRIES:
                logger.error("LLM call failed after %d retries (model=%s): %s",
                             config.LLM_MAX_RETRIES, model, type(error).__name__)
                raise
            CALL_STATS["retries"] += 1
            wait = min(config.LLM_RETRY_BASE_SECONDS * (2 ** attempt),
                       config.LLM_RETRY_MAX_SECONDS)
            logger.warning("Temporary LLM error (%s). Retry %d of %d in %.0fs.",
                           type(error).__name__, attempt + 1, config.LLM_MAX_RETRIES, wait)
            time.sleep(wait)
        except Exception:
            logger.exception("LLM call failed (model=%s)", model)
            raise


def _complete(**kwargs):
    """Main model behind the circuit breaker, then the optional fallback model."""
    problem = None
    if LLM_BREAKER.allow():
        try:
            response = _call_model(config.LLM_MODEL, False, **kwargs)
            LLM_BREAKER.record_success()
            return response
        except TRANSIENT_ERRORS as error:
            LLM_BREAKER.record_failure()  # counted once per failed call
            problem = f"{type(error).__name__} after retries"
    else:
        problem = "circuit open"
        logger.warning("LLM circuit open: not calling %s", config.LLM_MODEL)

    if config.LLM_FALLBACK_MODEL:
        logger.warning("Main model unavailable (%s); using fallback %s",
                       problem, config.LLM_FALLBACK_MODEL)
        CALL_STATS["fallbacks"] += 1
        try:
            return _call_model(config.LLM_FALLBACK_MODEL, True, **kwargs)
        except TRANSIENT_ERRORS as error:
            problem = f"{problem}; fallback {type(error).__name__}"
    raise LLMUnavailable(f"The language model is unavailable ({problem}).")


def call_llm(messages: list[dict], json_mode: bool = False) -> str:
    """Send chat messages to the configured model and return the text reply.

    Raises:
        LLMUnavailable: the model (and any fallback) could not answer.
    """
    kwargs = {"messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return _complete(**kwargs).choices[0].message.content


def chat_with_tools(messages: list[dict], tools: list[dict]):
    """Send messages plus tool definitions; return the model's message.

    The returned message has .content and, when the model wants to use
    tools, .tool_calls.

    Raises:
        LLMUnavailable: the model (and any fallback) could not answer.
    """
    return _complete(messages=messages, tools=tools, tool_choice="auto").choices[0].message


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}])
    print(reply)