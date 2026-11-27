"""The single provider-agnostic LLM layer for CareFlow.

All LLM calls in the project go through this file. To switch provider,
change CAREFLOW_LLM_MODEL in .env (no code edit needed).

Rate limits (free tiers) are handled in two ways:
  1. Pacing: if CAREFLOW_LLM_MIN_SECONDS_BETWEEN_CALLS is set, calls in
     this process are spaced at least that far apart, so we stay under
     the provider's requests-per-minute limit instead of hitting it.
  2. Retries: temporary provider errors (rate limit, overload, timeout)
     are retried with exponential backoff: 2s, 4s, 8s, ... Permanent
     errors (wrong model name, bad API key) are NOT retried.

Every request sent is counted in CALL_STATS, so a script can report how
many calls it used (useful for daily quotas).
"""

import logging
import threading
import time

import litellm
from litellm import completion

from careflow import config, tracing

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

# Requests sent to the provider by this process (retries included).
CALL_STATS = {"requests": 0, "retries": 0}

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


def _complete_with_retry(**kwargs):
    """Call the provider, pacing requests and retrying temporary errors."""
    kwargs.update(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        _wait_for_turn()
        try:
            with tracing.span("llm.completion", as_type="generation", model=config.LLM_MODEL,
                              attempt=attempt + 1, tools=len(kwargs.get("tools") or [])) as gen:
                response = completion(**kwargs)
                gen.record_llm_response(response, config.LLM_MODEL)
            return response
        except TRANSIENT_ERRORS as error:
            if attempt == config.LLM_MAX_RETRIES:
                logger.error(
                    "LLM call failed after %d retries (model=%s): %s",
                    config.LLM_MAX_RETRIES,
                    config.LLM_MODEL,
                    type(error).__name__,
                )
                raise
            CALL_STATS["retries"] += 1
            wait = min(
                config.LLM_RETRY_BASE_SECONDS * (2**attempt),
                config.LLM_RETRY_MAX_SECONDS,
            )
            logger.warning(
                "Temporary LLM error (%s). Retry %d of %d in %.0fs.",
                type(error).__name__,
                attempt + 1,
                config.LLM_MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
        except Exception:
            logger.exception("LLM call failed (model=%s)", config.LLM_MODEL)
            raise


def call_llm(messages: list[dict], json_mode: bool = False) -> str:
    """Send chat messages to the configured model and return the text reply.

    Args:
        messages: Chat messages, e.g. [{"role": "user", "content": "Hi"}].
        json_mode: If True, ask the model to answer with a JSON object.
    """
    kwargs = {"messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _complete_with_retry(**kwargs)
    return response.choices[0].message.content


def chat_with_tools(messages: list[dict], tools: list[dict]):
    """Send messages plus tool definitions; return the model's message.

    The returned message has .content and, when the model wants to use
    tools, .tool_calls.
    """
    response = _complete_with_retry(
        messages=messages, tools=tools, tool_choice="auto"
    )
    return response.choices[0].message


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}])
    print(reply)