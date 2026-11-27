"""The single provider-agnostic LLM layer for CareFlow.

All LLM calls in the project go through this file. To switch provider,
change CAREFLOW_LLM_MODEL in .env (no code edit needed).

Temporary provider errors (rate limit, overload, timeout) are retried
with exponential backoff: 2s, 4s, 8s, ... Permanent errors (wrong model
name, bad API key) are NOT retried and are raised immediately.
"""

import litellm
from litellm import completion

# Stop LiteLLM printing its "Provider List / Give Feedback" help text on
# every error. Our own logger still records each error and retry.
litellm.suppress_debug_info = True

import logging
import time

import litellm
from litellm import completion

from careflow import config

logger = logging.getLogger(__name__)

# Only these errors are worth retrying: they usually go away on their own.
TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.APIConnectionError,
)


def _complete_with_retry(**kwargs):
    """Call the provider, retrying temporary errors with backoff."""
    kwargs.update(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            return completion(**kwargs)
        except TRANSIENT_ERRORS as error:
            if attempt == config.LLM_MAX_RETRIES:
                logger.error(
                    "LLM call failed after %d retries (model=%s): %s",
                    config.LLM_MAX_RETRIES,
                    config.LLM_MODEL,
                    type(error).__name__,
                )
                raise
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