"""The single provider-agnostic LLM layer for CareFlow.

All LLM calls in the project go through call_llm(). To switch
provider, change CAREFLOW_LLM_MODEL in .env (no code edit needed).

Temporary provider errors (rate limit, overload, timeout) are retried
with exponential backoff: 2s, 4s, 8s, ... Permanent errors (wrong model
name, bad API key) are NOT retried and are raised immediately.
"""

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


def call_llm(messages: list[dict], json_mode: bool = False) -> str:
    """Send chat messages to the configured model and return the text reply.

    Args:
        messages: Chat messages, e.g. [{"role": "user", "content": "Hi"}].
        json_mode: If True, ask the model to answer with a JSON object.

    Raises:
        Exception: a permanent error immediately, or a temporary error
        once all retries are used up.
    """
    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "timeout": config.LLM_TIMEOUT_SECONDS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            response = completion(**kwargs)
            return response.choices[0].message.content
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


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}])
    print(reply)