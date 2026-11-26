"""The single provider-agnostic LLM layer for CareFlow.

All LLM calls in the project go through call_llm(). To switch
provider, change CAREFLOW_LLM_MODEL in .env (no code edit needed).
"""

import logging

from litellm import completion

from careflow import config

logger = logging.getLogger(__name__)


def call_llm(messages: list[dict], json_mode: bool = False) -> str:
    """Send chat messages to the configured model and return the text reply.

    Args:
        messages: Chat messages, e.g. [{"role": "user", "content": "Hi"}].
        json_mode: If True, ask the model to answer with a JSON object.

    Raises:
        Exception: any provider error is logged and re-raised so the
        caller can decide how to handle it.
    """
    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "timeout": config.LLM_TIMEOUT_SECONDS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = completion(**kwargs)
    except Exception:
        logger.exception("LLM call failed (model=%s)", config.LLM_MODEL)
        raise

    return response.choices[0].message.content


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}])
    print(reply)