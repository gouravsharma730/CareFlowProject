"""A bounded tool-calling loop shared by the team's specialist agents.

The caller decides which tools the agent may see (schemas) and how a tool
call is carried out (execute). The loop never runs more than max_steps
model calls and always ends with an answer or an explicit stop reason.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from careflow.llm_client import chat_with_tools

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    answer: str
    stop_reason: str            # "final_answer" or "max_steps"
    steps: int
    tool_calls: list[dict] = field(default_factory=list)


def run_tool_loop(
    system: str,
    user: str,
    schemas: list[dict],
    execute: Callable[[str, dict], dict],
    max_steps: int,
) -> LoopResult:
    """Let the model call tools (through execute) until it answers."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls: list[dict] = []
    for step in range(1, max_steps + 1):
        message = chat_with_tools(messages, schemas)
        if not message.tool_calls:
            return LoopResult(message.content or "", "final_answer", step, calls)
        messages.append({"role": "assistant", "content": message.content,
                         "tool_calls": message.tool_calls})
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                result = execute(call.function.name, arguments)
            except json.JSONDecodeError:
                arguments, result = {}, {"success": False, "error_code": "INVALID_ARGUMENT",
                                         "message": "Tool arguments were not valid JSON."}
            calls.append({"tool": call.function.name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": call.function.name, "content": json.dumps(result)})
    logger.warning("Tool loop stopped after %d steps", max_steps)
    return LoopResult("", "max_steps", max_steps, calls)