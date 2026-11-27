"""Client for CareFlow's MCP server.

The rest of CareFlow is synchronous, while the MCP SDK is asynchronous.
CareFlowMCP runs the MCP session on a small background event loop and
offers plain methods: list_tools(), tool_schemas(), call_tool(),
read_resource() and get_prompt().

One server process is started per `with` block and reused for every call
inside it:

    with CareFlowMCP() as ehr:
        ehr.call_tool("check_insurance_eligibility", {"patient_ref": "MHP-P-00000"})
        ehr.read_resource("policy://clinical-escalation")
"""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from careflow import config

logger = logging.getLogger(__name__)

CALL_TIMEOUT_SECONDS = 60


class MCPToolError(Exception):
    """The MCP server could not be reached or rejected the request."""


def server_parameters(extra_env: dict | None = None) -> StdioServerParameters:
    """How to start the CareFlow MCP server as a child process."""
    env = dict(os.environ)
    env.update(extra_env or {})
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "careflow.mcp_server"],
        cwd=str(config.PROJECT_ROOT),
        env=env,
    )


class CareFlowMCP:
    """A synchronous handle on one MCP session."""

    def __init__(self, params: StdioServerParameters | None = None):
        self.params = params or server_parameters()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._session: ClientSession | None = None
        self._ready: asyncio.Future | None = None
        self._stop: asyncio.Event | None = None
        self._owner: asyncio.Future | None = None

    # -- lifecycle --

    def __enter__(self) -> "CareFlowMCP":
        self._thread.start()
        try:
            self._run(self._start_owner())
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    async def _start_owner(self) -> None:
        """Start the task that owns the session and wait until it is ready."""
        self._ready = self._loop.create_future()
        self._stop = asyncio.Event()
        self._owner = asyncio.ensure_future(self._own_session())
        await self._ready

    async def _own_session(self) -> None:
        """Open the session, keep it open until close(), then close it.

        Opening and closing must happen in this same task (an anyio rule),
        so one task owns the whole session lifetime.
        """
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(self.params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._ready.set_result(None)
                await self._stop.wait()
        except Exception as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            else:
                logger.warning("MCP session closed with an error: %s", type(error).__name__)
        finally:
            self._session = None

    def close(self) -> None:
        if self._owner is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(asyncio.shield(self._owner), 10), self._loop
                ).result(timeout=15)
            except Exception:
                logger.warning("MCP session did not close in time")
            self._owner = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def _run(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=CALL_TIMEOUT_SECONDS)

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPToolError("MCP session is not open; use 'with CareFlowMCP() as ehr:'")
        return self._session

    # -- tools --

    def list_tools(self) -> list:
        return self._run(self._require_session().list_tools()).tools

    def tool_schemas(self, allowed: set[str] | None = None) -> list[dict]:
        """MCP tools as LLM tool definitions, optionally only the allowed ones."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self.list_tools()
            if allowed is None or tool.name in allowed
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call one tool. Never raises for tool problems: they come back as errors."""
        try:
            result = self._run(self._require_session().call_tool(name, arguments))
        except Exception as error:
            logger.warning("MCP call %s failed: %s", name, type(error).__name__)
            return {"success": False, "error_code": "MCP_UNAVAILABLE",
                    "message": f"The clinic system could not be reached ({type(error).__name__})."}
        text = "".join(getattr(block, "text", "") for block in result.content)
        if result.isError:
            # e.g. an argument failed the server's validation (bad patient id)
            return {"success": False, "error_code": "INVALID_ARGUMENT", "message": text[:500]}
        if isinstance(result.structuredContent, dict):
            data = result.structuredContent
            # FastMCP wraps non-object results as {"result": ...}
            return data.get("result", data) if set(data) == {"result"} else data
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"success": False, "error_code": "BAD_RESPONSE", "message": text[:500]}

    # -- resources and prompts --

    def read_resource(self, uri: str) -> str:
        result = self._run(self._require_session().read_resource(uri))
        return "".join(getattr(block, "text", "") for block in result.contents)

    def get_prompt(self, name: str, arguments: dict) -> str:
        result = self._run(self._require_session().get_prompt(name, arguments))
        return "\n".join(m.content.text for m in result.messages if hasattr(m.content, "text"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    with CareFlowMCP() as ehr:
        print("Tools:", [tool.name for tool in ehr.list_tools()])
        print("Eligibility:", ehr.call_tool("check_insurance_eligibility",
                                            {"patient_ref": "MHP-P-00000"}))
        print("Bad id:", ehr.call_tool("check_insurance_eligibility", {"patient_ref": "../etc"}))
        print("Referrals:", ehr.call_tool("get_referral_status", {"patient_ref": "MHP-P-00353"}))
        print("Policies:", ehr.read_resource("policy://index"))
        print("SOP starts:", ehr.read_resource("policy://clinical-escalation")[:80], "...")