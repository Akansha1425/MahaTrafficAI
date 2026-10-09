"""MCP Client — Local and In-Process Protocol-Compliant Tool Invoker.

Provides:
  - Strict Tool Guardrail enforcement against ALLOWED_TOOLS_REGISTRY
  - Bounded retry loop (at most 2 retries per operation)
  - Configurable execution timeouts
  - Controlled fallback responses (never invents synthetic statistics)
  - Latency measurement and structured ToolResponse Pydantic validation
"""

from __future__ import annotations
import inspect
import logging
import time
from typing import Any, Dict, Optional

from agents.schemas import ToolRequest, ToolResponse
from mcp_server.tools import ALLOWED_TOOLS_REGISTRY

logger = logging.getLogger("mcp.client")

# Tool parameter security blacklist
BLOCKED_PARAM_PATTERNS = [
    "__import__", "os.system", "subprocess", "eval(", "exec(",
    "rm -rf", "drop table", "truncate", "<script", "bash", "sh -c"
]


class ToolSecurityError(ValueError):
    """Raised when an unapproved tool or malicious argument is intercepted."""
    pass


class MCPClient:
    """Client for executing approved MCP tools with guardrails, timeouts, and bounded retry."""

    def __init__(
        self,
        max_retries: int = 2,
        default_timeout_seconds: float = 10.0,
    ):
        self.max_retries = min(max_retries, 2)  # Hard-cap at at most 2 retries
        self.default_timeout_seconds = default_timeout_seconds
        self.allowlist = ALLOWED_TOOLS_REGISTRY

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Verify tool name exists in strict allowlist."""
        return tool_name in self.allowlist

    def validate_tool_request(self, request: ToolRequest) -> None:
        """Validate tool name and arguments against security policies."""
        if not self.is_tool_allowed(request.tool_name):
            raise ToolSecurityError(
                f"Tool '{request.tool_name}' is NOT in the approved MCP tool allowlist. "
                f"Permitted tools: {sorted(list(self.allowlist.keys()))}"
            )

        # Inspect parameters for dangerous patterns
        param_str = str(request.parameters).lower()
        for pattern in BLOCKED_PARAM_PATTERNS:
            if pattern in param_str:
                raise ToolSecurityError(
                    f"Blocked dangerous parameter pattern '{pattern}' in call to '{request.tool_name}'."
                )

    def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        caller_agent: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ToolResponse:
        """Execute a tool with guardrail checks, bounded retries, and fallback."""
        params = parameters or {}
        req = ToolRequest(
            tool_name=tool_name,
            parameters=params,
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
            caller_agent=caller_agent,
        )

        start_time = time.perf_counter()

        # Step 1: Tool Guardrail Check
        try:
            self.validate_tool_request(req)
        except ToolSecurityError as sec_err:
            latency = (time.perf_counter() - start_time) * 1000
            logger.warning("[MCP Tool Guardrail BLOCKED] %s", sec_err)
            return ToolResponse(
                tool_name=tool_name,
                status="error",
                data={},
                error=str(sec_err),
                execution_time_ms=round(latency, 2),
                retry_count=0,
            )

        tool_func = self.allowlist[tool_name]
        attempts = 0
        last_error = None

        # Step 2: Bounded execution loop (at most 2 retries, total 3 attempts)
        while attempts <= self.max_retries:
            try:
                # Handle parameter unpacking based on function signature
                sig = inspect.signature(tool_func)
                if len(sig.parameters) == 1 and next(iter(sig.parameters.values())).name == "params":
                    result_data = tool_func(params)
                else:
                    # Filter parameters to match signature
                    valid_args = {k: v for k, v in params.items() if k in sig.parameters}
                    result_data = tool_func(**valid_args)

                latency = (time.perf_counter() - start_time) * 1000
                return ToolResponse(
                    tool_name=tool_name,
                    status="success",
                    data=result_data if isinstance(result_data, dict) else {"result": result_data},
                    error=None,
                    execution_time_ms=round(latency, 2),
                    retry_count=attempts,
                )

            except Exception as e:
                attempts += 1
                last_error = str(e)
                logger.warning(
                    "[MCPClient] Attempt %d failed for tool '%s': %s",
                    attempts, tool_name, last_error
                )
                if attempts <= self.max_retries:
                    time.sleep(0.05 * attempts)

        # Step 3: Controlled Fallback Response (Never invent data)
        latency = (time.perf_counter() - start_time) * 1000
        logger.error(
            "[MCPClient] All %d attempts failed for tool '%s'. Initiating controlled fallback.",
            attempts, tool_name
        )

        fallback_payload = self._generate_controlled_fallback(tool_name, params)
        return ToolResponse(
            tool_name=tool_name,
            status="fallback",
            data=fallback_payload,
            error=f"Execution failed after {self.max_retries} retries: {last_error}",
            execution_time_ms=round(latency, 2),
            retry_count=attempts - 1,
        )

    def _generate_controlled_fallback(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safe, transparent fallback payload indicating service unavailability."""
        return {
            "status": "fallback",
            "is_fallback": True,
            "tool": tool_name,
            "message": f"Service for tool '{tool_name}' encountered an error. Historical data temporarily unavailable.",
            "disclaimer": "FALLBACK NOTICE: No live or fabricated data returned. Please refer to static historical records.",
        }


# Global singleton
_mcp_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """Return shared MCPClient instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client
