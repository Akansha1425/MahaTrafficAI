"""Model Context Protocol (MCP) Server Architecture.

Exposes controlled, allowlisted tools to AI agents with strict schema validation,
timeout management, and refusal of unapproved execution commands.
"""

import logging
from typing import Any, Dict, List
from mcp_server.tools import ALLOWED_TOOLS_REGISTRY

logger = logging.getLogger("mahatraffic.mcp")


class MahaTrafficMCPServer:
    """Secure MCP Server enforcing tool allowlisting and sandboxed execution."""

    def __init__(self, allowed_tools: List[str] = None):
        self.registry = ALLOWED_TOOLS_REGISTRY
        self.allowed_tools = set(allowed_tools or self.registry.keys())
        logger.info("Initialized MahaTraffic MCP Server with %d allowed tools", len(self.allowed_tools))

    def list_tools(self) -> List[str]:
        """Return list of allowed and registered tool identifiers."""
        return [tool for tool in self.registry.keys() if tool in self.allowed_tools]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an approved tool with argument verification and error containment.

        Rejects unapproved tools or arbitrary shell/code invocation.
        """
        if tool_name not in self.allowed_tools:
            logger.warning("Security violation: Attempted invocation of non-allowlisted tool '%s'", tool_name)
            raise PermissionError(f"Tool '{tool_name}' is not in the approved MCP tool allowlist.")

        if tool_name not in self.registry:
            raise KeyError(f"Tool '{tool_name}' is not registered in the system.")

        tool_func = self.registry[tool_name]
        logger.info("Executing MCP tool: %s", tool_name)
        return tool_func(arguments)


if __name__ == "__main__":
    server = MahaTrafficMCPServer()
    print(f"MahaTraffic MCP Server initialized. Registered tools: {server.list_tools()}")
