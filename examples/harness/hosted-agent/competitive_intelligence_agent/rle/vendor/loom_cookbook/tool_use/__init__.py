"""Vendored subset of ``loom_cookbook.tool_use``.

``rl/simulated_tools.py`` is shared verbatim between Loom RL training and this
RLE harness container, so it imports ``tool``, ``ToolResult`` and
``simple_tool_result`` from here. Installing the real cookbook into the image
to get three symbols would pull torch and a CUDA stack into a container whose
only job is to serve four HTTP routes.

``tools.py`` and ``types.py`` are copied byte-for-byte from the cookbook so the
tool schemas and dispatch behave identically; only ``renderers/base.py`` is a
shim, reduced to the three typed dicts those two modules import.
``tests/test_rle_vendor_parity.py`` fails if the copies drift.

``agent_tool_message_env`` is deliberately absent: it is the Loom RL episode
loop, and in a Harness RLE the agent owns its own loop.
"""

from loom_cookbook.tool_use.tools import (
    FunctionTool,
    error_tool_result,
    handle_tool_call,
    simple_tool_result,
    tool,
)
from loom_cookbook.tool_use.types import (
    Tool,
    ToolInput,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "FunctionTool",
    "Tool",
    "ToolInput",
    "ToolResult",
    "ToolSpec",
    "error_tool_result",
    "handle_tool_call",
    "simple_tool_result",
    "tool",
]
