"""Builds MCP-callable wrappers around the surface's simulated tools.

The tools themselves are ``loom_cookbook.tool_use.FunctionTool`` templates,
shared unmodified with Loom RL training and the legacy harness. They have no
native Python call signature -- they are invoked through ``tool.run(ToolInput(
...))`` -- so each one needs a permissive wrapper built here before FastMCP can
register it, and a real schema published afterward so the agent does not read
a uselessly vague one.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable

from fastmcp.tools.function_tool import FunctionTool
from loom_cookbook.tool_use import ToolInput

from rle.rl.simulated_tools import ToolSession, session_tools

#: The surface both halves of a rollout must agree on. The harness container
#: and the agent package read the *same* variable name with the *same*
#: default, which is the only reason they cannot silently disagree -- they are
#: deployed separately and nothing plumbs the value from one to the other.
#:
#: Default `production`: `routine` serves four invented semantic tools that the
#: deployed agent does not have. A policy trained on it scored 0.97 tool
#: discipline in the simulator and 0.43 on the real benchmark, because only
#: `web_search` was common to both. `routine` is kept to reproduce those runs.
TOOL_SURFACE = os.environ.get("COMPETITIVE_INTEL_TOOL_SURFACE", "production")


def _tool_templates() -> list[Any]:
    """The surface's tools, read off the class instead of an instance.

    MCP tools register once, when the environment is constructed; the tools
    themselves bind to a ``ToolSession`` that does not exist until ``reset``.
    ``session_tools`` only reads attributes, and ``FunctionTool`` is a
    descriptor that returns itself when read from the class, so reading the
    surface from ``ToolSession`` yields unbound tools -- names, descriptions and
    argument models, with no session attached. That is exactly what registration
    needs; the session is looked up per call, in the wrapper below.
    """
    return session_tools(ToolSession, TOOL_SURFACE)


def _make_tool_wrapper(template: Any, resolve: Callable[[str], Any]) -> Callable[..., Any]:
    """An MCP-callable wrapper around one simulated tool.

    Dispatch goes through ``FunctionTool.run``, the entry point Loom's episode
    loop uses, so a call recorded here is the call the rubric scores. ``run``
    never raises: invalid arguments and tool faults both come back as an error
    result. That is deliberate -- a malformed call is the agent's mistake, and
    the rubric needs to see the wasted call rather than have the attempt fault.

    That contract only holds if the call reaches ``run``, so the signature
    build here is deliberately permissive: every argument optional and
    untyped. FastMCP validates a call against this signature
    *before* the body runs and turns a rejection into a JSON-RPC error, which
    would fault the agent's attempt instead of handing it a tool error it
    could read and correct. Keeping the signature open leaves Loom's own
    validation the only one, which is what the legacy HTTP route did.

    The permissive signature would otherwise publish a uselessly vague schema,
    so ``register_tools`` replaces it with one generated from the tool's own
    pydantic argument model. The schema an agent reads is therefore still
    generated from the same model ``run`` validates against, and cannot drift
    from it.

    Arguments the caller omitted arrive as ``None`` and are dropped, so Loom
    sees a genuinely absent argument and applies its own default or its own
    "required" error, rather than a ``None`` the tool never offered.

    One divergence from the legacy route survives: an argument name no tool
    declares is still rejected by FastMCP, which cannot wrap a function that
    takes ``**kwargs``. Loom ignored those. A missing or mistyped argument,
    the far commoner mistake, now behaves as it did.
    """
    parameters = [
        inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Any
        )
        for name in template._params_model.model_fields
    ]
    annotations: dict[str, Any] = {
        name: Any for name in template._params_model.model_fields
    }
    annotations["return"] = str

    async def call(**arguments: Any) -> str:
        tool = resolve(template.name)
        supplied = {name: value for name, value in arguments.items() if value is not None}
        result = await tool.run(ToolInput(arguments=supplied, call_id=""))
        messages = result.messages or []
        # The tool's single message carries a JSON string, and that string is
        # what the model saw as the tool's output during training.
        return messages[0].get("content", "") if messages else ""

    call.__name__ = template.name
    call.__doc__ = template.description
    call.__signature__ = inspect.Signature(parameters, return_annotation=str)
    call.__annotations__ = annotations
    return call


def register_tools(environment: Any, resolve: Callable[[str], Any]) -> list[str]:
    """Registers the surface's tools and publishes their real schemas.

    Registration goes through the inherited ``tool`` decorator so the base
    class keeps its own bookkeeping, then each tool is re-registered with the
    schema generated from its pydantic argument model, replacing the vague one
    FastMCP derives from the permissive signature above. ``remove_tool`` first,
    because adding over a live name is a duplicate registration.
    """
    names = []
    for template in _tool_templates():
        wrapper = _make_tool_wrapper(template, resolve)
        environment.tool()(wrapper)
        described = FunctionTool.from_function(
            wrapper, name=template.name, description=template.description
        )
        described.parameters = template._params_model.model_json_schema()
        environment.mcp_server.local_provider.remove_tool(template.name)
        environment.mcp_server.local_provider.add_tool(described)
        names.append(template.name)
    return names
