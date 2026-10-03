"""The three typed dicts ``tool_use`` imports from the cookbook's renderers.

The real module is the rendering layer: it turns messages and tool specs into
model-specific token streams. None of that runs in this container, which never
renders, decodes or tokenizes anything -- the agent owns its own model calls
and this image only serves ``/reset``, ``/tools/*`` and ``/grade``. What
``tool_use`` actually needs from it are these three structural types, so they
are all that is reproduced here.

``ToolCall`` is a pydantic model upstream because renderers parse it out of
model output. Here it is only ever constructed by a caller and read back, so
the same three fields as a typed dict carry the same information without the
dependency.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ToolCallFunctionBody(TypedDict):
    """The function a tool call names, and its JSON-encoded arguments."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """A request to invoke a tool, in OpenAI function-calling shape."""

    function: ToolCallFunctionBody
    id: NotRequired[str]


class Message(TypedDict):
    """One turn in a conversation."""

    role: str
    content: NotRequired[Any]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    trainable: NotRequired[bool]


class ToolSpec(TypedDict):
    """A tool offered to the model, in OpenAI function-calling shape."""

    name: str
    description: str
    parameters: dict[str, Any]
