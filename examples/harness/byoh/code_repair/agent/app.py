"""Example BYOH agent harness for the code-repair RLE.

Deploy this anywhere reachable over HTTPS and register its URL with
``azd ai rle init --type Harness --subtype BYOH --base-url <this-url>/invoke``.
RLE then POSTs directly to that URL for every rollout with the body shape
below and expects ``{"output_text": "..."}`` back:

    POST <base-url>/invoke
    {
      "rollout_id": "...",
      "agent_input": {"...": "agent-specific input"},
      "rollout_context": {
        "capture_proxy_endpoint": "https://.../v1",
        "capture_proxy_session_key": "...",
        "sandbox_tools_endpoint": "https://.../tools",
        "sandbox_tools_bearer_token": "..."
      }
    }

RLE never attaches caller, workspace, or identity headers to this request, so
protect the endpoint yourself (for example, require a shared secret header
and validate it before dispatching to the agent loop).

This file and ``../../../hosted-agent/code_repair/agent/main.py`` share the same
``run_agent_loop``: the harness's native agent loop is unchanged between the
two RLE subtypes. Only how ``model`` and ``call_tool`` are constructed
differs -- see ``create_model_client``/``call_tool`` here versus the
header-driven equivalents there.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI(title="byoh-code-repair-agent")


class RolloutContext(BaseModel):
    capture_proxy_endpoint: str
    capture_proxy_session_key: str
    sandbox_tools_endpoint: str
    sandbox_tools_bearer_token: str


class InvocationRequest(BaseModel):
    rollout_id: str
    agent_input: dict[str, Any]
    rollout_context: RolloutContext


def create_model_client(rollout_context: RolloutContext) -> AsyncOpenAI:
    """Points the harness's normal model client at RLE's capture proxy.

    The capture proxy speaks the same model dialect as the harness's real
    model endpoint and returns the ordinary response the harness expects,
    while recording a detailed trajectory for the rollout graph.
    """
    return AsyncOpenAI(
        base_url=rollout_context.capture_proxy_endpoint,
        api_key=rollout_context.capture_proxy_session_key,
    )


async def call_tool(rollout_context: RolloutContext, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Routes a tool call to RLE's sandbox instead of the real production tool."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{rollout_context.sandbox_tools_endpoint}/tools/{tool_name}",
            json=arguments,
            headers={"Authorization": f"Bearer {rollout_context.sandbox_tools_bearer_token}"},
        )
        response.raise_for_status()
        return response.json()


# A harness's output contract belongs in its own system prompt. Unlike the
# Gym/OpenEnv subtype -- where RLE drives the model and the environment's
# observation carries the instruction -- here the harness owns the model call
# and the environment never sees the prompt. `/reset` hands over the raw
# GitHub issue, and a real issue does not tell you to answer in unified-diff
# form. Without this, a model answers the way a person would -- prose, or a
# fenced code block -- and the `workspace.apply_patch` mock's real `git apply`
# rejects it with "No valid patches in input", so the loop below returns early
# and the rollout scores 0 for a formatting reason rather than for a wrong fix.
PATCH_FORMAT_INSTRUCTION = """\
Reply with a unified diff and nothing else.

- Output only the diff: no explanation, no commentary, no Markdown code fences.
- Use the form `git diff` produces, with `a/` and `b/` path prefixes and `@@` hunk headers.
- Paths must be relative to the repository root.
- The diff must apply cleanly with `git apply`.

Expected shape:

--- a/pkg/module.py
+++ b/pkg/module.py
@@ -10,7 +10,7 @@ def example(value):
-    return value * 2
+    return value * 3
"""


async def run_agent_loop(model: AsyncOpenAI, rollout_context: RolloutContext, agent_input: dict[str, Any]) -> str:
    """The harness's native agent loop: read the issue, propose a patch, open a PR.

    Unchanged between RLE subtypes apart from how `model` and tool calls are
    routed -- see this module's ``create_model_client``/``call_tool`` versus
    ``../../../hosted-agent/code_repair/agent/main.py``'s header-driven equivalents.
    """
    issue = agent_input["issue"]

    completion = await model.chat.completions.create(
        model="code-repair-agent",
        messages=[
            {
                "role": "system",
                "content": f"You fix reported issues in this repository.\n\n{PATCH_FORMAT_INSTRUCTION}",
            },
            {"role": "user", "content": issue},
        ],
    )
    patch = completion.choices[0].message.content or ""

    apply_result = await call_tool(rollout_context, "workspace.apply_patch", {"patch": patch})
    if not apply_result.get("applied"):
        return f"Failed to apply patch: {apply_result.get('error')}"

    await call_tool(
        rollout_context,
        "github.create_pull_request",
        {
            "branch": "fix/reported-issue",
            "title": "Fix reported issue",
            "body": f"Applies the following patch in response to the report:\n\n{patch}",
        },
    )
    return "Opened a pull request with the fix."


@app.post("/invoke")
async def invoke(request: InvocationRequest) -> dict[str, str]:
    model = create_model_client(request.rollout_context)
    output_text = await run_agent_loop(model, request.rollout_context, request.agent_input)
    return {"output_text": output_text}


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
