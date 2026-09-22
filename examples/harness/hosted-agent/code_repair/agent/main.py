"""Example Foundry Hosted Agent for the code-repair RLE.

Register this agent's name and version with
``azd ai rle init --type Harness --subtype HostedAgent --agent-name ... --agent-version ...``.
For each rollout, RLE creates a session for this Hosted Agent version and
calls its Responses API with five extra request headers that carry the
rollout-scoped runtime context:

  x-client-rle-rollout-id
  x-client-rle-model-endpoint          # capture proxy base URL
  x-client-rle-model-api-key           # capture proxy session key
  x-client-rle-sandbox-tools-endpoint  # sandbox tool routes for this rollout
  x-client-rle-sandbox-tools-token     # bearer token authorizing those routes

This file and the ``BYOH`` code-repair sample share the same ``run_agent_loop``:
the harness's native agent loop is unchanged between the two RLE subtypes.
Only how ``model`` and tool calls are constructed differs -- ``BYOH`` reads them
from its invocation request body, this reads them from headers, falling back
to the agent's normal production model endpoint and tools when absent
(ordinary, non-rollout traffic). Scaffold that sample with
``azd ai rle init --type Harness --subtype BYOH``.
"""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

# Adapt to your actual hosted-agent hosting framework: any framework that
# exposes the incoming request's headers to the handler works the same way.
# See azure-ai-agentserver's ResponseContext for one such integration point.
ROLLOUT_ID_HEADER = "x-client-rle-rollout-id"
MODEL_ENDPOINT_HEADER = "x-client-rle-model-endpoint"
MODEL_API_KEY_HEADER = "x-client-rle-model-api-key"
SANDBOX_TOOLS_ENDPOINT_HEADER = "x-client-rle-sandbox-tools-endpoint"
SANDBOX_TOOLS_TOKEN_HEADER = "x-client-rle-sandbox-tools-token"

PRODUCTION_MODEL_ENDPOINT = "https://api.contoso-models.example.com/v1"


def create_model_client(headers: dict[str, str]) -> AsyncOpenAI:
    """Points the harness's normal model client at RLE's capture proxy when present."""
    model_endpoint = headers.get(MODEL_ENDPOINT_HEADER, PRODUCTION_MODEL_ENDPOINT)
    model_api_key = headers.get(MODEL_API_KEY_HEADER, "production-key-from-secret-store")
    return AsyncOpenAI(base_url=model_endpoint, api_key=model_api_key)


async def call_tool(headers: dict[str, str], tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Routes a tool call to RLE's sandbox mocks during a rollout, production tools otherwise."""
    sandbox_tools_endpoint = headers.get(SANDBOX_TOOLS_ENDPOINT_HEADER)
    if sandbox_tools_endpoint is None:
        return await call_production_tool(tool_name, arguments)
    # The tool routes are rollout-scoped: this token authorizes calls for this
    # rollout only, and only under /tools. It is not a workspace credential.
    tools_token = headers.get(SANDBOX_TOOLS_TOKEN_HEADER)
    request_headers = {"Authorization": f"Bearer {tools_token}"} if tools_token else {}
    async with httpx.AsyncClient() as client:
        # The header's value already ends in /tools, so the tool name is all
        # that is appended. Adding another /tools yielded /tools/tools/<name>.
        response = await client.post(
            f"{sandbox_tools_endpoint}/{tool_name}",
            json=arguments,
            headers=request_headers,
        )
        response.raise_for_status()
        return response.json()


async def call_production_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(f"Wire {tool_name} to your production tool implementation.")


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


async def run_agent_loop(headers: dict[str, str], agent_input: dict[str, Any]) -> str:
    """The harness's native agent loop: read the issue, propose a patch, open a PR.

    Unchanged between RLE subtypes apart from how ``model`` and tool calls
    are routed -- this module builds them with header-driven
    ``create_model_client``/``call_tool``, while the ``BYOH`` sample builds
    them from its invocation request body.
    """
    model = create_model_client(headers)
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

    apply_result = await call_tool(headers, "workspace.apply_patch", {"patch": patch})
    if not apply_result.get("applied"):
        return f"Failed to apply patch: {apply_result.get('error')}"

    await call_tool(
        headers,
        "github.create_pull_request",
        {
            "branch": "fix/reported-issue",
            "title": "Fix reported issue",
            "body": f"Applies the following patch in response to the report:\n\n{patch}",
        },
    )
    return "Opened a pull request with the fix."


async def handle_responses_request(headers: dict[str, str], agent_input: dict[str, Any]) -> dict[str, str]:
    """Entry point your Hosted Agent's Responses API handler should call."""
    rollout_id = headers.get(ROLLOUT_ID_HEADER)
    output_text = await run_agent_loop(headers, agent_input)
    return {"output_text": output_text, "rollout_id": rollout_id or ""}
