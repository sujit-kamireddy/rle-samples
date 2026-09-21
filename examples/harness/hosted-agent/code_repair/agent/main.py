"""Example Foundry Hosted Agent for the code-repair RLE.

Register this agent's name and version with
``azd ai rle init --type Harness --subtype HostedAgent --agent-name ... --agent-version ...``.
For each rollout, RLE creates a session for this Hosted Agent version and
calls its Responses API with four extra request headers that carry the
rollout-scoped runtime context:

  x-client-rle-rollout-id
  x-client-rle-model-endpoint          # capture proxy base URL
  x-client-rle-model-api-key           # capture proxy session key
  x-client-rle-sandbox-tools-endpoint  # sandbox tool routes for this rollout

This file and ``../../../byoh/code_repair/agent/app.py`` share the same ``run_agent_loop``:
the harness's native agent loop is unchanged between the two RLE subtypes.
Only how ``model`` and tool calls are constructed differs -- BYOH reads them
from its invocation request body, this reads them from headers, falling back
to the agent's normal production model endpoint and tools when absent
(ordinary, non-rollout traffic).
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
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{sandbox_tools_endpoint}/tools/{tool_name}", json=arguments)
        response.raise_for_status()
        return response.json()


async def call_production_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(f"Wire {tool_name} to your production tool implementation.")


async def run_agent_loop(headers: dict[str, str], agent_input: dict[str, Any]) -> str:
    """The harness's native agent loop: read the issue, propose a patch, open a PR.

    Unchanged between RLE subtypes apart from how ``model`` and tool calls
    are routed -- see this module's header-driven ``create_model_client``/
    ``call_tool`` versus ``../../../byoh/code_repair/agent/app.py``'s equivalents.
    """
    model = create_model_client(headers)
    issue = agent_input["issue"]

    completion = await model.chat.completions.create(
        model="code-repair-agent",
        messages=[
            {
                "role": "system",
                "content": "You fix reported issues in this repository. Reply with only a unified diff patch.",
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
