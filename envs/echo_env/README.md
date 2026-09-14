# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `echo_env` OpenEnv MCP environment

Self-hosted, OpenEnv-compatible MCP environment that demonstrates an agent
discovering and calling tools. It has no task dataset: each tool call operates
only on the message supplied in that action. Use it to iterate locally with
your own OpenEnv-compatible agent before publishing an environment version.
It is a RLE connectivity and protocol sanity test, not a training or
evaluation environment. See [`code_rl`](../code_rl/README.md) or
[`math_rl`](../math_rl/README.md) for tasksets with datasets and graders.
See [`../../README.md`](../../README.md) for the general RLE contract.

## What's here

| File | Purpose |
| --- | --- |
| `server/echo_environment.py` | `EchoEnvironment`, an `MCPEnvironment` exposing the `echo_message` and `echo_with_length` tools. |
| `server/app.py` | OpenEnv-compatible FastAPI server with HTTP and WebSocket support for MCP actions. |
| `server/Dockerfile` | Dataset-free runtime image that installs OpenEnv and server dependencies only. |
| `rle.toml` | Host-agnostic RLE identity and control-plane interface (`Gym` / `OpenEnv`). |

## MCP tool contract

| Tool | Arguments | Result |
| --- | --- | --- |
| `echo_message` | `{"message": "<text>"}` | The same text. |
| `echo_with_length` | `{"message": "<text>"}` | `{"message": "<text>", "length": <character-count>}`. |

An agent sends a `ListToolsAction` to discover the tool schemas, then a
`CallToolAction` to invoke one. The JSON action forms are:

```json
{"type":"list_tools"}
```

```json
{"type":"call_tool","tool_name":"echo_message","arguments":{"message":"Hello"}}
```

## Build and run locally

```bash
cd envs/echo_env
azd ai rle run --dockerfile server/Dockerfile
```

This starts an OpenEnv-compatible local runtime at the printed URL and opens a
local playground. There is no task dataset to download or bake; the image
build only retrieves its base image and Python dependencies.

### Copy-paste smoke test

After `azd ai rle run` opens the `rle>` shell, enter these commands
separately. Do not type the `rle>` prompt itself.

```text
reset {}

step {"type":"list_tools"}

step {"type":"call_tool","tool_name":"echo_with_length","arguments":{"message":"OpenEnv"}}
```

The final response includes the structured result
`{"message":"OpenEnv","length":7}` and remains `done: false`.

Needs Docker running and the `azd` RLE extension.

The checked-in manifest declares the initial `echo_env` release as version
`1.0.0`. Update its name when copying this source outside `azd ai rle init`,
and update its version before publishing a subsequent release.

## Iterate with your agent, run, publish, and invoke

Your OpenEnv-compatible agent can drive the same MCP lifecycle as the local
playground: optionally send `reset`, send `{"type":"list_tools"}` to discover
the available tools, then send a `CallToolAction` with the tool name and
arguments.

1. **Iterate locally with your agent.** Edit the environment or your agent,
   then start the local runtime with `--watch`. Point the agent at the URL
   printed by the command. It rebuilds and restarts the container when the
   environment source changes; reconnect the agent after a restart.

   ```bash
   azd ai rle run --watch --dockerfile server/Dockerfile
   ```

2. **Publish an immutable version.** Set your Foundry project endpoint and
   Azure Container Registry endpoint, then publish the name and version from
   `rle.toml`. The command builds the image, pushes it to the registry, and
   registers the environment.

   ```bash
   export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
   export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
   azd ai rle publish --dockerfile server/Dockerfile
   ```

3. **Invoke the published environment.** With
   `FOUNDRY_PROJECT_ENDPOINT` still set, `invoke` reads `rle.name` and
   `rle.version` from `rle.toml`, starts a remote runtime, and opens the
   interactive `rle>` shell. Use the same MCP action commands as the local
   shell.

   ```bash
   azd ai rle invoke
   ```

## Use the published environment for SDK sanity testing

`echo_env` deliberately has no task distribution, terminal success condition,
reward signal, or grader. Its tool calls remain non-terminal (`done: false`).
It is therefore not compatible with the training or task-quality evaluation
loops used by `code_rl` and `math_rl`; do not use its echo output as an RL
reward or evaluation metric.

Use a published Echo version to verify that your project credentials, RLE
environment registration, managed instance lease, and OpenEnv MCP actions work
end to end. For the complete SDK contract, see the
[RLE OpenEnv/Gym guide](https://aka.ms/rle).

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export RLE_ENV_NAME="echo_env"
export RLE_ENV_VERSION="1.0.0"

pip install --force-reinstall \
  "https://rle-onboarding-docs.orangeground-ba9696de.eastus2.azurecontainerapps.io/downloads/azure_ai_projects-2.6.0-py3-none-any.whl" \
  azure-identity aiohttp
```

```python
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


with DefaultAzureCredential() as credential:
    with AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=credential,
        allow_preview=True,
    ) as project_client:
        with project_client.rle.get_openenv_client(
            name=os.environ["RLE_ENV_NAME"],
            version=os.environ["RLE_ENV_VERSION"],
            max_active_instances=1,
            instance_acquire_timeout=900,
        ) as openenv_client:
            with openenv_client.get_instance() as instance:
                instance.reset()
                tools = instance.step({"type": "list_tools"})
                echo = instance.step(
                    {
                        "type": "call_tool",
                        "tool_name": "echo_with_length",
                        "arguments": {"message": "OpenEnv"},
                    }
                )

                print(tools.observation)
                print(echo.observation)  # Contains {"message": "OpenEnv", "length": 7}.
                print(echo.done)  # False
```

The client contexts release the instance and instance group after the test.
Move to [`code_rl`](../code_rl/README.md) or
[`math_rl`](../math_rl/README.md) when you need a dataset-backed, graded
environment for a trainer loop or evaluation.
