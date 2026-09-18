# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `echo_env` OpenEnv MCP environment

Self-hosted OpenEnv environment that demonstrates tool discovery and tool
calls. It has no task dataset, grader, terminal success condition, or training
reward.

## What's here

| File | Purpose |
| --- | --- |
| `server/echo_environment.py` | `EchoEnvironment`, exposing the `echo_message` and `echo_with_length` tools. |
| `server/app.py` | FastAPI server exposing the OpenEnv HTTP and WebSocket APIs. |
| `Dockerfile` | Container image for the environment server. |
| `rle.toml` | Environment identity and OpenEnv interface metadata. |

## MCP tool contract

| Tool | Arguments | Result |
| --- | --- | --- |
| `echo_message` | `{"message": "<text>"}` | The same text. |
| `echo_with_length` | `{"message": "<text>"}` | `{"message": "<text>", "length": <character-count>}`. |

Example actions:

```json
{"type":"list_tools"}
```

```json
{"type":"call_tool","tool_name":"echo_message","arguments":{"message":"Hello"}}
```

## Build and run locally

From this directory:

```bash
docker build -t echo-env:latest .
docker run --rm -p 8000:8000 echo-env:latest
```

The OpenEnv API is available at `http://localhost:8000`. Connect with an
OpenEnv client, reset the environment, and then submit MCP actions.

The server intentionally exposes only the OpenEnv runtime API and does not
provide an environment-specific `/web` interface.

## Publish and invoke

1. **Publish an immutable version.** Set your Foundry project endpoint and
   Azure Container Registry endpoint, then publish the name and version from
   `rle.toml`.

   ```bash
   export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
   export AZURE_CONTAINER_REGISTRY_ENDPOINT="<registry>.azurecr.io"
   azd ai rle publish
   ```

2. **Execute one rollout of the published environment.** `invoke` reads
   `rle.name`/`rle.version` from `rle.toml`, provisions a real Loom training
   session and sampler checkpoint for `--model`, calls Execute Rollout, and
   prints the reward. `reset()` ignores any task payload, so `--task '{}'` is
   enough.

   ```bash
   azd ai rle invoke --model Qwen/Qwen3-32B --task '{}'
   ```

## Environment behavior

- `reset()` starts a new episode and returns an empty observation with reward
  `0.0`.
- Tool calls return `CallToolObservation` values containing either a tool
  result or an error.
- Tool calls are non-terminal (`done: false`) and do not assign a training
  reward.
