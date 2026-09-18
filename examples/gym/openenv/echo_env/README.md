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

## Environment behavior

- `reset()` starts a new episode and returns an empty observation with reward
  `0.0`.
- Tool calls return `CallToolObservation` values containing either a tool
  result or an error.
- Tool calls are non-terminal (`done: false`) and do not assign a training
  reward.
