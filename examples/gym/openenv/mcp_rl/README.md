# `mcp_rl`: SDK `RLEnvironment` Gym sample

This is the smallest self-contained Gym/OpenEnv sample for the SDK's MCP
authoring mode. The reset task `{"left":2,"right":3}` presents one task:
call the typed `add(left: int, right: int)` tool, then submit the integer answer.

Unlike the schema-driven `math_rl` and `code_rl` samples, this environment:

- subclasses `azure.ai.projects.rle.environments.RLEnvironment`;
- registers typed MCP tools with `self.tool()`;
- receives the terminal answer as `GradeAction`;
- sets top-level `environment_protocol = "mcp_environment"`; and
- deliberately has no `model_response_field`.

`ArithmeticState` records `instance_id`, `tool_called`, `tool_call_count`, and
`last_tool_result`. The reward is `1` only when the final answer is exactly
`5` **and** `add` produced `5` on that same environment instance; otherwise it
is `0`.

## Layout

| Path | Purpose |
| --- | --- |
| `server/environment.py` | deterministic reset, typed tool, and grader |
| `server/models.py` | OpenEnv observation and public per-instance state |
| `server/app.py` | single-session ASGI app built with `create_app` |
| `job_data/` | deterministic training and validation selectors |
| `tests/` | reward, typed-tool, state-isolation, and manifest checks |

## Run

```bash
docker build -t mcp-rl .
docker run --rm -p 8000:8000 mcp-rl
```

The Dockerfile pins `openenv==0.6.0` and an immutable SDK source revision whose
metadata reports `azure-ai-projects` 2.7.0 and whose tree contains
`RLEnvironment`. Repository evidence also records that the released public
2.7.0 wheel does **not** contain `azure.ai.projects.rle`, so the source pin
cannot yet be replaced with `azure-ai-projects[rle]==2.7.0`. Replace it with a
normal exact package pin once an RLE-enabled wheel is published.

For local tests, install those same dependencies, then run:

```bash
python -m pytest -q tests
```

The dependency-free contract tests still run when the private wheel is
unreachable; runtime tests report a skip rather than testing against a public
SDK that lacks `RLEnvironment`.
