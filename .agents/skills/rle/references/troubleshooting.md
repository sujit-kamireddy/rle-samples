# Troubleshooting and migration checks

| Symptom | Check / fix |
| --- | --- |
| `azd` says unknown command `ai` | Check compatible azd host and installation of `azure.ai.rle` from the intended extension source. Environment gates do not install commands. |
| Harness init or `train` missing | Use `AZD_AI_RLE_ENABLE_ALL=true` for these staged surfaces. Core RLE commands no longer need `AZD_AI_RLE_ENABLE`. |
| `rollout` rejects `--mode eval` | Inspect installed help and extension channel. Public `0.8.29-preview` is Loom-only; the verified development source adds eval. Do not remove the flag and send a deployment name to Loom. |
| RLE list/publish returns HTTP 408 before reset | Preserve request ID, endpoint and operation; confirm service access before changing environment code. A working model endpoint or registry push does not establish RLE control-plane health. After an ambiguous write, check resource/version persistence before retrying. |
| Loom returns HTTP 503 `engine_busy` | No sampler capacity is available for the requested base model. Honor bounded retry/backoff; do not fabricate IDs or treat this as an environment/grader failure. Close owned sessions, including queued sessions recovered after a timeout. |
| WebSocket upgrades but Foundry policy gets `InvalidRolloutRequest` | Verify deployed service capability independently of CLI/development source. An explicit pre-admission payload rejection can be diagnosed using the HTTP API with a new correlation ID; do not resubmit an accepted or uncertain execution. HTTP model-binding errors can identify an unsupported `policy.deployment_name`; semantic validation may state that only `loom` is supported. Do not fall back to Loom with a deployment name. |
| HostedAgent create returns dependent-resource 403 | Check the actual resource and service identity, registry connection and encryption configuration. CMK coverage limitations alone do not prove the cause; do not grant speculative broad roles or disable encryption. |
| HostedAgent create returns `hosted_agent_infrastructure_provisioning_failed` | Preserve HTTP status/request ID, check version persistence and supported deployment prerequisites, then escalate persistent infrastructure failure. Local host tests or an image-format change do not establish deployment success. |
| Manifest rejects type/default keys | Use lowercase `[harness] type = "gym"` or `"byoh"` with endpoint/protocol, `[model]`, `[train.grpo]`. Removed subtype/agent_name/agent_version/invocation_url keys are rejected. Match the CLI to the current schema; do not downgrade samples to an older release's fields. |
| Obsolete `model_response_field`/protocol/budget key | Migrate to SDK `RLEnvironment`, MCP tools and `GradeAction`; remove legacy routing fields instead of recreating an action union. |
| Cannot import `azure.ai.projects.rle.environments` or SDK `create_app` | Install a registry-listed RLE SDK wheel with the sample installer; verify checksum and rebuild the image. Same public version number need not imply the same modules. |
| MCP tool absent or vague schema | Register after `super().__init__()`; check `tools/list`, concrete annotations and descriptions. |
| Tool worked but grader sees no evidence | Preserve full supplied MCP URL/query; use SDK wrapper for HTTP query routing; reset/tools/grade must share one session. Persist copied SDK state with `_set_state`. |
| Correct grade belongs to wrong task | Capture and validate `**kwargs` in reset; test explicit selectors and first-turn parsing. Never default failed input parsing to another row. |
| Hosted agent researches `"None"` | Unwrap serialized chat messages, validate prompt keys and reject missing task input before string conversion. |
| HTTP 200 but MCP failed | Check JSON-RPC `error`, OpenEnv `is_error` and standard MCP `isError`, not just HTTP status. Test the concrete adapter transport, not only fake clients. |
| BYOH contract failure at start | Return 202 immediately, use RLE's `operation_id`, not a harness-generated `invocation_id` or synchronous 200 result. |
| BYOH poll 404 or never finishes | Resource is `<endpoint>/rollouts/{operation_id}` for `rle-invocation`; key state by operation, retain result until polled, log async failures safely. |
| BYOH auth fails after adding a secret header requirement | RLE has no automatic arbitrary invocation auth header. Verify an explicitly supported ingress arrangement; don't invent credentials it cannot send. |
| HostedAgent has missing rollout context | Read request-local `x-client-rle-*` headers; old `sandbox-tools-*` headers and startup env vars are not the current contract. |
| HostedAgent accepted but final response never arrives | Check background Responses status/retrieval and stored response ID. The inspected service uses `background:true, store:true`; old synchronous-only guidance may not match the deployed revision. |
| HostedAgent background response fails with a masked internal error | Inspect bounded runtime logs for the exact owned agent/version/session. A missing `x-client-rle-*` header can be explicit in the handler logs while the client only sees a generic error. Never require the public API to expose internal exception text or print header values. |
| Capture Proxy registration/deletion returns HTTP 503 | Diagnose the reported capture dependency, preserve rollout correlation and cleanup outcome, and use bounded backoff. Reward available is not proof cleanup completed; do not grant speculative registry roles or change the grader. Retry only confirmed failed/rejected attempts with a new rollout ID. |
| Successful managed episode but evidence says no model nodes | Check the capture-export schema: the verified exporter uses `turns` and `sequences`, not `nodes`. Preserve safe structural evidence before assertions, validate actual model turns and token/logprob trainability, and report service completion separately from verifier failure. |
| Harness terminal grade is rejected | Return `done:true`, finite numeric reward and boolean `is_success` in the terminal observation. |
| Foundry eval capture says not trainable | Expected for `--mode eval` without `--loom`: it is text-only. Use Loom-backed rollout for training data. |
| Rollout answer is fine but capture untrainable | Use the supplied capture-proxy model URL/key for every policy call; inspect actual captured calls and terminal grading. |
| `degenerate_rollout` | Check terminal `done`/reward and model capture. Don't add artificial turns to a valid graded single-turn episode. |
| Truncated boxed/fenced answer | Inspect token usage/finish reason and service limits; shorten expected output. No current sample manifest token knob exists. |
| Early Gym termination | Check service turn ceiling vs actual tool/model turns; training `max_steps` is unrelated. |
| `SessionCapacityError` | Clean up detached sessions; close after WebSocket detach. Don't add workers or silently overcommit. |
| Tool/grade loses connection | Avoid blocking the event loop, preserve WebSocket keepalive, bound grader runtime, and don't send unsolicited frames. |
| Version registered but disk image failed | Check image readiness, registry pull permission on project identity, build context and Docker BuildKit output. Local ACR login only covers publishing. A generic ADC 400 image-pull hint is not proof of a missing role; retain the underlying error if verified corrections do not resolve it. |
| Works locally, fails managed | Check baked data/dependencies, runtime egress assumptions, build context, SDK version, and per-request credentials before blaming the grader. |
| `EnvironmentContractViolation` | Read the specific key/protocol/status violation; compare current SDK/sample contract with the published immutable version. |
| `RolloutDependencyFailed` | Preserve rollout ID and safe logs; check agent, model proxy, MCP and grading independently. It is not proof the environment is correct or that every such error is service-side. |

## Diagnose by boundary

1. **Build/start:** SDK installation, imports, Docker context, worker count,
   capacity and health.
2. **Before model calls:** manifest binding, reset task validation, observation
   rendering and agent invocation acknowledgement.
3. **During agent work:** request-local proxy/MCP wiring, protocol results,
   session identity, model and total-agent timeouts.
4. **At grading:** final-text extraction, private task identity, finite reward,
   terminal lifecycle and safe grader errors.
5. **At export/cleanup:** captured model calls (trainable when using Loom),
   graded termination, session close and BYOH withdrawal.

Reproduce one known task locally at the failing boundary. Capture timestamps,
task identity, error class and credential **presence**, not token values or
private routing URLs. Escalate with rollout correlation and sanitized logs only
after identifying which boundary fails.
