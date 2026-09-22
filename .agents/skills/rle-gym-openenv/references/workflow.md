## The authoring loop

Work in this order. Each step catches a class of failure that the next step cannot.

### 1. Start from a sample

```bash
azd ai rle init <folder-name> --type Gym --subtype OpenEnv
```

This copies a working sample out of the samples repo. Pick the closest one:

| If the episode is… | Copy |
| --- | --- |
| one prompt, one graded answer, no tools | `math_rl` |
| multi-turn with a tool the policy may call | `code_rl` |
| grounded in a real repository checkout | `code_repair` |

### 2. Adapt, in dependency order

1. `env_data/` — bake the dataset in, with its `NOTICE`/`source.json`.
2. `server/dataset.py` — loading, `EpisodePicker`, `reject_unknown_selectors`.
3. `server/schema.py` — the action vocabulary (response variant, plus a union member per tool).
4. `server/grading.py` — scoring, testable on its own.
5. `server/<name>_environment.py` — `reset()` / `step()`.
6. `rle.toml` — name, version, `model_response_field`, step and token budgets.
7. `README.md` — what the episode is and how it is graded.

### 3. Run it locally

```bash
azd ai rle run            # builds and serves the container; --watch to rebuild on change
curl -s localhost:8000/schema | jq '.action'
```

Check on the `/schema` output *before* publishing:

- exactly one variant carries your `model_response_field`
- every other variant has a discriminator value, and those are the tool names you expect
- descriptions read as instructions to the model

Then drive a scripted `reset` + `step` over `/ws` for a known seed and assert the reward. This is
where logic bugs surface — cheaply, without a model in the loop.

Also test the selector guard: send a reset with a selector the environment does not declare, and
confirm it **raises** rather than quietly returning an episode.

### 4. Publish

```bash
azd ai rle publish
azd ai rle list
azd ai rle show <name>
```

Publish validates the manifest. `list` confirms the version landed in the project named by
`FOUNDRY_PROJECT_ENDPOINT`.

Note: `show` renders a typed view of the environment and may not display every manifest key it
stored. Treat `show` as a convenience, not as proof of what was persisted — if you need certainty
about a key, check the version's stored defaults directly.

### 5. Roll out against a real model

```bash
azd ai rle rollout <name> --version <version> --model <model> --task '{"seed": 0}'
```

`--task-file` takes the same payload from a file. This is the first step that exercises the model,
the capture proxy, the action vocabulary and grading together.

Verify three things in the result, not just that it returned 200:

1. the graded episode is the one `--task` asked for
2. the reward is the value your local scripted run produced for that seed
3. no completion was truncated at the token budget

Repeat across several seeds — a single passing seed hides both a random-episode selector bug and a
grader that only handles one row shape.

### CLI gating

`azd ai rle` commands are feature-gated: `AZD_AI_RLE_ENABLE=true` enables the core set, and
`AZD_AI_RLE_ENABLE_ALL=true` additionally enables commands still in development. If a documented
command reports as unknown, check the gate before assuming the CLI is out of date.
