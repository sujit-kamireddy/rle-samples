## Sample anatomy

Every sample under `examples/gym/openenv/` has the same shape. Copy it; the layout is what
`azd ai rle init` produces and what `azd ai rle publish` expects.

```
<env_name>/
  rle.toml                  # the RLE manifest — the only file the service reads directly
  Dockerfile                # builds the image; must serve the OpenEnv app
  .dockerignore             # keep env_data/ in, keep build junk out
  README.md                 # what the episode is, how it is graded, how to run it
  __init__.py
  env_data/                 # dataset baked INTO the image + NOTICE + source.json provenance
  job_data/                 # training-job manifests that reference the published environment
  scripts/                  # dataset build/refresh scripts (not run at rollout time)
  server/
    app.py                  # create_fastapi_app(...) — the ASGI entry point
    schema.py               # Action / Observation pydantic models -> GET /schema
    dataset.py              # loading, episode selection, selector validation
    grading.py              # scoring, kept separate from the environment loop
    <env_name>_environment.py  # Environment subclass: reset() / step()
```

### What goes where, and why it matters

**`env_data/` is baked into the image.** A rollout container has no guaranteed egress and no
mounted volume; a dataset fetched at `reset()` time is a rollout that fails in production and
passes locally. Keep provenance next to it (`NOTICE`, `source.json`) so the licence of the data
travels with the image.

**`schema.py` is a public contract, not an implementation detail.** It is serialized to
`GET /schema` and becomes the model's entire action vocabulary. Docstrings and `Field(description=)`
text are sent to the model. Write them as instructions to the policy, not as notes to a maintainer —
see `examples/gym/openenv/math_rl/server/schema.py`.

**`grading.py` is separate from the environment on purpose.** Grading is the part you will want to
unit-test against fixtures without standing up a server, and the part a second environment may
reuse.

**`job_data/` is not used by a rollout.** It holds the training-job manifests that point at the
published environment version. Changing it never changes rollout behavior.

### Adding a new sample to this repo

`examples/gym/openenv/catalog.toml` controls which sample directories `azd ai rle init` shows in its
picker. A directory with no `[[sample]]` entry defaults to visible, so a new sample is discoverable
automatically — add an entry with `visible = false` to stage one that is not ready for users yet.
