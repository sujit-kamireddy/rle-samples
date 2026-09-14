# RLE environment server for the `code_rl` recipe.
#
# Self-contained: this whole folder is the Docker build context. `_common/`
# is a vendored copy of `envs/_common/` (kept identical across all four
# recipes -- see `tests/test_common_parity.py`), so this build needs nothing
# from outside this directory.
#
# Also independent of `loom_cookbook`: the grading logic is vendored under
# `grading/`. That keeps the runtime image to the handful of packages the
# server actually imports rather than the recipe's full training stack
# (torch, transformers, CUDA wheels) which the server never touches.
#
# The compact, checked-in data snapshot contains 900 training tasks and 100
# validation tasks. It avoids a multi-gigabyte Hugging Face download during
# every developer build; `tools/generate_compact_dataset.py` is only a
# maintainer utility for refreshing that snapshot.
#
# `step()` grades by calling the test runner directly, in-process, so the
# image needs no sandbox service and no egress at run time.
#
#   cd envs/code_rl && docker build -t ft-code-rl-env:latest .
#   docker run -p 8101:8000 ft-code-rl-env:latest

# Package index for this build. Defaults to public PyPI; override it where that
# is unreachable -- see "Build fails downloading Python packages" in
# ../README.md. Both are needed: `pip` bootstraps `uv`, `uv` installs the rest.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=${PIP_INDEX_URL}

# Index to fall back to when the one above cannot be reached. `azd ai rle run`
# and `azd ai rle publish` take no --build-arg, so a build that needs a
# different index has no way to be told about one -- without this fallback the
# local-iteration path the README documents is unusable wherever public PyPI is
# blocked. Set it empty to disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.11-slim

ARG PIP_INDEX_URL
ARG UV_DEFAULT_INDEX
ARG PIP_FALLBACK_INDEX_URL

WORKDIR /app

# Everything the server imports: `openenv`, whose real `Environment`/
# `HTTPEnvServer` (`fastapi`/`uvicorn` included as its own dependencies) the
# server in `server/app.py` and `server/code_rl_environment.py` builds on --
# and `numpy` for the vendored grader (`grading/lcb_utils.py`).
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
        export UV_DEFAULT_INDEX="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        openenv numpy

COPY . /app/envs/code_rl

ENV CODE_RL_DATASET_PATH=/app/envs/code_rl/data/train.jsonl.gz \
    CODE_RL_VALIDATION_DATASET_PATH=/app/envs/code_rl/data/validation.jsonl.gz \
    PYTHONPATH=/app

# Fail at build time, not at the first rollout. This also verifies the
# compressed sample data can be loaded without any network access.
RUN python -c "import envs.code_rl.server.app as m; print('code_rl OpenEnv server OK:', m.app)"

# Foundry RLE forwards to port 8000 inside the container, so the server must listen there.
# The host-side port below only matters when you run several environments locally at once.
# One worker, on purpose: the environment instance lives in the server process
# (see `openenv.core.env_server.http_server.create_fastapi_app`'s
# `max_concurrent_envs=1` in `server/app.py`), so a second worker would be
# serving a second, unrelated episode.
EXPOSE 8000
CMD ["uvicorn", "envs.code_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
