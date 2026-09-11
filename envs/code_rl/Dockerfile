# RLE environment server for the `code_rl` recipe.
#
# Self-contained: this whole folder is the Docker build context. `_common/`
# is a vendored copy of `envs/_common/` (kept identical across all four
# recipes -- see `tests/test_common_parity.py`), so this build needs nothing
# from outside this directory.
#
# Also independent of `loom_cookbook`: the grading logic lives in
# `grading/` and the task loader in `dataset_source.py`, both vendored from the
# recipe. That keeps the runtime image to the handful of packages the server actually
# imports rather than the recipe's full training stack (torch, transformers,
# CUDA wheels) which the server never touches.
#
# Two stages: the `dataset` stage bakes `train.jsonl`/`validation.jsonl` at
# *build time* (see `build_dataset.py`), and only those files cross into the
# runtime stage -- `datasets`/`pyarrow`/`pandas` are needed to write them but
# never to serve them, so they stay out of the shipped image.
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
# local-iteration path the READMEs document is unusable wherever public PyPI is
# blocked. Set it empty to disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/


FROM python:3.11-slim AS dataset

ARG PIP_INDEX_URL
ARG UV_DEFAULT_INDEX
ARG PIP_FALLBACK_INDEX_URL
ARG MAX_TRAIN_EXAMPLES=2000
ARG MAX_VALIDATION_EXAMPLES=200

RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
        export UV_DEFAULT_INDEX="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache datasets numpy

COPY . /app/envs/code_rl

ENV PYTHONPATH=/app

# The HF download dominates this build: 7.8 GB of parquet for the 2200
# examples actually kept (`lcbv5` alone is 5.8 GB for 878 rows). Caching it
# outside the layer means editing anything under `envs/` rebuilds in seconds
# instead of re-downloading. The cache never reaches the runtime stage.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python /app/envs/code_rl/build_dataset.py \
        --out-dir /out \
        --max-train ${MAX_TRAIN_EXAMPLES} \
        --max-validation ${MAX_VALIDATION_EXAMPLES}


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
COPY --from=dataset /out/train.jsonl /out/validation.jsonl /app/envs/code_rl/

ENV CODE_RL_DATASET_PATH=/app/envs/code_rl/train.jsonl \
    PYTHONPATH=/app

# Fail at build time, not at the first rollout.
RUN python -c "import envs.code_rl.server.app as m; print('code_rl OpenEnv server OK:', m.app)"

# Foundry RLE forwards to port 8000 inside the container, so the server must listen there.
# The host-side port below only matters when you run several environments locally at once.
# One worker, on purpose: the environment instance lives in the server process
# (see `openenv.core.env_server.http_server.create_fastapi_app`'s
# `max_concurrent_envs=1` in `server/app.py`), so a second worker would be
# serving a second, unrelated episode.
EXPOSE 8000
CMD ["uvicorn", "envs.code_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
