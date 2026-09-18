# RLE environment server for the `math_rl` recipe.
#
# Self-contained: this whole folder is the Docker build context. `_common/`
# is a vendored copy of `envs/_common/` (kept identical across all four
# recipes -- see `tests/test_common_parity.py`), so this build needs nothing
# from outside this directory.
#
# Also independent of `loom_cookbook`: the grading logic lives in
# `grading/` and the dataset loader in `dataset_source.py`, both vendored from
# the recipe. That keeps the runtime image to the handful of packages the
# server actually imports (sympy/pylatexenc for grading) rather than the recipe's full
# training stack (torch, transformers, CUDA wheels) which the server never
# touches.
#
# Two stages: the `dataset` stage bakes `train.jsonl`/`validation.jsonl` at
# *build time* (see `build_dataset.py`), and only those files cross into the
# runtime stage -- `datasets`/`pyarrow`/`pandas` are needed to write them but
# never to serve them, so they stay out of the shipped image.
#
# Grading defaults to the `sympy` backend, which is always installed. The
# optional `math_verify` backend (`grader="math_verify"`) is a lazily
# imported extra, not installed here -- see `math_grading.grade_answer_math_verify`.
#
#   cd examples/gym/openenv/math_rl && docker build -t ft-math-rl-env:latest .
#   docker run -p 8100:8000 ft-math-rl-env:latest

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
ARG MAX_TRAIN_EXAMPLES=4000
ARG MAX_VALIDATION_EXAMPLES=500

RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
        export UV_DEFAULT_INDEX="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache datasets

COPY . /app/envs/math_rl

ENV PYTHONPATH=/app

RUN python /app/envs/math_rl/build_dataset.py \
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
# server in `server/app.py` and `server/math_rl_environment.py` builds on --
# and sympy/pylatexenc for the vendored grader (`grading/math_grading.py`).
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
        export UV_DEFAULT_INDEX="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        openenv sympy pylatexenc

COPY . /app/envs/math_rl
COPY --from=dataset /out/train.jsonl /out/validation.jsonl /app/envs/math_rl/

ENV MATH_RL_DATASET_PATH=/app/envs/math_rl/train.jsonl \
    PYTHONPATH=/app

# Fail at build time, not at the first rollout.
RUN python -c "import envs.math_rl.server.app as m; print('math_rl OpenEnv server OK:', m.app)"

# Foundry RLE forwards to port 8000 inside the container, so the server must listen there.
# The host-side port below only matters when you run several environments locally at once.
# One worker, on purpose: the environment instance lives in the server process
# (see `openenv.core.env_server.http_server.create_fastapi_app`'s
# `max_concurrent_envs=1` in `server/app.py`), so a second worker would be
# serving a second, unrelated episode.
EXPOSE 8000
CMD ["uvicorn", "envs.math_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
