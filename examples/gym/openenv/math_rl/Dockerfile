# RLE environment server for `math_rl`.
#
# Self-contained: this whole folder is the Docker build context, with no
# dependency on anything outside it. `server/dataset.py` and
# `server/grading.py` implement dataset loading and grading directly, so the
# runtime image only needs the handful of packages the server actually
# imports (sympy/pylatexenc for grading) rather than a full training stack
# (torch, transformers, CUDA wheels) which the server never touches.
#
# The compact, checked-in `env_data/train.jsonl.gz`/`validation.jsonl.gz` snapshot
# (see `env_data/NOTICE.md`) is baked in ahead of time by `scripts/build_dataset.py`, run
# once against a real network/Hugging Face connection -- not at `docker
# build` time -- so this build needs no dataset-network access.
# Re-run `scripts/build_dataset.py` and re-gzip its output into `env_data/` to refresh
# the snapshot.
#
# Grading defaults to the `sympy` backend, which is always installed. The
# optional `math_verify` backend (`grader="math_verify"`) is a lazily
# imported extra, not installed here -- see `grading.grade_answer_math_verify`.
#
#   cd examples/gym/openenv/math_rl && docker build -t ft-math-rl-env:latest .
#   docker run -p 8100:8000 ft-math-rl-env:latest

# Package index for this build. Defaults to public PyPI and can be overridden
# by the container builder.
ARG PIP_INDEX_URL=https://pypi.org/simple

FROM python:3.12-slim

ARG PIP_INDEX_URL

WORKDIR /app

# This immutable SDK source revision
# reports 2.7.0 and contains RLEnvironment, while the public 2.7.0 wheel does
# not yet contain azure.ai.projects.rle.
ARG AZURE_AI_PROJECTS_RLE_REF=05ba0ba72d3de0cfe0c742653da9eb87e9e02730
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
      "azure-ai-projects[rle] @ git+https://github.com/sujit-kamireddy/azure-sdk-for-python@${AZURE_AI_PROJECTS_RLE_REF}#subdirectory=sdk/ai/azure-ai-projects" \
      "openenv==0.6.0" \
      sympy \
      pylatexenc

COPY . /app/envs/math_rl

ENV PYTHONPATH=/app

# Fail at build time, not at the first rollout. This also verifies the
# checked-in dataset snapshot can be loaded without any network access.
RUN python -c "import envs.math_rl.server.app as m; print('math_rl OpenEnv server OK:', m.app)"

# Foundry RLE forwards to port 8000 inside the container, so the server must listen there.
# The host-side port below only matters when you run several environments locally at once.
# One worker preserves same-instance MCP tool and grading state.
EXPOSE 8000
CMD ["uvicorn", "envs.math_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
