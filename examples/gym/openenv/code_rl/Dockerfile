# RLE environment server for `code_rl`.
#
# Self-contained: this whole folder is the Docker build context, with no
# dependency on anything outside it. `server/dataset.py` and
# `server/code_grading.py`/`server/lcb_utils.py` implement dataset loading
# and grading directly, so the runtime image only needs the handful of
# packages the server actually imports (`openenv`, `numpy`) rather than a
# full training stack (torch, transformers, CUDA wheels) which the server
# never touches.
#
# The compact, checked-in data snapshot contains 900 training tasks and 100
# validation tasks. It avoids a multi-gigabyte Hugging Face download during
# every developer build.
#
# Tool checks and final grading launch a disposable child process, so the
# image needs no external sandbox service or runtime egress.
#
#   cd examples/gym/openenv/code_rl && docker build -t ft-code-rl-env:latest .
#   docker run -p 8101:8000 ft-code-rl-env:latest

# Package index for this build. Defaults to public PyPI and can be overridden
# by the container builder.
ARG PIP_INDEX_URL=https://pypi.org/simple

FROM python:3.12-slim

ARG PIP_INDEX_URL

WORKDIR /app

ARG AZURE_AI_PROJECTS_RLE_REF=2520ed7b6606f0e6eaff24affa49e888dbb7732e
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
      "azure-ai-projects[rle] @ git+https://github.com/sujit-kamireddy/azure-sdk-for-python@${AZURE_AI_PROJECTS_RLE_REF}#subdirectory=sdk/ai/azure-ai-projects" \
      "openenv==0.6.0" \
      numpy

COPY . /app/envs/code_rl

ENV CODE_RL_DATASET_PATH=/app/envs/code_rl/env_data/train.jsonl.gz \
    CODE_RL_VALIDATION_DATASET_PATH=/app/envs/code_rl/env_data/validation.jsonl.gz \
    PYTHONPATH=/app

# Fail at build time, not at the first rollout. This also verifies the
# compressed sample data can be loaded without any network access.
RUN python -c "from azure.ai.projects.rle.environments import RLEnvironment; import envs.code_rl.server.app as m; print('code_rl MCP server OK:', m.app)"

# Foundry RLE forwards to port 8000 inside the container, so the server must listen there.
# The host-side port below only matters when you run several environments locally at once.
# One worker, on purpose: the SDK MCP session retains explicit episode state
# on one environment instance (see the concurrency configuration in app.py).
EXPOSE 8000
CMD ["uvicorn", "envs.code_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
