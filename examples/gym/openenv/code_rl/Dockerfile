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

ENV AZURE_AI_PROJECTS_VERSION=latest
COPY requirements.txt /app/
ADD --checksum=sha256:86bcd508cbca5ca978bf201ed67a1bcc1362c49a16bf2031c995f107c27c5f52 https://raw.githubusercontent.com/sujit-kamireddy/rle-samples/b7c131b6ce0d9e33b492b782c07a6cb2a97c9222/tools/install_rle_sdk.py /app/install_rle_sdk.py
# Resolve the public branch commit; raw HTTP manifests can be CDN-cached.
ADD https://github.com/Azure/azure-sdk-for-python.git#rle-releases:sdk/ai/azure-ai-projects/rle /app/rle-sdk-dist/
RUN python /app/install_rle_sdk.py --manifest /app/rle-sdk-dist/registry.json \
    --requirements /app/requirements.txt

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
