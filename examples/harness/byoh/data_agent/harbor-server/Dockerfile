FROM python:3.12-slim

# Package index for this build. Defaults to public PyPI; override it where that
# is unreachable -- see "Build fails downloading Python packages" in
# ../../../gym/openenv/README.md.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. Set it empty to
# disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

# git: `harbor` sandboxes and task fetch shell out to it. build-essential: a couple of the
# per-sandbox extras pull packages that need to compile.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Repeat the defaults, don't just re-declare the names. BuildKit carries a
# stage's ARG default across a re-declaration; the classic builder (which is
# what `az acr build` runs) resets it to empty, which silently turns
# --index-url into "" and fails the install below.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" uv

# The agent's own toolbox. Every task instruction promises these are installed
# ("Installed: pandas, numpy, ... pip install more if needed"), and on the
# Harbor path they came from the task's own base image. The agent runs in this
# container now, so they have to be here instead.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        pandas numpy matplotlib seaborn scipy scikit-learn statsmodels tabulate plotly

# `opencode` is the agent. Installing it at build time rather than per rollout
# is the main latency win over the Harbor path, which ran `npm i -g opencode-ai`
# inside every freshly booted sandbox.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm i -g opencode-ai@latest \
    && opencode --version

# The fetcher that pulls a task's input files from the CDN. On the Harbor path
# this was COPY'd into each task's sandbox image; here it runs as a subprocess
# of the server, once per rollout. `vendor/pull_bucket.py` is a generated copy
# of `tools/pull_bucket.py` (Docker cannot COPY above the build context);
# `tools/build_task_index.py --write` refreshes it.
COPY vendor/pull_bucket.py /opt/pull_bucket.py

WORKDIR /app
COPY pyproject.toml .
COPY server ./server

# openenv releases faster than the mirrored package feed proxy can catch up
# (it was still stuck on 0.4.2 when this was written, well behind the >=0.5.0
# this sample needs), so the wheel is vendored here -- downloaded straight
# from PyPI on a host that isn't behind that proxy -- rather than pinned to
# whatever the proxy happens to have. `--find-links` lets uv pick it up
# without ever hitting the network for this one package; everything else
# still resolves normally (falling back to the proxy) below.
COPY vendor/wheels ./vendor/wheels

# No lockfile checked in yet (phase 1); resolve fresh. Pin a uv.lock here once the dependency
# set settles, the way OpenEnv/envs/harbor_env does, so rebuilds stop re-resolving.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    uv pip install --system --no-cache --index-url "${PIP_INDEX_URL}" --find-links ./vendor/wheels .

# The task suite is deliberately NOT baked into this image.
#
# It used to be, because Harbor needed real task directories to build a sandbox
# image and run its own verifier from. The direct path needs none of that -- it
# needs each task's instruction, its bucket coordinates and its timeout, which
# is what `vendor/task-index.json.gz` holds (0.4MB, vs 216MB extracted).
#
# This is a security boundary, not a diet. Every `task.toml` in the suite
# carries `metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, and every
# instruction quotes its question verbatim while `task.description` repeats it.
# An agent with a shell -- which is exactly what `opencode
# --dangerously-skip-permissions` is -- can grep the suite for its own question,
# find its own task directory and read its own answer without analysing
# anything. That resolved the agent's own gold answer for 25 of 25 sampled
# tasks. It was harmless while the agent ran in a remote Harbor sandbox (Harbor
# uploads only a task's `environment/` dir, never the task root); running the
# agent in this container removes that boundary, so the answers do not ship.
#
# `../rle`'s container holds the grading key and gives the agent no shell.
#
# Consequence: ROLLOUT_BACKEND=harbor does not work in this image, by design.
# To fall back to the Harbor path, deploy the last image built from the previous
# Dockerfile (tag 0.2.0) rather than re-baking the suite here.
COPY vendor/task-index.json.gz ./vendor/task-index.json.gz

ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000
ENV ROLLOUT_BACKEND=direct
# Backstop only, and now a thin one: nothing in the direct path resolves a
# dataset spec at all. It stays because gradio pulls `huggingface_hub` in
# transitively, and no rollout should be able to turn into a Hub round trip.
ENV HF_HUB_OFFLINE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
