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

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" uv

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

# Bake the task suite into the image rather than downloading it on a rollout's
# first request. Vendored as plain files in this repo (see
# vendor/harbor-datasets/) rather than fetched here, so `docker build` never
# needs HF Hub network access either -- this is a straight local COPY. It is
# small (~196MB of task defs, instructions, and graders -- the per-task data
# itself is pulled from a bucket at *sandbox* build time, not stored here).
# Override OPENENV_DATASETS below (and vendor the matching directory) to bake
# a different split.
ARG DATASETS=FineEnvs/data-agent-harbor-train
ENV OPENENV_DATASET_CACHE=/opt/harbor-datasets
COPY vendor/harbor-datasets/ ${OPENENV_DATASET_CACHE}/

ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000
ENV OPENENV_DATASETS=${DATASETS}
# The dataset is already on disk from the bake step above; this stops
# `huggingface_hub` from making a network call to re-verify it on first use,
# which would otherwise be a hard dependency on reaching the HF Hub from
# wherever this image ends up running.
ENV HF_HUB_OFFLINE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
