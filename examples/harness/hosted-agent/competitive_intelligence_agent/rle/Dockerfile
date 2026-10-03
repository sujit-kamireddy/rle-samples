# Package index for this build. Defaults to public PyPI; override where that is
# unreachable -- Microsoft CFS enforcement makes the TLS handshake to
# files.pythonhosted.org fail from inside a build.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. `azd ai rle
# publish` takes no --build-arg, so a build needing a different index has no way
# to be told about one; without a fallback, publishing is unusable wherever
# public PyPI is blocked. Set it empty to fail on the primary index instead.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.12-slim

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

# `git` is here only because `requirements.txt` still pins the SDK from a git
# ref; it can go once the `rle` extra ships on PyPI. See that file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unlike the legacy harness image, this one does install openenv: the OpenEnv
# step protocol is exactly what it serves. `azure-ai-projects[rle]` brings the
# `RLEnvironment` base class and fastmcp with it. loom is still not installed --
# it pulls torch, and `vendor/` carries the few hundred lines of its tool
# decorator that the simulated tools actually need.
COPY rle/requirements.txt /app/requirements.txt
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --progress-bar off --index-url "${PIP_INDEX_URL}" \
        -r /app/requirements.txt

# This image is self-contained: the world, the tasks, the simulated tools and
# the rubric are copied from `rle/` itself, not from the legacy
# harness. `rle/` carries an identical copy for now, and
# `tests/test_environment.py` asserts the two still grade the same, but nothing
# in this image reads it -- which is what lets `rle/` be deleted outright.
#
# The agent, the dataset and the legacy Dockerfile are excluded by
# `Dockerfile.dockerignore`: those rows carry the expected decision and the
# verified snapshot, which is answer key, and do not belong in an image the
# agent's sandbox can reach.
COPY rle/__init__.py /app/rle/__init__.py
COPY rle/rl/ /app/rle/rl/
COPY rle/vendor/ /app/rle/vendor/
COPY rle/server/ /app/rle/server/

# `rle.rl.simulated_tools` imports `loom_cookbook.tool_use`, which
# resolves to the vendored copy; everything else is a `rle.*` import
# rooted at the workdir.
ENV PYTHONPATH=/app/rle/vendor:/app
ENV PYTHONUNBUFFERED=1

# Import the app and the server that will run it. A missing dependency reported
# here names the module; reported at rollout time it arrives as an opaque
# RolloutDependencyFailed.
RUN python -c "import uvicorn, rle.server.app as m; print('OpenEnv RLE server OK:', m.app)"

EXPOSE 8000

# One worker: `create_app` keeps its sessions in process memory, so a second
# worker would not recognise the session ids the first one minted.
#
# `python -m uvicorn`, not the `uvicorn` console script: the module is what was
# installed and verified above, while the script is a PATH lookup that can come
# out differently in the sandbox than it does in a build step.
CMD ["python", "-m", "uvicorn", "rle.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
