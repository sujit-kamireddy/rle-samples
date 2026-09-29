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

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --progress-bar off --index-url "${PIP_INDEX_URL}" \
        "fastapi==0.115.6" "uvicorn==0.34.0" "pydantic==2.10.4"

# A Harness RLE serves /health, /reset, /tools/* and /grade, so this image needs
# an HTTP framework and nothing more. It deliberately does not install openenv:
# that package carries the Gym/OpenEnv step protocol, which RLE never calls on a
# Harness container. Nor does it install loom, whose tool decorator the
# simulated tools use -- loom pulls torch, and a four-route container has no use
# for a training stack. `vendor/` carries the few hundred lines actually needed;
# `tests/test_rle_vendor_parity.py` fails if they drift from upstream.

WORKDIR /app

# The world, the tasks, the simulated tools and the rubric are shared verbatim
# with the Loom RL runs (`rl/`). Copying rather than re-implementing is what
# makes a reward measured in this container comparable with one measured there.
#
# The dataset is deliberately absent. Every task arrives whole in the `/reset`
# body, so baking `data/rl/` in would add nothing the container can use -- and
# those rows carry the expected decision and the verified snapshot, which is
# answer key. It does not belong in an image the agent's sandbox can reach.
COPY rl/ /app/rl/
COPY vendor/ /app/vendor/
COPY server/ /app/rle/server/

# `rl/simulated_tools.py` imports `loom_cookbook.tool_use`, which resolves to
# the vendored copy; `rle.server.env` imports `rl.*` from the workdir.
ENV PYTHONPATH=/app/vendor:/app
ENV PYTHONUNBUFFERED=1

# Import the app and the server that will run it. RLE polls /health and gives
# up with RolloutDependencyFailed if the container never answers, which reports
# a missing dependency as a rollout failure -- so it is worth failing here,
# where the error names the module, instead of there.
RUN python -c "import uvicorn, rle.server.env as m; print('RLE harness server OK:', m.app)"

EXPOSE 8000

# `python -m uvicorn`, not the `uvicorn` console script: the module is what was
# installed and verified above, while the script is a PATH lookup that can come
# out differently in the sandbox than it does in a build step.
CMD ["python", "-m", "uvicorn", "rle.server.env:app", "--host", "0.0.0.0", "--port", "8000"]
