FROM python:3.12-slim

# Package index for this build. Defaults to public PyPI. Where that is blocked,
# override it with --build-arg PIP_INDEX_URL=<your mirror>, or rely on the
# fallback below, which is probed only when the primary index is unreachable.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. Set it empty to
# disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

# git: `opencode` shells out to it. build-essential: kept for any data-stack
# package that has no wheel for this platform and builds from source.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Repeat the defaults, don't just re-declare the names. BuildKit carries a
# stage's ARG default across a re-declaration; the classic builder (which is
# what `az acr build` runs) resets it to empty, which silently turns
# --index-url into "" and fails the install below.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

# The agent's own toolbox, plus this service's own dependencies. Every task
# instruction promises these are installed ("Installed: pandas, numpy, ...
# pip install more if needed"), and the agent runs in this container, so they
# have to be here.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        pandas numpy matplotlib seaborn scipy scikit-learn statsmodels tabulate plotly

# `opencode` is the agent. It is installed at build time, so a rollout starts
# the agent instead of first installing it.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm i -g opencode-ai@latest \
    && opencode --version

# The fetcher that pulls a task's input files from the CDN. It runs as a
# subprocess of this service, once per rollout. `vendor/pull_bucket.py` is a
# generated copy of `tools/pull_bucket.py` (Docker cannot COPY above the build
# context); `tools/build_task_index.py --write` refreshes it.
COPY vendor/pull_bucket.py /opt/pull_bucket.py

WORKDIR /app

COPY requirements.txt .
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY app.py opencode_direct.py ./

# The task suite is deliberately NOT baked into this image.
#
# Running `opencode` here needs each task's instruction, its bucket coordinates
# and its timeout, which is what `vendor/task-index.json.gz` holds (0.4MB, vs
# 216MB extracted).
#
# This is a security boundary, not a diet. Every `task.toml` in the suite
# carries `metadata.gold_answer` and `verifier.env.EXPECTED_ANSWER`, and every
# instruction quotes its question verbatim while `task.description` repeats it.
# An agent with a shell -- which is exactly what `opencode
# --dangerously-skip-permissions` is -- can grep the suite for its own question,
# find its own task directory and read its own answer without analysing
# anything. That resolved the agent's own gold answer for 25 of 25 sampled
# tasks. So the answers do not ship here.
#
# Anything added to this image is readable by the agent. `../rle`'s container
# holds the grading key and gives the agent no shell.
COPY vendor/task-index.json.gz ./vendor/task-index.json.gz

ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
