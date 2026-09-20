# Package index for this build. Defaults to public PyPI; override it where that
# is unreachable -- see "Build fails downloading Python packages" in
# ../../README.md.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. `azd ai rle run`
# and `azd ai rle publish` take no --build-arg, so a build that needs a
# different index has no way to be told about one -- without this fallback the
# local-iteration path the READMEs document is unusable wherever public PyPI is
# blocked. Set it empty to disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.10-slim

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

# This environment wraps 40 real SWE-bench-Verified instances, all
# django/django version 3.2 (see env_data/NOTICE.md), so one clone + one
# dependency set covers the whole baked-in pool. Django 3.2 supports up to
# Python 3.10 -- the newest interpreter it does, and openenv's own floor
# (requires-python >=3.10) -- avoiding both the EOL/packaging headaches of
# the version the original SWE-bench harness spec calls for (3.6) and any
# collections.abc compatibility shim.

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Real, pinned upstream clone -- not a synthetic fixture. A *full* clone
# (no --filter) so every base_commit's blobs are local: each episode's
# reset() checks out its instance's commit into its own working directory
# via `git worktree add`, which needs zero network access at runtime.
RUN git clone --quiet https://github.com/django/django.git base-repo

WORKDIR /app

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        "asgiref>=3.3.2" "sqlparse>=0.2.2" "tblib>=1.5.0" pytz tzdata \
        "backports.zoneinfo; python_version<'3.9'" openenv

COPY . /app/envs/code_repair

ENV PYTHONPATH=/app

RUN python -c "import envs.code_repair.server.app as m; print('code_repair server OK:', m.app)"

EXPOSE 8000

CMD ["uvicorn", "envs.code_repair.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# To build and run locally:
#   cd examples/gym/openenv/code_repair && docker build -t ft-code-repair-env:latest .
#   docker run --rm -p 8000:8000 ft-code-repair-env:latest
