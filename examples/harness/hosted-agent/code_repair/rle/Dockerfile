# Package index for this build. Defaults to public PyPI; override it where that
# is unreachable -- see "Build fails downloading Python packages" in
# ../../../gym/openenv/README.md.
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

# This RLE wraps a real SWE-bench-Lite instance (see fixtures/instance.json):
# psf/requests at its pinned base commit. requests 2.10 predates Python's
# collections.abc split and does `collections.Mapping` directly, which was
# removed in Python 3.10. openenv requires Python >=3.10, so instead of an
# older interpreter this image restores the removed aliases via
# sitecustomize.py (auto-imported by every Python process, including the
# pytest subprocess spawned by /grade) rather than pinning Python <3.10.

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY server/sitecustomize.py /usr/local/lib/python3.10/site-packages/sitecustomize.py

WORKDIR /opt

# Real, pinned upstream checkout -- not a synthetic fixture. Every rollout's
# /reset copies this read-only reference into a fresh working directory.
RUN git clone --quiet https://github.com/psf/requests.git base-repo \
    && cd base-repo \
    && git checkout --quiet 36453b95b13079296776d11b09cab2567ea3e703

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" pytest==6.2.5 openenv

# The checkout under /opt/base-repo is deliberately NOT pip-installed. It is
# requests 2.10, and openenv requires requests>=2.25, so installing it would
# only be undone moments later when pip resolves openenv and replaces it with a
# modern wheel -- leaving an install that looks pinned but is not. Nothing needs
# it on sys.path anyway: every rollout grades a *copy* of this checkout, and
# /grade points PYTHONPATH at that copy so the code under test wins the import.
# requests 2.10 vendors its own dependencies under requests/packages/, so the
# copy is self-contained.

WORKDIR /app
COPY fixtures/ /app/fixtures/
COPY server/ /app/server/

RUN python -c "import server.env as m; print('RLE harness server OK:', m.app)"

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "8000"]
