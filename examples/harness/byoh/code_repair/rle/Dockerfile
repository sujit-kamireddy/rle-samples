# Package index for this build. Defaults to public PyPI; override it where that
# is unreachable. A TLS handshake failure reaching files.pythonhosted.org
# means the network blocks public PyPI: build with
# --build-arg PIP_INDEX_URL=<your mirror>, or let the fallback below take over.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. `azd ai rle
# publish` takes no --build-arg, so a build that needs a different index has no
# way to be told about one -- without this fallback, publishing is unusable
# wherever public PyPI is blocked. Set it empty to disable the fallback and
# fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.10-slim

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

# This RLE wraps a real SWE-bench-Lite instance (see fixtures/instance.json):
# psf/requests at its pinned base commit. requests 2.10 predates Python's
# collections.abc split and does `collections.Mapping` directly, which was
# removed in Python 3.10. Rather than pin this image to an end-of-life
# interpreter for one 2016 checkout, sitecustomize.py restores the removed
# aliases; it is auto-imported by every Python process here, including the
# pytest subprocess spawned by /grade.

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
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" pytest==6.2.5 fastapi uvicorn

# A Harness RLE serves /health, /reset, /tools/* and /grade, so this image needs
# an HTTP framework and nothing more. It deliberately does not install openenv:
# that package carries the Gym: OpenEnv step protocol, which RLE never calls on
# a Harness container, and it pulls in a modern `requests` that then competes
# with the 2016 checkout under test.

# The checkout under /opt/base-repo is deliberately NOT pip-installed. Nothing
# needs it on sys.path: every rollout grades a *copy* of this checkout, and
# /grade points PYTHONPATH at that copy so the code under test wins the import.
# requests 2.10 vendors its own dependencies under requests/packages/, so the
# copy is self-contained.

WORKDIR /app
COPY fixtures/ /app/fixtures/
COPY server/ /app/server/

RUN python -c "import server.env as m; print('RLE harness server OK:', m.app)"

EXPOSE 8000

CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "8000"]
