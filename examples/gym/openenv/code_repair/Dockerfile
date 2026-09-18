FROM python:3.10-slim

# This environment wraps a real SWE-bench-Lite instance (see
# fixtures/instance.json): psf/requests at its pinned base commit. requests
# 2.10 predates Python's collections.abc split and does `collections.Mapping`
# directly, which was removed in Python 3.10. openenv requires Python >=3.10,
# so instead of an older interpreter this image restores the removed aliases
# via sitecustomize.py (auto-imported by every Python process, including the
# pytest subprocess step() spawns) rather than pinning Python <3.10. Same
# fix as examples/harness/byoh/rle's Dockerfile, for the same reason.

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY server/sitecustomize.py /usr/local/lib/python3.10/site-packages/sitecustomize.py

WORKDIR /opt

# Real, pinned upstream checkout -- not a synthetic fixture. Every rollout's
# reset() copies this read-only reference into a fresh working directory.
RUN git clone --quiet https://github.com/psf/requests.git base-repo \
    && cd base-repo \
    && git checkout --quiet 36453b95b13079296776d11b09cab2567ea3e703

RUN pip install --no-cache-dir -e /opt/base-repo \
    && pip install --no-cache-dir pytest==6.2.5 openenv

WORKDIR /app
COPY . /app/envs/code_repair

ENV PYTHONPATH=/app

RUN python -c "import envs.code_repair.server.app as m; print('code_repair server OK:', m.app)"

EXPOSE 8000

CMD ["uvicorn", "envs.code_repair.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# To build and run locally:
#   cd examples/gym/openenv/code_repair && docker build -t ft-code-repair-env:latest .
#   docker run --rm -p 8000:8000 ft-code-repair-env:latest
