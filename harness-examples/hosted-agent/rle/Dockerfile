FROM python:3.8-slim

# This RLE wraps a real SWE-bench-Lite instance (see fixtures/instance.json):
# psf/requests at its pinned base commit. python:3.8 matches this era of the
# repo; requests 2.10 does not import on Python >= 3.10 (collections.Mapping
# was removed there).

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Real, pinned upstream checkout -- not a synthetic fixture. Every rollout's
# /reset copies this read-only reference into a fresh working directory.
RUN git clone --quiet https://github.com/psf/requests.git base-repo \
    && cd base-repo \
    && git checkout --quiet 36453b95b13079296776d11b09cab2567ea3e703

RUN pip install --no-cache-dir -e /opt/base-repo \
    && pip install --no-cache-dir pytest==6.2.5 openenv

WORKDIR /app
COPY fixtures/ /app/fixtures/
COPY server/ /app/server/

RUN python -c "import server.env as m; print('RLE harness server OK:', m.app)"

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "8000"]
