FROM python:3.12-slim

# git: `harbor` sandboxes and task fetch shell out to it. build-essential: a couple of the
# per-sandbox extras pull packages that need to compile.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml .
COPY server ./server

# No lockfile checked in yet (phase 1); resolve fresh. Pin a uv.lock here once the dependency
# set settles, the way OpenEnv/envs/harbor_env does, so rebuilds stop re-resolving.
RUN uv pip install --system --no-cache .

ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
