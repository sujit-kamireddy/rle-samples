FROM python:3.12-slim

# `git` is here only because `requirements.txt` still pins the SDK from a git
# ref; it can go once the `rle` extra ships on PyPI. See that file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY rle_openenv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rle_openenv/__init__.py ./rle_openenv/__init__.py
# Everything this image runs lives under rle_openenv/, including its own copies
# of the grading modules. The deprecated harness is excluded from the build
# context outright, so a COPY that reached back into it would fail the build.
COPY rle_openenv/server/ ./rle_openenv/server/

EXPOSE 8000
CMD ["uvicorn", "rle_openenv.server.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
