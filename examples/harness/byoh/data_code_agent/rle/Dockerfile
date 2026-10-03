FROM python:3.12-slim

# `git` is here only because `requirements.txt` still pins the SDK from a git
# ref; it can go once the `rle` extra ships on PyPI. See that file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY __init__.py ./rle/__init__.py
# The build context is this folder (`rle/`) itself -- `azd ai rle publish`
# always builds from the directory holding rle.toml, so COPY sources are
# relative to here, not the sample root. The deprecated harness is a sibling
# folder outside this context already, so a COPY that reached for it would
# fail the build rather than quietly re-coupling the image to a folder that
# is going away.
COPY server/ ./rle/server/

EXPOSE 8000
CMD ["uvicorn", "rle.server.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
