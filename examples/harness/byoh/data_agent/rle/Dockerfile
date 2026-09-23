# Package index for this build. Defaults to public PyPI. Where that is blocked,
# override it with --build-arg PIP_INDEX_URL=<your mirror>, or rely on the
# fallback below, which is probed only when the primary index is unreachable.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. `azd ai rle
# publish` takes no --build-arg, so a build that needs a different index has no
# way to be told about one -- without this fallback, publishing is unusable
# wherever public PyPI is blocked. Set it empty to disable the fallback and
# fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.12-slim

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

WORKDIR /app

# This container has no agent and no checkout to bake in -- see
# `server/env.py`'s module docstring for why: `../agent` has already run the
# agent loop by the time `/grade` is called here.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" fastapi uvicorn

COPY server/ /app/server/

RUN python -c "import server.env as m; print('RLE harness server OK:', m.app)"

EXPOSE 8000

CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "8000"]
