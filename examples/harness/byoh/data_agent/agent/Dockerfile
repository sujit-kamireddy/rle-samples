# Package index for this build. Defaults to public PyPI. Where that is blocked,
# override it with --build-arg PIP_INDEX_URL=<your mirror>, or rely on the
# fallback below, which is probed only when the primary index is unreachable.
ARG PIP_INDEX_URL=https://pypi.org/simple

# Index to fall back to when the one above cannot be reached. Set it empty to
# disable the fallback and fail on the primary index.
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.12-slim

ARG PIP_INDEX_URL
ARG PIP_FALLBACK_INDEX_URL

WORKDIR /app

COPY requirements.txt .
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --index-url "${PIP_INDEX_URL}" --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY app.py .

EXPOSE 8080

# Base URL of the separately deployed harness (../harness). This
# has no default because a shim with no harness server to call cannot do
# anything useful -- fail at startup, not on the first rollout.
ENV HARNESS_SERVER_URL=""

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
