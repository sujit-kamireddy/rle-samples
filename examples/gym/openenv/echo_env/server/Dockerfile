# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# `echo_env` has no task dataset. This image installs only the OpenEnv MCP
# runtime and its server dependencies; no data is downloaded or baked.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=${PIP_INDEX_URL}
ARG PIP_FALLBACK_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/

FROM python:3.11-slim

ARG PIP_INDEX_URL
ARG UV_DEFAULT_INDEX
ARG PIP_FALLBACK_INDEX_URL

WORKDIR /app/env

# Use the fallback only when the primary package index cannot be reached.
RUN if [ -n "${PIP_FALLBACK_INDEX_URL}" ] && \
       ! pip download --no-deps --quiet --dest /tmp/idxprobe six >/dev/null 2>&1; then \
        echo "Primary index unreachable; falling back to ${PIP_FALLBACK_INDEX_URL}"; \
        export PIP_INDEX_URL="${PIP_FALLBACK_INDEX_URL}"; \
        export UV_DEFAULT_INDEX="${PIP_FALLBACK_INDEX_URL}"; \
    fi; \
    rm -rf /tmp/idxprobe; \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache "openenv==0.3.1"

COPY . /app/env

ENV PYTHONPATH=/app/env

# Fail during the build if the application cannot be imported.
RUN python -c "import server.app as m; print('echo_env OpenEnv server OK:', m.app)"

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
