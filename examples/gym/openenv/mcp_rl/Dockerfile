FROM python:3.12-slim

WORKDIR /app

# This exact SDK source revision reports version 2.7.0 and contains
# azure.ai.projects.rle.environments.RLEnvironment. The released public 2.7.0
# wheel does not contain that package, so a normal PyPI version pin is not an
# equivalent dependency. Keep this immutable until the rle extra is released.
ARG AZURE_AI_PROJECTS_RLE_REF=05ba0ba72d3de0cfe0c742653da9eb87e9e02730
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
      "azure-ai-projects[rle] @ git+https://github.com/sujit-kamireddy/azure-sdk-for-python@${AZURE_AI_PROJECTS_RLE_REF}#subdirectory=sdk/ai/azure-ai-projects" \
      "openenv==0.6.0"

COPY __init__.py /app/mcp_rl/__init__.py
COPY server/ /app/mcp_rl/server/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -c "from azure.ai.projects.rle.environments import RLEnvironment; from mcp_rl.server.app import app; print('mcp_rl server OK:', app)"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "mcp_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
