FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app
ENV AZURE_AI_PROJECTS_RLE_VERSION=latest
COPY requirements.txt /app/
ADD --checksum=sha256:5aa3de2890dce17242594ba2cb7f7d1ec9403a5455ed95c21b303a6775bb0a9a https://raw.githubusercontent.com/sujit-kamireddy/rle-samples/8c1424cd7383fc5bae4493a2f16c5c08131c0a83/tools/install_rle_sdk.py /app/install_rle_sdk.py
# Resolve the public branch commit; raw HTTP manifests can be CDN-cached.
ADD https://github.com/Azure/azure-sdk-for-python.git#rle-external-releases:sdk/ai/azure-ai-projects/rle-external /app/rle-sdk-dist/
RUN python /app/install_rle_sdk.py --manifest /app/rle-sdk-dist/registry.json \
    --requirements /app/requirements.txt
COPY __init__.py ./rle/__init__.py
# The build context is this folder (`rle/`) itself -- `azd ai rle publish`
# always builds from the directory holding rle.toml, so COPY sources are
# relative to here, not the sample root.
COPY server/ ./rle/server/

# Fail the build if the SDK lacks the query-routing app factory.
RUN python -c "import uvicorn, rle.server.app as m; m.build_app()"

EXPOSE 8000
CMD ["uvicorn", "rle.server.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
