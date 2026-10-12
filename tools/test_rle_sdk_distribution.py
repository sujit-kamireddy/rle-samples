"""Dependency-light checks for self-contained public SDK installation."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    ROOT / "examples" / "gym" / "openenv" / "math_rl",
    ROOT / "examples" / "gym" / "openenv" / "code_rl",
    ROOT / "examples" / "harness" / "byoh" / "data_code_agent" / "rle",
    ROOT / "examples" / "harness" / "hosted-agent" / "competitive_intelligence_agent" / "rle",
)
PUBLIC_GIT_SOURCE = (
    "https://github.com/Azure/azure-sdk-for-python.git#"
    "rle-external-releases:sdk/ai/azure-ai-projects/rle-external"
)


def test_sample_installers_are_byte_identical():
    expected = (SAMPLES[0] / "install_rle_sdk.py").read_bytes()
    for sample in SAMPLES[1:]:
        assert (sample / "install_rle_sdk.py").read_bytes() == expected


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample.parent.name + "-" + sample.name for sample in SAMPLES])
def test_sample_installs_public_sdk_and_refreshes_manifest_cache(sample):
    dockerfile = (sample / "Dockerfile").read_text()
    requirements = (sample / "requirements.txt").read_text()
    assert not (sample / "sdk-version.txt").exists()
    assert "ENV AZURE_AI_PROJECTS_RLE_VERSION=latest\n" in dockerfile
    assert "ARG AZURE_AI_PROJECTS_RLE_VERSION" not in dockerfile
    assert "ARG PIP_INDEX_URL" in dockerfile
    assert "COPY install_rle_sdk.py requirements.txt /app/" in dockerfile
    assert f"ADD {PUBLIC_GIT_SOURCE} /app/rle-sdk-dist/" in dockerfile
    assert "--manifest /app/rle-sdk-dist/registry.json" in dockerfile
    assert "--requirements /app/requirements.txt" in dockerfile
    assert "--version-file" not in dockerfile
    assert dockerfile.index("ENV AZURE_AI_PROJECTS_RLE_VERSION=latest") < dockerfile.index("python /app/install_rle_sdk.py")
    assert dockerfile.index(f"ADD {PUBLIC_GIT_SOURCE}") < dockerfile.index("python /app/install_rle_sdk.py")
    assert "git+" not in dockerfile + requirements
    assert "sujit-kamireddy/azure-sdk-for-python" not in dockerfile + requirements
    assert "apt-get install" not in dockerfile
    assert "openenv==0.6.0" in requirements
    ignore = sample / "Dockerfile.dockerignore"
    if ignore.exists() and "**" in ignore.read_text().splitlines():
        rules = ignore.read_text().splitlines()
        assert "!install_rle_sdk.py" in rules
