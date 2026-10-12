"""Dependency-light checks for shared public SDK installation."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

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
    "rle-external-releases:sdk/ai/azure-ai-projects/rle-releases"
)
SHARED_INSTALLER = ROOT / "tools" / "install_rle_sdk.py"
SHARED_INSTALLER_URL = (
    "https://raw.githubusercontent.com/sujit-kamireddy/rle-samples/"
    "c2b4db2904d524d3f1793b0fb3eed31954f3729f/tools/install_rle_sdk.py"
)
VERSION = "2.7.0b1"
SECOND_VERSION = "2.7.0b2"
LEGACY_VERSION = "2.7.0.dev20261012000000"


@pytest.fixture(scope="module")
def installer():
    spec = importlib.util.spec_from_file_location(
        "sample_rle_sdk_installer", SHARED_INSTALLER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wheel_bytes(version=VERSION, *, extra="rle", module=True):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            f"azure_ai_projects-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: azure-ai-projects\nVersion: {version}\nProvides-Extra: {extra}\n",
        )
        if module:
            archive.writestr("azure/ai/projects/rle/environments/__init__.py", "")
    return output.getvalue()


def manifest_for(installer, version=VERSION):
    filename = f"azure_ai_projects-{version}-py3-none-any.whl"
    return {
        "schema_version": 1, "package": "azure-ai-projects", "channel": "rle-external",
        "latest": version,
        "versions": {version: {
            "filename": filename,
            "url": (
                f"https://raw.githubusercontent.com/{installer.PUBLIC_REPOSITORY}/"
                f"{'a' * 40}/{installer.PUBLIC_PATH}/wheels/{version}/{filename}"
            ),
            "sha256": hashlib.sha256(wheel_bytes(version)).hexdigest(),
        }},
    }


def test_samples_do_not_duplicate_the_shared_installer():
    assert SHARED_INSTALLER.is_file()
    for sample in SAMPLES:
        assert not (sample / "install_rle_sdk.py").exists()


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample.parent.name + "-" + sample.name for sample in SAMPLES])
def test_sample_installs_public_sdk_and_refreshes_manifest_cache(sample):
    dockerfile = (sample / "Dockerfile").read_text()
    requirements = (sample / "requirements.txt").read_text()
    assert not (sample / "sdk-version.txt").exists()
    assert "ENV AZURE_AI_PROJECTS_RLE_VERSION=latest\n" in dockerfile
    assert "ARG AZURE_AI_PROJECTS_RLE_VERSION" not in dockerfile
    assert "ARG PIP_INDEX_URL" in dockerfile
    digest = hashlib.sha256(SHARED_INSTALLER.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    shared_add = f"ADD --checksum=sha256:{digest} {SHARED_INSTALLER_URL} /app/install_rle_sdk.py"
    assert shared_add in dockerfile
    assert "COPY requirements.txt /app/" in dockerfile
    assert "COPY install_rle_sdk.py" not in dockerfile
    assert f"ADD {PUBLIC_GIT_SOURCE} /app/rle-sdk-dist/" in dockerfile
    assert "--manifest /app/rle-sdk-dist/registry.json" in dockerfile
    assert "--requirements /app/requirements.txt" in dockerfile
    assert "--version-file" not in dockerfile
    assert dockerfile.index("ENV AZURE_AI_PROJECTS_RLE_VERSION=latest") < dockerfile.index("python /app/install_rle_sdk.py")
    assert dockerfile.index(shared_add) < dockerfile.index("python /app/install_rle_sdk.py")
    assert dockerfile.index(f"ADD {PUBLIC_GIT_SOURCE}") < dockerfile.index("python /app/install_rle_sdk.py")
    assert "git+" not in dockerfile + requirements
    assert "sujit-kamireddy/azure-sdk-for-python" not in dockerfile + requirements
    assert "apt-get install" not in dockerfile
    assert "openenv==0.6.0" in requirements
    ignore = sample / "Dockerfile.dockerignore"
    if ignore.exists() and "**" in ignore.read_text().splitlines():
        rules = ignore.read_text().splitlines()
        assert "!requirements.txt" in rules
        assert "!install_rle_sdk.py" not in rules


def test_beta_latest_and_legacy_pins_resolve_independently(installer):
    manifest = manifest_for(installer, LEGACY_VERSION)
    for version in (VERSION, SECOND_VERSION):
        manifest["versions"].update(manifest_for(installer, version)["versions"])
    manifest["latest"] = SECOND_VERSION
    assert installer.resolve_release(manifest).version == SECOND_VERSION
    for version in (VERSION, SECOND_VERSION, LEGACY_VERSION):
        assert installer.resolve_release(manifest, version).version == version


def test_versioned_release_branch_wheel_is_verified_before_installation(installer, tmp_path):
    manifest = manifest_for(installer)
    entry = manifest["versions"][VERSION]
    entry["url"] = entry["url"].replace("a" * 40, installer.PUBLIC_BRANCH)
    release = installer.resolve_release(manifest)
    with patch.object(installer, "urlopen", return_value=io.BytesIO(wheel_bytes())):
        assert installer.download_wheel(release, tmp_path).read_bytes() == wheel_bytes()
    with patch.object(installer, "urlopen", return_value=io.BytesIO(b"tampered")):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            installer.download_wheel(release, tmp_path)


@pytest.mark.parametrize("reference", ["main", "other-branch", "a" * 39, "latest"])
def test_unapproved_wheel_references_are_rejected(installer, reference):
    manifest = manifest_for(installer)
    entry = manifest["versions"][VERSION]
    entry["url"] = entry["url"].replace("a" * 40, reference)
    with pytest.raises(ValueError, match="public SDK commit or release branch"):
        installer.resolve_release(manifest)


@pytest.mark.parametrize("version", [
    "", "2.7.0", "missing", "../escape", SECOND_VERSION, "2.7.0.preview1", "2.7.0b0", "2.7.0b01",
])
def test_explicit_invalid_or_missing_version_never_falls_back(installer, version):
    with pytest.raises(ValueError):
        installer.resolve_release(manifest_for(installer), version)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("package", "other"), ("channel", "other"),
    ("latest", ""), ("versions", []),
])
def test_invalid_manifest_fails(installer, field, value):
    manifest = manifest_for(installer)
    manifest[field] = value
    with pytest.raises(ValueError):
        installer.resolve_release(manifest)


@pytest.mark.parametrize("field,value", [
    ("filename", "../../escape.whl"),
    ("url", "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/main/sdk.whl"),
    ("url", "https://example.com/sdk.whl"),
    ("url", (
        "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/"
        f"rle-external-releases/sdk/ai/azure-ai-projects/rle-external/wheels/{VERSION}/"
        f"azure_ai_projects-{VERSION}-py3-none-any.whl"
    )),
    ("sha256", "not-a-hash"),
])
def test_invalid_artifact_entry_fails(installer, field, value):
    manifest = manifest_for(installer)
    manifest["versions"][VERSION][field] = value
    with pytest.raises(ValueError):
        installer.resolve_release(manifest)


@pytest.mark.parametrize("version", [VERSION, LEGACY_VERSION])
def test_download_checks_hash_and_wheel_metadata(installer, tmp_path, version):
    data = wheel_bytes(version)
    with patch.object(installer, "urlopen", return_value=io.BytesIO(data)):
        wheel = installer.download_wheel(
            installer.resolve_release(manifest_for(installer, version)), tmp_path,
        )
    assert wheel.read_bytes() == data


def test_checksum_failure_removes_download_and_never_installs(installer, tmp_path):
    with patch.object(installer, "urlopen", return_value=io.BytesIO(b"tampered")):
        with patch.object(installer.subprocess, "run") as run:
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                installer.download_wheel(installer.resolve_release(manifest_for(installer)), tmp_path)
    assert not list(tmp_path.iterdir())
    run.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"version": SECOND_VERSION}, {"extra": "voice"}, {"module": False},
])
def test_wrong_wheel_contents_fail(installer, tmp_path, kwargs):
    wheel = tmp_path / "sdk.whl"
    wheel.write_bytes(wheel_bytes(**kwargs))
    with pytest.raises(ValueError):
        installer.validate_wheel(wheel, VERSION)


def test_selector_precedence_and_empty_values_fail(installer, monkeypatch):
    monkeypatch.delenv("AZURE_AI_PROJECTS_RLE_VERSION", raising=False)
    assert installer.selected_version(None) == "latest"
    monkeypatch.setenv("AZURE_AI_PROJECTS_RLE_VERSION", SECOND_VERSION)
    assert installer.selected_version(None) == SECOND_VERSION
    assert installer.selected_version(VERSION) == VERSION
    monkeypatch.setenv("AZURE_AI_PROJECTS_RLE_VERSION", "")
    with pytest.raises(ValueError):
        installer.resolve_release(manifest_for(installer), installer.selected_version(None))


@pytest.mark.parametrize("version", [VERSION, LEGACY_VERSION])
def test_pip_receives_verified_wheel_extra_and_requirements(installer, tmp_path, monkeypatch, version):
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(manifest_for(installer, version)))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("openenv==0.6.0\n")
    monkeypatch.setattr(sys, "argv", [
        "install_rle_sdk.py", "--manifest", str(manifest), "--version", version,
        "--requirements", str(requirements), "--index-url", "https://pypi.org/simple",
    ])
    with patch.object(installer, "urlopen", return_value=io.BytesIO(wheel_bytes(version))):
        with patch.object(installer.subprocess, "run") as run:
            installer.main()
    command = run.call_args.args[0]
    assert any(argument.startswith("azure-ai-projects[rle] @ file:") for argument in command)
    assert command[-4:] == ["--index-url", "https://pypi.org/simple", "--requirement", str(requirements)]
    assert run.call_args.kwargs["check"] is True
