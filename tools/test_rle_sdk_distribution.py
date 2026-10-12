"""Dependency-light checks for sample-local public SDK installation."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.request import url2pathname

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    ROOT / "examples" / "gym" / "openenv" / "math_rl",
    ROOT / "examples" / "gym" / "openenv" / "code_rl",
    ROOT / "examples" / "harness" / "byoh" / "data_code_agent" / "rle",
    ROOT / "examples" / "harness" / "hosted-agent" / "competitive_intelligence_agent" / "rle",
)
PUBLIC_MANIFEST_URL = (
    "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/"
    "rle-releases/sdk/ai/azure-ai-projects/rle/registry.json"
)
REFERENCE_INSTALLER = SAMPLES[0] / "install_rle_sdk.py"
VERSION = "2.7.0b1"
SECOND_VERSION = "2.7.0b2"


@pytest.fixture(scope="module")
def installer():
    spec = importlib.util.spec_from_file_location(
        "sample_rle_sdk_installer", REFERENCE_INSTALLER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wheel_bytes(version=VERSION):
    return f"SDK wheel fixture for {version}".encode("utf-8")


def manifest_for(installer, version=VERSION):
    filename = f"azure_ai_projects-{version}-py3-none-any.whl"
    return {
        "schema_version": 1, "package": "azure-ai-projects", "channel": "rle",
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


def test_sample_installers_match_the_reference_copy():
    reference = REFERENCE_INSTALLER.read_text(encoding="utf-8")
    for sample in SAMPLES:
        assert (sample / "install_rle_sdk.py").read_text(encoding="utf-8") == reference, sample


@pytest.mark.parametrize("version", ["latest", "2.7.0b1"])
def test_final_upstream_registry_contract_resolves_latest_and_pin(installer, version):
    registry_url = (
        "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/"
        "rle-releases/sdk/ai/azure-ai-projects/rle/registry.json"
    )
    wheel_url = (
        "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/"
        "rle-releases/sdk/ai/azure-ai-projects/rle/wheels/2.7.0b1/"
        "azure_ai_projects-2.7.0b1-py3-none-any.whl"
    )
    manifest = {
        "schema_version": 1, "package": "azure-ai-projects", "channel": "rle",
        "latest": "2.7.0b1",
        "versions": {"2.7.0b1": {
            "filename": "azure_ai_projects-2.7.0b1-py3-none-any.whl",
            "url": wheel_url,
            "sha256": "004b027dc2629b834b1b957bc2b1b94f64c4fc0de901d2c90b8a1a21f5669979",
        }},
    }
    assert installer.MANIFEST_URL == registry_url
    release = installer.resolve_release(manifest, version)
    assert release.version == "2.7.0b1" and release.url == wheel_url


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample.parent.name + "-" + sample.name for sample in SAMPLES])
def test_sample_installs_public_sdk_and_refreshes_manifest_cache(sample):
    dockerfile = (sample / "Dockerfile").read_text()
    requirements = (sample / "requirements.txt").read_text()
    assert "ENV AZURE_AI_PROJECTS_VERSION=latest\n" in dockerfile
    assert "ARG AZURE_AI_PROJECTS_VERSION" not in dockerfile
    assert "ARG PIP_INDEX_URL" in dockerfile
    installer_copy = "COPY install_rle_sdk.py /app/"
    assert installer_copy in dockerfile
    assert "COPY requirements.txt " in dockerfile
    assert "raw.githubusercontent.com/sujit-kamireddy/rle-samples" not in dockerfile
    assert "ADD --checksum" not in dockerfile
    assert f"ADD {PUBLIC_MANIFEST_URL} /app/rle-sdk-dist/registry.json" in dockerfile
    assert "https://github.com/Azure/azure-sdk-for-python.git" not in dockerfile
    assert "--manifest /app/rle-sdk-dist/registry.json" in dockerfile
    assert "--requirements /app/requirements.txt" in dockerfile
    assert dockerfile.index("ENV AZURE_AI_PROJECTS_VERSION=latest") < dockerfile.index("python /app/install_rle_sdk.py")
    assert dockerfile.index(installer_copy) < dockerfile.index("python /app/install_rle_sdk.py")
    assert dockerfile.index(f"ADD {PUBLIC_MANIFEST_URL}") < dockerfile.index("python /app/install_rle_sdk.py")
    assert "git+" not in dockerfile + requirements
    assert "sujit-kamireddy/azure-sdk-for-python" not in dockerfile + requirements
    assert "apt-get install" not in dockerfile
    assert "openenv==0.6.0" in requirements
    ignore = sample / "Dockerfile.dockerignore"
    if ignore.exists() and "**" in ignore.read_text().splitlines():
        rules = ignore.read_text().splitlines()
        assert "!requirements.txt" in rules
        assert "!install_rle_sdk.py" in rules


def test_latest_and_pins_resolve_independently(installer):
    manifest = manifest_for(installer)
    manifest["versions"].update(manifest_for(installer, SECOND_VERSION)["versions"])
    manifest["latest"] = SECOND_VERSION
    assert installer.resolve_release(manifest).version == SECOND_VERSION
    for version in (VERSION, SECOND_VERSION):
        assert installer.resolve_release(manifest, version).version == version


def test_registry_wheel_url_does_not_require_a_specific_release_layout(installer):
    manifest = manifest_for(installer)
    entry = manifest["versions"][VERSION]
    entry["url"] = "https://example.com/sdk.whl"
    assert installer.resolve_release(manifest).url == entry["url"]


@pytest.mark.parametrize("version", [
    "", "missing", SECOND_VERSION,
])
def test_explicit_invalid_or_missing_version_never_falls_back(installer, version):
    with pytest.raises(ValueError):
        installer.resolve_release(manifest_for(installer), version)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("package", "other"), ("channel", "other"),
    ("channel", "rle-external"),
    ("latest", ""), ("versions", []),
])
def test_invalid_manifest_fails(installer, field, value):
    manifest = manifest_for(installer)
    manifest[field] = value
    with pytest.raises(ValueError):
        installer.resolve_release(manifest)


@pytest.mark.parametrize("field,value", [
    ("filename", "../../escape.whl"),
    ("filename", "..\\escape.whl"),
    ("filename", "C:escape.whl"),
    ("filename", "sdk.txt"),
    ("url", "http://example.com/sdk.whl"),
    ("sha256", None),
])
def test_invalid_artifact_entry_fails(installer, field, value):
    manifest = manifest_for(installer)
    manifest["versions"][VERSION][field] = value
    with pytest.raises(ValueError):
        installer.resolve_release(manifest)


@pytest.mark.parametrize("version", [".", "..", "../escape", "escape\\outside", "C:escape"])
def test_registry_versions_cannot_escape_the_wheel_directory(installer, version):
    manifest = manifest_for(installer)
    manifest["latest"] = version
    manifest["versions"][version] = manifest["versions"][VERSION]
    with pytest.raises(ValueError, match="safe wheel path components"):
        installer.resolve_release(manifest)


def test_download_checks_hash(installer, tmp_path):
    data = wheel_bytes()
    with patch.object(installer, "urlopen", return_value=io.BytesIO(data)):
        wheel = installer.download_wheel(
            installer.resolve_release(manifest_for(installer)), tmp_path,
        )
    assert wheel.read_bytes() == data


def test_checksum_failure_removes_download(installer, tmp_path):
    with patch.object(installer, "urlopen", return_value=io.BytesIO(b"tampered")):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            installer.download_wheel(installer.resolve_release(manifest_for(installer)), tmp_path)
    assert not list(tmp_path.iterdir())


def test_selector_precedence_and_empty_values_fail(installer, monkeypatch):
    monkeypatch.delenv("AZURE_AI_PROJECTS_VERSION", raising=False)
    assert installer.selected_version(None) == "latest"
    monkeypatch.setenv("AZURE_AI_PROJECTS_VERSION", SECOND_VERSION)
    assert installer.selected_version(None) == SECOND_VERSION
    assert installer.selected_version(VERSION) == VERSION
    monkeypatch.setenv("AZURE_AI_PROJECTS_VERSION", "")
    with pytest.raises(ValueError):
        installer.resolve_release(manifest_for(installer), installer.selected_version(None))


@pytest.mark.parametrize("relative_manifest", [False, True])
def test_pip_receives_verified_wheel_extra_and_requirements(installer, tmp_path, monkeypatch, relative_manifest):
    version = VERSION
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(manifest_for(installer, version)))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("openenv==0.6.0\n")
    release = installer.resolve_release(manifest_for(installer, version), version)
    installed_wheels = []

    def install(command, **kwargs):
        uri = next(arg for arg in command if arg.startswith("azure-ai-projects[rle] @ file:"))
        wheel = Path(url2pathname(uri.split(" @ ", 1)[1].removeprefix("file://")))
        assert wheel.name == release.filename
        assert wheel.read_bytes() == wheel_bytes(version)
        installed_wheels.append(wheel)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "install_rle_sdk.py", "--manifest", manifest.name if relative_manifest else str(manifest), "--version", version,
        "--requirements", str(requirements), "--index-url", "https://pypi.org/simple",
    ])
    with patch.object(installer, "urlopen", return_value=io.BytesIO(wheel_bytes(version))) as download:
        with patch.object(installer.subprocess, "run", side_effect=install) as run:
            installer.main()
    command = run.call_args.args[0]
    assert any(argument.startswith("azure-ai-projects[rle] @ file:") for argument in command)
    assert installed_wheels
    assert not installed_wheels[0].exists()
    download.assert_called_once_with(release.url, timeout=120)
    assert command[-4:] == ["--index-url", "https://pypi.org/simple", "--requirement", str(requirements)]
    assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize("selected", ["latest", VERSION])
def test_local_manifest_downloads_only_latest_or_environment_pin(installer, tmp_path, monkeypatch, selected):
    registry = manifest_for(installer, SECOND_VERSION)
    registry["versions"].update(manifest_for(installer)["versions"])
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(registry))
    version = SECOND_VERSION if selected == "latest" else selected
    monkeypatch.setenv("AZURE_AI_PROJECTS_VERSION", selected)
    monkeypatch.setattr(sys, "argv", ["install_rle_sdk.py", "--manifest", str(manifest)])
    with patch.object(installer, "urlopen", return_value=io.BytesIO(wheel_bytes(version))) as download, patch.object(installer.subprocess, "run") as run:
        installer.main()
    download.assert_called_once_with(registry["versions"][version]["url"], timeout=120)
    assert any(registry["versions"][version]["filename"] in arg for arg in run.call_args.args[0])


def test_local_manifest_tampered_download_never_installs(installer, tmp_path, monkeypatch):
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(manifest_for(installer)))
    monkeypatch.setattr(sys, "argv", ["install_rle_sdk.py", "--manifest", str(manifest)])
    with patch.object(installer, "urlopen", return_value=io.BytesIO(b"tampered")), patch.object(installer.subprocess, "run") as run:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            installer.main()
    run.assert_not_called()


def test_local_manifest_download_failure_never_installs(installer, tmp_path, monkeypatch):
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(manifest_for(installer)))
    monkeypatch.setattr(sys, "argv", ["install_rle_sdk.py", "--manifest", str(manifest)])
    with patch.object(installer, "urlopen", side_effect=OSError("Download failed")), patch.object(installer.subprocess, "run") as run:
        with pytest.raises(OSError, match="Download failed"):
            installer.main()
    run.assert_not_called()


def test_remote_manifest_downloads_and_installs_verified_wheel(installer, monkeypatch):
    manifest = manifest_for(installer)
    monkeypatch.setattr(sys, "argv", ["install_rle_sdk.py", "--manifest", installer.MANIFEST_URL])
    responses = [io.BytesIO(json.dumps(manifest).encode("utf-8")), io.BytesIO(wheel_bytes())]
    with patch.object(installer, "urlopen", side_effect=responses) as download:
        with patch.object(installer.subprocess, "run") as run:
            installer.main()
    assert download.call_count == 2
    assert download.call_args.args[0] == manifest["versions"][VERSION]["url"]
    run.assert_called_once()


def test_remote_checksum_failure_never_installs(installer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install_rle_sdk.py", "--manifest", installer.MANIFEST_URL])
    responses = [io.BytesIO(json.dumps(manifest_for(installer)).encode("utf-8")), io.BytesIO(b"tampered")]
    with patch.object(installer, "urlopen", side_effect=responses):
        with patch.object(installer.subprocess, "run") as run:
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                installer.main()
    run.assert_not_called()
