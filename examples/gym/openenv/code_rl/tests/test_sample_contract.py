"""Dependency-free checks for the distributable MCP sample contract."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_selects_mcp_and_has_no_schema_driven_fields():
    text = (ROOT / "rle.toml").read_text()
    manifest = tomllib.loads(text)

    assert manifest["harness"]["type"] == "Gym"
    assert manifest["harness"]["subtype"] == "OpenEnv"
    assert "rollout" not in manifest
    assert "defaults" not in manifest
    assert "model_response_field" not in text.replace("# model_response_field", "")


def test_environment_subclasses_sdk_and_registers_typed_tool():
    tree = ast.parse((ROOT / "server" / "code_rl_environment.py").read_text())
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CodeRLEnvironment"
    )
    methods = {
        node.name: node
        for node in environment.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert isinstance(environment.bases[0], ast.Name)
    assert environment.bases[0].id == "RLEnvironment"
    tool = methods["check_solution"]
    assert isinstance(tool, ast.AsyncFunctionDef)
    assert tool.args.args[1].arg == "code"
    assert isinstance(tool.args.args[1].annotation, ast.Name)
    assert tool.args.args[1].annotation.id == "str"
    assert methods["grade"].args.args[1].annotation.id == "GradeAction"
    assert "self.tool()(self.check_solution)" in ast.unparse(environment)


def test_obsolete_schema_contract_is_removed():
    assert not (ROOT / "server" / "schema.py").exists()


def test_dockerfile_matches_sibling_sdk_and_openenv_pins():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "ENV AZURE_AI_PROJECTS_VERSION=latest" in dockerfile
    assert "install_rle_sdk.py" in dockerfile
    assert "rle-releases:sdk/ai/azure-ai-projects/rle" in dockerfile
    assert "git+" not in dockerfile
    assert "openenv==0.6.0" in (ROOT / "requirements.txt").read_text()


def test_app_uses_grade_action_and_explicit_state_model():
    app = (ROOT / "server" / "app.py").read_text()

    assert "create_app(" in app
    assert "GradeAction" in app
    assert "state_cls=CodeState" in app


def test_catalog_and_root_readme_publish_only_the_migrated_sdk_samples():
    repository_root = ROOT.parents[3]
    catalog = tomllib.loads(
        (repository_root / "examples" / "gym" / "openenv" / "catalog.toml").read_text()
    )
    visible = [sample["name"] for sample in catalog["sample"] if sample["visible"]]
    readme = (repository_root / "README.md").read_text()

    assert visible == ["code_rl", "math_rl"]
    assert "`code_rl`" in readme
    assert "`math_rl`" in readme
    assert "SDK `RLEnvironment` MCP contract" in readme
