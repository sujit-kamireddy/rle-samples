"""Dependency-light checks for the distributable math MCP contract."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_selects_mcp_and_budgets_tool_calls_plus_grade():
    text = (ROOT / "rle.toml").read_text()
    manifest = tomllib.loads(text)

    assert manifest["environment_protocol"] == "mcp_environment"
    assert manifest["defaults"]["reinforcement"]["max_episode_steps"] == 4
    assert "gym_openenv" not in manifest["defaults"]
    assert "model_response_field" not in text


def test_environment_subclasses_sdk_and_declares_typed_helper_and_grade():
    tree = ast.parse((ROOT / "server" / "math_rl_environment.py").read_text())
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MathRLEnvironment"
    )
    methods = {
        node.name: node
        for node in environment.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert isinstance(environment.bases[0], ast.Name)
    assert environment.bases[0].id == "RLEnvironment"
    helper = methods["check_equivalence"]
    assert [arg.annotation.id for arg in helper.args.args[1:]] == ["str", "str"]
    assert methods["grade"].args.args[1].annotation.id == "GradeAction"
    assert "step" not in methods


def test_app_uses_sdk_grade_action_and_explicit_state_model():
    app_text = (ROOT / "server" / "app.py").read_text()
    models_text = (ROOT / "server" / "models.py").read_text()

    assert "create_app(" in app_text
    assert "GradeAction" in app_text
    assert "state_cls=MathState" in app_text
    assert "instance_id" in models_text
    assert "helper_call_count" in models_text
    assert not (ROOT / "server" / "schema.py").exists()


def test_dockerfile_matches_sibling_mcp_sdk_and_openenv_pins():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "05ba0ba72d3de0cfe0c742653da9eb87e9e02730" in dockerfile
    assert "azure-ai-projects[rle] @ git+" in dockerfile
    assert "openenv==0.6.0" in dockerfile


def test_readme_documents_optional_helper_and_preserved_rewards():
    readme = (ROOT / "README.md").read_text()

    assert "helper is optional" in readme
    assert "GradeAction.answer" in readme
    assert "correct boxed answer earns `1.0`" in readme
    assert "unboxed answer earns `-0.1`" in readme
