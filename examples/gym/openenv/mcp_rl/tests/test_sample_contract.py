"""Dependency-free checks for the distributable sample contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_selects_mcp_without_model_response_field():
    manifest = tomllib.loads((ROOT / "rle.toml").read_text())

    assert manifest["environment_protocol"] == "mcp_environment"
    assert "environment_protocol" not in manifest["rle"]
    assert "environmentProtocol" not in manifest["rle"]
    assert 0 < manifest["defaults"]["reinforcement"]["max_episode_steps"] <= 32
    assert "gym_openenv" not in manifest["defaults"]
    assert "model_response_field" not in (ROOT / "rle.toml").read_text()


def test_job_rows_select_the_one_deterministic_task():
    for name in ("train.jsonl", "validation.jsonl"):
        rows = [
            json.loads(line)
            for line in (ROOT / "job_data" / name).read_text().splitlines()
            if line
        ]
        assert rows == [{"left": 2, "right": 3}]


def test_environment_declares_typed_tool_and_sdk_base():
    tree = ast.parse((ROOT / "server" / "environment.py").read_text())
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ArithmeticRLEnvironment"
    )
    methods = {
        node.name: node
        for node in environment.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert isinstance(environment.bases[0], ast.Name)
    assert environment.bases[0].id == "RLEnvironment"
    assert [arg.annotation.id for arg in methods["add"].args.args[1:]] == ["int", "int"]
    reset_parameters = [arg.arg for arg in methods["reset"].args.args]
    assert reset_parameters == ["self", "seed", "episode_id", "left", "right"]
    assert "grade" in methods


def test_sample_has_no_legacy_fixed_task_id():
    sample_text = "\n".join(
        path.read_text()
        for path in [ROOT / "server" / "environment.py", ROOT / "server" / "models.py"]
    )

    assert "task_id" not in sample_text
    assert "19" not in sample_text
    assert "23" not in sample_text


def test_dockerfile_pins_the_known_rle_sdk_and_openenv():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "05ba0ba72d3de0cfe0c742653da9eb87e9e02730" in dockerfile
    assert "azure-ai-projects[rle] @ git+" in dockerfile
    assert "openenv==0.6.0" in dockerfile


def test_sample_is_registered_in_catalog_and_root_readme():
    catalog = tomllib.loads((ROOT.parent / "catalog.toml").read_text())
    visible = {
        sample["name"]
        for sample in catalog["sample"]
        if sample.get("visible", True)
    }
    repository_root = ROOT.parents[3]

    assert "mcp_rl" in visible
    assert "mcp_rl" in (repository_root / "README.md").read_text()
