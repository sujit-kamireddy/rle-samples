"""Dependency-free checks for the distributable RLE authoring skill."""

import json
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents" / "skills" / "rle"
MIRROR = ROOT / ".claude" / "skills" / "rle"
SAMPLES = (
    ROOT / "examples" / "gym" / "openenv" / "math_rl" / "rle.toml",
    ROOT / "examples" / "gym" / "openenv" / "code_rl" / "rle.toml",
    ROOT / "examples" / "harness" / "byoh" / "data_code_agent" / "rle" / "rle.toml",
    ROOT / "examples" / "harness" / "hosted-agent" / "competitive_intelligence_agent" / "rle" / "rle.toml",
)


class RLESkillTests(unittest.TestCase):
    def test_skill_trees_are_byte_identical(self):
        source = {path.relative_to(SKILL) for path in SKILL.rglob("*") if path.is_file()}
        mirror = {path.relative_to(MIRROR) for path in MIRROR.rglob("*") if path.is_file()}
        self.assertEqual(source, mirror)
        for relative in sorted(source):
            with self.subTest(file=str(relative)):
                self.assertEqual((SKILL / relative).read_bytes(), (MIRROR / relative).read_bytes())

    def test_discovery_metadata(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        frontmatter, body = text[4:].split("\n---\n", 1)
        self.assertIn(f"name: {SKILL.name}\n", frontmatter)
        description = frontmatter.split("description: >-\n", 1)[1]
        self.assertTrue(all(line.startswith("  ") for line in description.splitlines()))
        description = " ".join(line.strip() for line in description.splitlines())
        self.assertGreater(len(description), 0)
        self.assertLessEqual(len(description), 1024)
        for trigger in ("Gym", "BYOH", "HostedAgent", "rle.toml", "MCP", "invocation"):
            self.assertIn(trigger, description)
        self.assertLess(len(body.splitlines()), 500)

    def test_reference_links_exist_and_are_reachable(self):
        entry = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        references = set(re.findall(r"\]\(\./(references/[^)#]+)\)", entry))
        self.assertEqual(
            references,
            {path.relative_to(SKILL).as_posix() for path in (SKILL / "references").glob("*.md")},
        )
        for path in SKILL.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("{{", text, str(path))
            for target in re.findall(r"\]\((\./[^)#]+)(?:#[^)]*)?\)", text):
                with self.subTest(file=str(path), link=target):
                    self.assertTrue((path.parent / target).is_file())

    def test_json_examples_parse(self):
        count = 0
        for path in SKILL.rglob("*.md"):
            for block in re.findall(r"```json\n(.*?)\n```", path.read_text(encoding="utf-8"), re.S):
                with self.subTest(file=str(path), example=count):
                    json.loads(block)
                count += 1
        self.assertGreater(count, 0)

    def test_manifest_examples_cover_current_sample_types(self):
        text = (SKILL / "references" / "manifest.md").read_text(encoding="utf-8")
        examples = [
            tomllib.loads(block)
            for block in re.findall(r"```toml\n(.*?)\n```", text, re.S)
        ]
        sample_types = set()
        for path in SAMPLES:
            manifest = tomllib.loads(path.read_text(encoding="utf-8"))
            sample_types.add((manifest["harness"]["type"], manifest["harness"].get("protocol")))
        self.assertEqual(
            {(item["harness"]["type"], item["harness"].get("protocol")) for item in examples},
            sample_types,
        )
        for manifest in examples:
            with self.subTest(harness=manifest["harness"]):
                self.assertEqual(manifest["schema_version"], "1.0.0")
                self.assertNotIn("type", manifest["rle"])
                harness = manifest["harness"]
                for removed in ("subtype", "invocation_url", "agent_name", "agent_version"):
                    self.assertNotIn(removed, harness)
                if harness["type"] == "byoh":
                    self.assertTrue(harness["endpoint"].startswith("https://"))
                    self.assertIn(harness["protocol"], ("responses", "rle-invocation"))
                    if harness["protocol"] == "responses":
                        self.assertRegex(harness["endpoint"], r"/api/projects/[^/]+/agents/[^/]+$")
                else:
                    self.assertEqual(harness, {"type": "gym"})
                self.assertNotIn("defaults", manifest)
                self.assertNotIn("rollout", manifest)

    def test_invocation_and_session_examples(self):
        invocation = (SKILL / "references" / "invocation.md").read_text(encoding="utf-8")
        examples = [
            json.loads(block)
            for block in re.findall(r"```json\n(.*?)\n```", invocation, re.S)
        ]
        start = next(item for item in examples if "operation_id" in item)
        self.assertEqual(set(start), {"rollout_id", "operation_id", "agent_input", "rollout_context"})
        self.assertEqual(
            set(start["rollout_context"]),
            {"model_endpoint", "model_api_key", "mcp_endpoint", "mcp_bearer_token"},
        )
        self.assertIsInstance(start["agent_input"], list)
        self.assertIn("GET <endpoint>/rollouts/{operation_id}", invocation)
        self.assertIn("DELETE <endpoint>/rollouts/{operation_id}", invocation)
        succeeded = next(item for item in examples if item.get("status") == "succeeded")
        self.assertIsInstance(succeeded["output_text"], str)
        hosted = next(item for item in examples if "agent_session_id" in item)
        self.assertIs(hosted["background"], True)
        self.assertIs(hosted["stream"], False)
        self.assertIs(hosted["store"], True)
        self.assertEqual(hosted["input"][0]["content"][0]["type"], "input_text")
        mcp = (SKILL / "references" / "mcp.md").read_text(encoding="utf-8")
        for method in ("tools/list", "tools/call", "openenv/session/create", "openenv/session/close"):
            self.assertIn(f'"method":"{method}"', mcp)
        self.assertIn("/mcp?rle_session_id=", mcp)

    def test_manifest_documents_verified_cli_schema(self):
        manifest = (SKILL / "references" / "manifest.md").read_text(encoding="utf-8")
        fields = (
            "schema_version",
            "[rle].name",
            "[rle].version",
            "[rle].imageTag",
            "[harness].type",
            "[task].format",
            "[harness].endpoint",
            "[harness].protocol",
            "[harness].name",
            "[model].name",
            "[model].renderer_name",
            "[rollout.gym_openenv].model_response_field",
        )
        for field in fields:
            with self.subTest(field=field):
                self.assertIn(f"| `{field}` |", manifest)
        for path in SAMPLES:
            sample = tomllib.loads(path.read_text(encoding="utf-8"))
            for field in sample["train"]["grpo"]:
                self.assertIn(f"| `{field}` |", manifest)
        server = (SKILL / "references" / "server-contract.md").read_text(encoding="utf-8")
        self.assertIn("boolean `is_success`", server)

    def test_runtime_boundary_regression_guidance(self):
        mcp = (SKILL / "references" / "mcp.md").read_text(encoding="utf-8")
        for field in ("structured_content", "structuredContent", "is_error", "isError"):
            self.assertIn(field, mcp)
        self.assertIn('result.get("is_error") or result.get("isError")', mcp)
        server = (SKILL / "references" / "server-contract.md").read_text(encoding="utf-8")
        self.assertIn("fresh session", server)
        self.assertIn("WebSocket reset path", server)
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        self.assertIn("httpx.MockTransport", workflow)
        self.assertIn("azd ai rle skill install --no-prompt", workflow)
        invocation = (SKILL / "references" / "invocation.md").read_text(encoding="utf-8")
        self.assertIn("not a runnable Hosted Agent", invocation)
        self.assertIn("azure.ai.agentserver.responses.aio.streaming", invocation)
        self.assertIn("local mock capture proxy", invocation)
        anatomy = (SKILL / "references" / "anatomy.md").read_text(encoding="utf-8")
        self.assertIn("separate virtual environments", anatomy)
        self.assertIn("openai>=3.0.0", anatomy)
        self.assertIn("openai==2.54.0", anatomy)
        self.assertIn("python -m pip check", anatomy)
        self.assertIn("Do not disable TLS verification", workflow)

    def test_managed_preflight_and_release_capabilities_are_explicit(self):
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        for text in (
            "public `0.8.29-preview` is Loom-only",
            "development `0.8.30-preview`",
            "azd ai rle list --no-prompt",
            "DefaultAzureCredential",
            "--password-stdin",
            "before retrying a write",
            "Never drop `--mode eval`",
        ):
            self.assertIn(text, workflow)
        troubleshooting = (SKILL / "references" / "troubleshooting.md").read_text(encoding="utf-8")
        self.assertIn("before reset", troubleshooting)
        baseline = (SKILL / "references" / "upstream-contracts.md").read_text(encoding="utf-8")
        self.assertIn("has no `--mode`/`--loom` flags", baseline)
        self.assertIn("CaptureProxy/src/capture_proxy/server.py", baseline)

    def test_cloud_failure_diagnostics_and_cleanup(self):
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        for text in (
            "ContainerConfiguration.registry_connection_id",
            "not proof",
            "CMK",
            "InvalidRolloutRequest",
            "engine_busy",
            "session-creation timeout",
            "preflight snapshot",
            "temporary role assignments/connections",
            "verify their absence",
            "do not guess status semantics",
        ):
            with self.subTest(guidance=text):
                self.assertIn(text, workflow)
        troubleshooting = (SKILL / "references" / "troubleshooting.md").read_text(encoding="utf-8")
        for symptom in (
            "engine_busy",
            "InvalidRolloutRequest",
            "dependent-resource 403",
            "hosted_agent_infrastructure_provisioning_failed",
            "generic ADC 400",
            "policy.deployment_name",
            "only `loom` is supported",
            "do not resubmit an accepted or uncertain execution",
        ):
            self.assertIn(symptom, troubleshooting)

    def test_upstream_baseline_and_current_workflow_are_explicit(self):
        baseline = (SKILL / "references" / "upstream-contracts.md").read_text(encoding="utf-8")
        commits = set(re.findall(r"`([0-9a-f]{40})`", baseline))
        self.assertEqual(len(commits), 5)
        for component in ("RLE CLI", "RLE SDK", "Vienna RLE", "Public distributions"):
            self.assertIn(component, baseline)
        self.assertRegex(baseline, r"Checked on \*\*\d{4}-\d{2}-\d{2}\*\*")
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        for command in (
            "azd extension install azure.ai.rle --source rle",
            "AZD_AI_RLE_ENABLE_ALL",
            "--mode eval",
            "--mode train",
            "--mode eval --loom",
        ):
            self.assertIn(command, workflow)
        self.assertNotIn('$env:AZD_AI_RLE_ENABLE = "true"', workflow)
        self.assertIn("intentionally never trainable", workflow)

    def test_latest_schema_migration_and_cli_compatibility(self):
        manifest = (SKILL / "references" / "manifest.md").read_text(encoding="utf-8")
        for text in (
            "harbor", "not yet supported", "package", "opencode",
            "Removed `subtype`", "Agent versions are no longer",
            "same-project Foundry", "not the environment MCP contract",
        ):
            with self.subTest(guidance=text):
                self.assertIn(text, manifest)
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        self.assertIn("cannot publish these new manifests", workflow)
        self.assertIn("--type byoh --protocol responses --endpoint", workflow)
        self.assertIn("--type byoh --protocol rle-invocation --endpoint", workflow)
        self.assertNotIn("init math_lab --type Gym", workflow)
        self.assertNotIn("--subtype HostedAgent", workflow)

    def test_managed_evidence_and_hosted_region_guidance(self):
        workflow = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
        for text in (
            "HostedAgent availability list",
            "Sweden Central",
            "West US 2",
            "does not rebind that agent",
            "rollout_graph.turns",
            "loss_mask",
            "finite logprobs",
            "structural diagnostic before asserting",
            "don't replay an accepted execution",
        ):
            with self.subTest(guidance=text):
                self.assertIn(text, workflow)
        troubleshooting = (SKILL / "references" / "troubleshooting.md").read_text(encoding="utf-8")
        for text in ("masked internal error", "exact owned agent/version/session", "registration/deletion"):
            self.assertIn(text, troubleshooting)


if __name__ == "__main__":
    unittest.main()
