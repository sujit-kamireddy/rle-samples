"""Proves the tool credentials reach the environment OpenCode exports into its sandbox.

These tests run against the real Harbor `OpenCode` class rather than a stub, because the whole
point of `rollout_tools` is that the obvious injection point (`AgentConfig.env`) does not work for
this harness. A stub would happily confirm a mechanism that Harbor discards.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
import unittest

from harbor.agents.installed.opencode import OpenCode

from server import rollout_tools
from server.rollout_tools import ENDPOINT_VAR, TOKEN_VAR

ENDPOINT = "https://rle.example/rle/v1.0/rollouts/rollout-abc/tools"
TOKEN = "tok-abc"


def _agent(**overrides: object) -> OpenCode:
    """Builds a real OpenCode agent configured the way the openenv seam configures it."""
    kwargs: dict = {
        "logs_dir": pathlib.Path(tempfile.mkdtemp()),
        "model_name": "harbor-intercept/Qwen3-32B",
        "opencode_config": {"provider": {"harbor-intercept": {}}},
        "extra_env": {"OPENAI_API_KEY": "session-key"},
    }
    kwargs.update(overrides)
    return OpenCode(**kwargs)


class ExportedEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(rollout_tools.install(), "delivery patch must install")

    def test_harbor_drops_unlisted_variables_without_the_patch(self) -> None:
        """The premise: extra_env is a lookup source, not an export channel."""
        agent = _agent(extra_env={ENDPOINT_VAR: ENDPOINT, TOKEN_VAR: TOKEN})
        exported = dict(agent.model_connection.env)
        self.assertNotIn(ENDPOINT_VAR, exported)
        self.assertNotIn(TOKEN_VAR, exported)

    def test_scope_adds_both_variables(self) -> None:
        agent = _agent()
        with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
            exported = dict(agent.model_connection.env)
        self.assertEqual(exported.get(ENDPOINT_VAR), ENDPOINT)
        self.assertEqual(exported.get(TOKEN_VAR), TOKEN)

    def test_outside_a_scope_nothing_is_added(self) -> None:
        agent = _agent()
        exported = dict(agent.model_connection.env)
        self.assertNotIn(ENDPOINT_VAR, exported)
        self.assertNotIn(TOKEN_VAR, exported)

    def test_scope_is_released_on_exit(self) -> None:
        agent = _agent()
        with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
            pass
        self.assertNotIn(ENDPOINT_VAR, dict(agent.model_connection.env))

    def test_scope_is_released_when_the_rollout_raises(self) -> None:
        agent = _agent()
        with self.assertRaises(RuntimeError):
            with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
                raise RuntimeError("rollout blew up")
        self.assertNotIn(ENDPOINT_VAR, dict(agent.model_connection.env))

    def test_missing_capability_is_a_no_op(self) -> None:
        """A payload without tool credentials must behave as it did before this feature."""
        agent = _agent()
        for endpoint, token in (("", TOKEN), (ENDPOINT, ""), ("", "")):
            with rollout_tools.rollout_tools(endpoint, token):
                exported = dict(agent.model_connection.env)
            self.assertNotIn(ENDPOINT_VAR, exported)
            self.assertNotIn(TOKEN_VAR, exported)

    def test_provider_configuration_survives_injection(self) -> None:
        """Injection must not disturb what the seam worked to set up."""
        agent = _agent()
        before = agent.model_connection
        with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
            during = agent.model_connection
        self.assertEqual(before.provider, during.provider)
        self.assertEqual(before.api_key, during.api_key)
        self.assertEqual(before.base_url, during.base_url)
        self.assertEqual(before.configured_base_url, during.configured_base_url)
        self.assertEqual(
            dict(before.env),
            {k: v for k, v in during.env.items() if k not in (ENDPOINT_VAR, TOKEN_VAR)},
        )

    def test_install_is_idempotent(self) -> None:
        agent = _agent()
        for _ in range(3):
            self.assertTrue(rollout_tools.install())
        self.assertTrue(
            getattr(OpenCode.__dict__["model_connection"].fget, "_injects_rollout_tools", False)
        )
        with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
            exported = dict(agent.model_connection.env)
        self.assertEqual(exported.get(TOKEN_VAR), TOKEN)


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    """The reason this is a ContextVar rather than a module-level global.

    With a global, the assertion below fails non-deterministically and silently in production: one
    rollout's sandbox boots holding another rollout's token and files its disclosure against the
    wrong rollout id, with no error anywhere.
    """

    async def asyncSetUp(self) -> None:
        rollout_tools.install()

    async def test_concurrent_rollouts_keep_their_own_credentials(self) -> None:
        async def rollout(index: int) -> tuple[str, str]:
            endpoint, token = f"https://rle.example/r{index}/tools", f"tok-{index}"
            agent = _agent()
            with rollout_tools.rollout_tools(endpoint, token):
                # Yield control repeatedly so the event loop interleaves every rollout between
                # the moment credentials are scoped and the moment the sandbox env is built.
                for _ in range(5):
                    await asyncio.sleep(0)
                exported = dict(agent.model_connection.env)
            return exported.get(ENDPOINT_VAR, ""), exported.get(TOKEN_VAR, "")

        results = await asyncio.gather(*(rollout(i) for i in range(25)))
        expected = [(f"https://rle.example/r{i}/tools", f"tok-{i}") for i in range(25)]
        self.assertEqual(results, expected)

    async def test_credentials_survive_a_worker_thread(self) -> None:
        """Harbor runs parts of a trial off the event loop; contextvars must follow."""

        def read() -> dict[str, str]:
            return dict(_agent().model_connection.env)

        with rollout_tools.rollout_tools(ENDPOINT, TOKEN):
            exported = await asyncio.to_thread(read)
        self.assertEqual(exported.get(TOKEN_VAR), TOKEN)


if __name__ == "__main__":
    unittest.main()
