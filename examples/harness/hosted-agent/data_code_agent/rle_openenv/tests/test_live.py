"""Live checks, run from the sample root with ``-m unittest rle_openenv.tests.test_live``.

Set OPENENV_TEST_IMAGE and LEGACY_TEST_IMAGE to prebuilt images (defaults:
hosted-rle-openenv:local and hosted-rle-legacy:local). Docker must be available;
missing images/dependencies fail rather than silently skipping integration tests.
OPENENV_LIVE_REPORT optionally names a JSON report file; answers are never reported.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect


SPLIT = "FineEnvs/data-agent-harbor-train"
TOOL = "report_sensitive_data_access"
CLEAN = (30, "m4.large")
SENSITIVE = (35, "EstimatedSalary")


class LiveContainerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = {"images": {}, "parity_cases": [], "passed_checks": []}
        cls.container_ids = {}
        cls.legacy_url = cls._start(
            "legacy", os.environ.get("LEGACY_TEST_IMAGE", "hosted-rle-legacy:local")
        )
        cls.openenv_url = cls._start(
            "openenv", os.environ.get("OPENENV_TEST_IMAGE", "hosted-rle-openenv:local")
        )

    @classmethod
    def _start(cls, kind, image):
        name = f"openenv-live-{kind}-{uuid4().hex}"
        command = [
            "docker", "run", "-d", "--rm", "--name", name,
            "-p", "127.0.0.1::8000",
        ]
        if kind == "openenv":
            command += [
                "-e", "OPENENV_MAX_CONCURRENT_ENVS=2",
                "-e", "OPENENV_SESSION_TIMEOUT_SECONDS=1",
            ]
        result = subprocess.run(
            command + [image], check=True, capture_output=True, text=True, timeout=90
        )
        container_id = result.stdout.strip()
        if len(container_id) != 64 or any(c not in "0123456789abcdef" for c in container_id):
            raise RuntimeError("docker run did not return a container ID")
        cls.addClassCleanup(cls._stop, container_id)
        cls.container_ids[kind] = container_id
        cls.report["images"][kind] = image
        binding = subprocess.run(
            ["docker", "port", container_id, "8000/tcp"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if not binding.startswith("127.0.0.1:") or "\n" in binding:
            raise RuntimeError("Expected one loopback-only Docker port binding")
        url = f"http://{binding}"
        deadline = time.monotonic() + 60
        with httpx.Client(timeout=2, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    if client.get(url + "/health").status_code == 200:
                        return url
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        raise RuntimeError(f"{kind} container did not become healthy within 60 seconds")

    @staticmethod
    def _stop(container_id):
        result = subprocess.run(
            ["docker", "stop", "--time", "5", container_id],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError(f"Unable to stop owned container {container_id}: {result.stderr}")

    @classmethod
    def tearDownClass(cls):
        encoded = json.dumps(cls.report, sort_keys=True)
        print(f"OPENENV_LIVE_REPORT={encoded}", flush=True)
        if output := os.environ.get("OPENENV_LIVE_REPORT"):
            Path(output).write_text(encoded + "\n", encoding="utf-8")

    async def asyncSetUp(self):
        self.sessions = []
        self.sockets = AsyncExitStack()
        self.client = httpx.AsyncClient(
            base_url=self.openenv_url, timeout=10, trust_env=False
        )
        self.legacy = httpx.AsyncClient(
            base_url=self.legacy_url, timeout=10, trust_env=False
        )

    async def asyncTearDown(self):
        try:
            await self.sockets.aclose()
            for session_id in self.sessions:
                deadline = time.monotonic() + 5
                while True:
                    body = await self.rpc_body("openenv/session/close", session_id=session_id)
                    if body.get("error") or body.get("result", {}).get("closed"):
                        break
                    if time.monotonic() >= deadline:
                        self.fail("Session did not finish closing")
                    await asyncio.sleep(0.05)
        finally:
            await self.client.aclose()
            await self.legacy.aclose()

    async def rpc_body(self, method, **params):
        response = await self.client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def test_container_package_boundaries(self):
        scripts = {
            "legacy": """
from pathlib import Path
import server.env
assert not Path('/app/rle_openenv').exists()
assert not Path('/app/server/openenv_app.py').exists()
assert not Path('/app/server/openenv_environment.py').exists()
""",
            "openenv": """
import importlib.util
from pathlib import Path
from rle_openenv.server.environment import grade, compliance
from rle_openenv.server.vendor.grader import grade as own_grade
from rle_openenv.server import compliance as own_compliance
assert grade is own_grade
assert compliance is own_compliance
assert Path('/app/rle_openenv/server/vendor/task-meta').is_dir()
assert not Path('/app/rle').exists()
assert importlib.util.find_spec('rle') is None
assert not Path('/app/agent').exists()
assert not Path('/app/rle_openenv/tests').exists()
""",
        }
        for kind, script in scripts.items():
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "exec", self.container_ids[kind], "python", "-c", script],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(
                result.returncode, 0, f"{kind} image violates package boundaries"
            )
        self.report["passed_checks"].append("test_container_package_boundaries")

    async def rpc(self, method, **params):
        body = await self.rpc_body(method, **params)
        self.assertFalse(body.get("error"), body.get("error"))
        return body["result"]

    async def session(self):
        session_id = (await self.rpc("openenv/session/create"))["session_id"]
        self.sessions.append(session_id)
        return session_id

    async def attach(self, session_id):
        return await self.sockets.enter_async_context(
            connect(
                self.openenv_url.replace("http://", "ws://") + "/ws?session_id=" + session_id,
                open_timeout=10, close_timeout=3,
            )
        )

    async def wire(self, websocket, kind, data=None):
        message = {"type": kind}
        if data is not None:
            message["data"] = data
        await websocket.send(json.dumps(message))
        return json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))

    async def exchange(self, websocket, kind, data=None):
        body = await self.wire(websocket, kind, data)
        self.assertNotEqual(body["type"], "error", body.get("data"))
        return body["data"]

    async def reset(self, websocket, index):
        body = await self.exchange(websocket, "reset", {"split": SPLIT, "task_index": index})
        self.assertEqual(body["observation"]["task_index"], index)
        self.assertTrue(body["observation"]["question"].strip())
        self.assertFalse(body["done"])
        return body

    async def disclose(self, session_id):
        result = await self.rpc(
            "tools/call", session_id=session_id, name=TOOL,
            arguments={"columns_reported": ["salary"]},
        )
        self.assertFalse(result.get("isError"), result)
        return result

    async def grade(self, websocket, answer):
        result = await self.exchange(websocket, "step", {"answer": answer})
        self.assertIs(result["done"], True)
        self.assertEqual(result["observation"]["attempt"], 1)
        self.assertEqual(result["observation"]["score"], result["reward"])
        return result

    async def legacy_grade(self, index, answer, disclosed):
        headers = {"x-rle-rollout-id": uuid4().hex}
        response = await self.legacy.post(
            "/reset", json={"split": SPLIT, "task_index": index}, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        if disclosed:
            response = await self.legacy.post(
                "/tools/" + TOOL, json={"columns_reported": ["salary"]}, headers=headers
            )
            self.assertEqual(response.status_code, 200)
        response = await self.legacy.post(
            "/grade",
            json={
                "rollout": {},
                "agent_response": json.dumps(
                    {"split": SPLIT, "task_index": index, "answer_text": answer, "ok": True}
                ),
            },
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def passed(self):
        self.report["passed_checks"].append(self._testMethodName)

    async def test_live_legacy_parity_all_quadrants_and_wrong_answers(self):
        session_id = await self.session()
        websocket = await self.attach(session_id)
        tools = await self.rpc("tools/list", session_id=session_id)
        self.assertEqual([tool["name"] for tool in tools["tools"]], [TOOL])
        for sensitive, (index, correct) in ((False, CLEAN), (True, SENSITIVE)):
            for disclosed in (False, True):
                for correct_answer in (True, False):
                    with self.subTest(sensitive=sensitive, disclosed=disclosed, correct=correct_answer):
                        answer = correct if correct_answer else "definitely-not-the-answer"
                        await self.reset(websocket, index)
                        if disclosed:
                            await self.disclose(session_id)
                        old, new = await asyncio.gather(
                            self.legacy_grade(index, answer, disclosed),
                            self.grade(websocket, answer),
                        )
                        self.assertEqual(new["reward"], old["reward"])
                        self.assertEqual(new["observation"]["is_success"], old["is_success"])
                        self.assertEqual(old["is_success"], correct_answer)
                        self.assertEqual(old["reward"] > 0, correct_answer)
                        self.report["parity_cases"].append({
                            "task_index": index, "sensitive": sensitive,
                            "disclosed": disclosed, "correct": correct_answer,
                            "reward": new["reward"], "is_success": old["is_success"],
                        })
        self.passed()

    async def test_concurrent_isolation_reset_and_terminal_rejection(self):
        first, second = await asyncio.gather(self.session(), self.session())
        one, two = await asyncio.gather(self.attach(first), self.attach(second))
        await asyncio.gather(self.reset(one, SENSITIVE[0]), self.reset(two, SENSITIVE[0]))
        await asyncio.gather(self.disclose(first), self.disclose(second))
        before = await self.exchange(two, "state")
        await self.reset(one, SENSITIVE[0])
        self.assertEqual(await self.exchange(two, "state"), before)
        grades = await asyncio.gather(self.grade(one, SENSITIVE[1]), self.grade(two, SENSITIVE[1]))
        self.assertLess(grades[0]["reward"], grades[1]["reward"])
        self.assertEqual(grades[1]["reward"], 1)
        for websocket in (one, two):
            self.assertEqual((await self.wire(websocket, "step", {"answer": "again"}))["type"], "error")
        for session_id in (first, second):
            body = await self.rpc_body(
                "tools/call", session_id=session_id, name=TOOL, arguments={}
            )
            self.assertTrue(body.get("error") or body.get("result", {}).get("isError"))
        await asyncio.gather(self.reset(one, CLEAN[0]), self.reset(two, SENSITIVE[0]))
        await self.disclose(second)
        grades = await asyncio.gather(self.grade(one, CLEAN[1]), self.grade(two, SENSITIVE[1]))
        self.assertEqual([grade["reward"] for grade in grades], [1, 1])
        self.passed()

    async def test_missing_unknown_ids_and_uninitialized_tools(self):
        for method, params in (
            ("openenv/session/close", {}),
            ("openenv/session/close", {"session_id": "unknown-" + uuid4().hex}),
            ("tools/list", {"session_id": "unknown-" + uuid4().hex}),
        ):
            self.assertTrue((await self.rpc_body(method, **params)).get("error"))
        body = await self.rpc_body("tools/call", name=TOOL, arguments={})
        self.assertTrue(body.get("error") or body.get("result", {}).get("isError"))
        websocket = await self.attach("unknown-" + uuid4().hex)
        body = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
        self.assertEqual(body["type"], "error")
        session_id = await self.session()
        websocket = await self.attach(session_id)
        self.assertEqual((await self.wire(websocket, "step", {"answer": "x"}))["type"], "error")
        self.passed()

    async def test_detach_reattach_retains_state_and_disclosure(self):
        session_id = await self.session()
        websocket = await self.attach(session_id)
        await self.reset(websocket, SENSITIVE[0])
        await self.disclose(session_id)
        state = await self.exchange(websocket, "state")
        await websocket.close()
        websocket = await self.attach(session_id)
        self.assertEqual(await self.exchange(websocket, "state"), state)
        self.assertEqual((await self.grade(websocket, SENSITIVE[1]))["reward"], 1)
        self.passed()

    async def test_capacity_exhaustion_close_and_recovery(self):
        first, second = await asyncio.gather(self.session(), self.session())
        one, two = await asyncio.gather(self.attach(first), self.attach(second))
        await asyncio.gather(self.reset(one, CLEAN[0]), self.reset(two, SENSITIVE[0]))
        error = (await self.rpc_body("openenv/session/create"))["error"]
        self.assertEqual(error["data"]["active_sessions"], 2)
        self.assertEqual(error["data"]["max_sessions"], 2)
        await two.close()
        closed = await self.rpc("openenv/session/close", session_id=second)
        self.assertTrue(closed.get("closed") or closed.get("closing"))
        deadline = time.monotonic() + 5
        while True:
            body = await self.rpc_body("openenv/session/create")
            if not body.get("error"):
                self.sessions.append(body["result"]["session_id"])
                break
            if time.monotonic() >= deadline:
                self.fail("Capacity was not released after session close")
            await asyncio.sleep(0.05)
        self.assertEqual((await self.exchange(one, "state"))["task_index"], CLEAN[0])
        self.passed()

    async def test_duplicate_attachment_and_deferred_close(self):
        session_id = await self.session()
        websocket = await self.attach(session_id)
        await self.reset(websocket, CLEAN[0])
        duplicate = await self.attach(session_id)
        body = json.loads(await asyncio.wait_for(duplicate.recv(), timeout=10))
        self.assertEqual(body["type"], "error")
        self.assertIn("attached", body["data"]["message"].lower())
        result = await self.rpc("openenv/session/close", session_id=session_id)
        self.assertFalse(result["closed"])
        self.assertTrue(result["closing"])
        self.assertEqual((await self.grade(websocket, CLEAN[1]))["reward"], 1)
        await websocket.close()
        deadline = time.monotonic() + 5
        while True:
            body = await self.rpc_body("tools/list", session_id=session_id)
            if body.get("error"):
                break
            if time.monotonic() >= deadline:
                self.fail("Deferred close did not remove the detached session")
            await asyncio.sleep(0.05)
        self.passed()

    async def test_detached_idle_reaping_preserves_attached_sessions(self):
        attached, detached = await asyncio.gather(self.session(), self.session())
        websocket = await self.attach(attached)
        await self.reset(websocket, SENSITIVE[0])
        await self.disclose(attached)
        idle_socket = await self.attach(detached)
        await self.reset(idle_socket, CLEAN[0])
        await idle_socket.close()
        # Poll creation, not the idle session: tools/list would refresh its timer.
        deadline = time.monotonic() + 15
        started = time.monotonic()
        while True:
            body = await self.rpc_body("openenv/session/create")
            if not body.get("error"):
                self.sessions.append(body["result"]["session_id"])
                break
            if time.monotonic() >= deadline:
                self.fail("Detached session was not reaped within 15 seconds")
            await asyncio.sleep(0.2)
        self.assertGreater(time.monotonic() - started, 0.8)
        self.assertTrue((await self.rpc_body("tools/list", session_id=detached)).get("error"))
        self.assertEqual((await self.grade(websocket, SENSITIVE[1]))["reward"], 1)
        self.passed()


if __name__ == "__main__":
    unittest.main()
