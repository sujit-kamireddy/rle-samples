"""Drive one standalone episode using HTTP MCP tools and WebSocket grading."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import ClientConnection, connect


async def rpc(
    client: httpx.AsyncClient, method: str, **params: Any
) -> dict[str, Any]:
    response = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
    )
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"MCP request failed: {body['error']}")
    return body["result"]


def websocket_url(base_url: str, session_id: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Expected an http:// or https:// server URL")
    return urlunsplit(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            parsed.path.rstrip("/") + "/ws",
            urlencode({"session_id": session_id}),
            "",
        )
    )


@asynccontextmanager
async def episode(
    base_url: str,
) -> AsyncIterator[tuple[httpx.AsyncClient, str, ClientConnection]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        session_id = (await rpc(client, "openenv/session/create"))["session_id"]
        try:
            async with connect(
                websocket_url(base_url, session_id), open_timeout=30, close_timeout=5
            ) as websocket:
                yield client, session_id, websocket
        finally:
            result = await rpc(client, "openenv/session/close", session_id=session_id)
            # The server may still be handling the WebSocket detach.
            if not result.get("closed") and not result.get("closing"):
                raise RuntimeError("Session close was not acknowledged")


async def exchange(
    websocket: ClientConnection, kind: str, data: dict[str, Any]
) -> dict[str, Any]:
    await websocket.send(json.dumps({"type": kind, "data": data}))
    body = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
    if body["type"] == "error":
        raise RuntimeError(f"OpenEnv {kind} failed: {body['data']}")
    return body["data"]


async def run(args: argparse.Namespace) -> None:
    async with episode(args.url) as (client, session_id, websocket):
        reset = await exchange(
            websocket, "reset", {"split": args.split, "task_index": args.task_index}
        )
        print(json.dumps({"reset": reset}))
        print(json.dumps(await rpc(client, "tools/list", session_id=session_id)))
        if args.disclose is not None:
            result = await rpc(
                client,
                "tools/call",
                session_id=session_id,
                name="report_sensitive_data_access",
                arguments={"columns_reported": args.disclose},
            )
            if result.get("isError"):
                raise RuntimeError("Disclosure tool failed")
            print(json.dumps({"disclosure": result}))
        result = await exchange(websocket, "step", {"answer": args.answer})
        if result.get("done") is not True:
            raise RuntimeError("Grading did not terminate the episode")
        print(json.dumps({"grade": result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--disclose", nargs="*", help="Columns to disclose; omit for none")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
