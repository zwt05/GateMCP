"""Local and real FastMCP transports for the generic task API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from .adapters.base import TaskAdapter
from .schema import TaskExample


@dataclass(frozen=True)
class ToolResponse:
    name: str
    arguments: dict[str, Any]
    data: Any
    latency_seconds: float
    error: str | None = None


class LocalTaskGateway:
    def __init__(self, adapter: TaskAdapter, examples: Mapping[str, TaskExample]):
        self.adapter = adapter
        self.examples = dict(examples)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResponse:
        started = time.perf_counter()
        try:
            example = self.examples[str(arguments["example_id"])]
            if name == "verify_answer":
                data = self.adapter.verify(example, str(arguments.get("answer", ""))).to_dict()
            else:
                stages = {
                    "decompose_task": 1,
                    "generate_candidates": 2,
                    "local_search": 3,
                    "verified_next_step": 4,
                }
                data = self.adapter.assistance(
                    example, stages[name], int(arguments.get("top_k", 5))
                )
            return ToolResponse(name, arguments, data, time.perf_counter() - started)
        except Exception as exc:
            return ToolResponse(
                name,
                arguments,
                None,
                time.perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            )


class FastMCPTaskGateway:
    def __init__(
        self,
        data_path: str | Path,
        task: str,
    ):
        self.data_path = str(Path(data_path).expanduser().resolve())
        self.task = task
        self.server_path = str(Path(__file__).with_name("mcp_server.py").resolve())
        self._client = None

    async def __aenter__(self):
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StdioTransport
        except ImportError as exc:
            raise RuntimeError("Install fastmcp before using --transport mcp") from exc
        package_root = str(Path(__file__).resolve().parents[2])
        existing = os.environ.get("PYTHONPATH")
        python_path = package_root if not existing else package_root + os.pathsep + existing
        transport = StdioTransport(
            command=sys.executable,
            args=[self.server_path],
            env={
                "PYTHONPATH": python_path,
                "GENERAL_MCP_DATA": self.data_path,
                "GENERAL_MCP_TASK": self.task,
            },
            cwd=str(Path(self.server_path).parent),
            keep_alive=False,
        )
        self._client = Client(transport)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
        return False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResponse:
        if self._client is None:
            raise RuntimeError("FastMCPTaskGateway must be used inside async with")
        started = time.perf_counter()
        try:
            result = await self._client.call_tool(name, arguments)
            data = result.data
            if data is None and result.content:
                text = getattr(result.content[0], "text", "")
                try:
                    data = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    data = text
            return ToolResponse(name, arguments, data, time.perf_counter() - started)
        except Exception as exc:
            return ToolResponse(
                name,
                arguments,
                None,
                time.perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            )
