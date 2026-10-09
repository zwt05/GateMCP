#!/usr/bin/env python3
"""Task-scoped FastMCP server loaded from a frozen benchmark JSONL file."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from qwen3_framework.general_mcp.registry import create_adapter
from qwen3_framework.general_mcp.schema import load_examples


DATA_PATH = Path(os.environ["GENERAL_MCP_DATA"]).expanduser().resolve()
TASK = os.environ["GENERAL_MCP_TASK"].lower()
ADAPTER = create_adapter(TASK)
EXAMPLES = {item.id: item for item in load_examples(DATA_PATH, TASK)}

mcp = FastMCP(f"General Verification-Driven MCP: {TASK}")


def _example(example_id: str):
    try:
        return EXAMPLES[str(example_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown frozen example id: {example_id}") from exc


@mcp.tool
def verify_answer(example_id: str, answer: str) -> dict:
    """Strictly verify a student-generated final answer."""
    return ADAPTER.verify(_example(example_id), answer).to_dict()


@mcp.tool
def decompose_task(example_id: str, top_k: int = 5):
    """Return goal/subgoal structure without a complete answer."""
    return ADAPTER.assistance(_example(example_id), 1, top_k)


@mcp.tool
def generate_candidates(example_id: str, top_k: int = 5):
    """Return legal partial candidates without answer labels."""
    return ADAPTER.assistance(_example(example_id), 2, top_k)


@mcp.tool
def local_search(example_id: str, top_k: int = 5):
    """Check partial continuations while withholding the complete continuation."""
    return ADAPTER.assistance(_example(example_id), 3, top_k)


@mcp.tool
def verified_next_step(example_id: str, top_k: int = 5):
    """Return one verified next step, never the complete solution."""
    return ADAPTER.assistance(_example(example_id), 4, top_k)


if __name__ == "__main__":
    mcp.run()
