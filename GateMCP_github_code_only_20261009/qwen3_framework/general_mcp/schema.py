"""Common schemas for all reasoning and planning tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskExample:
    """One benchmark item with private reference data.

    ``input`` is safe to show to the student. ``reference`` is available only
    to deterministic verifiers and internal search procedures.
    """

    id: str
    task: str
    input: dict[str, Any]
    reference: dict[str, Any]
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskExample":
        return cls(
            id=str(value["id"]),
            task=str(value["task"]).lower(),
            input=dict(value.get("input", {})),
            reference=dict(value.get("reference", {})),
            split=str(value.get("split", "test")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Verification:
    valid: bool
    normalized_answer: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptRecord:
    stage: int
    stage_name: str
    prompt: str
    raw_output: str
    normalized_answer: str
    verification: dict[str, Any]
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_model: str | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    latency_seconds: float
    assistance: bool
    leakage_detected: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    task: str
    example_id: str
    method: str
    student: str
    success: bool
    final_answer: str | None
    solved_stage: int | None
    solved_stage_name: str | None
    attempts: tuple[AttemptRecord, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    total_latency_seconds: float
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_examples(path: str | Path, task: str | None = None, split: str | None = None) -> list[TaskExample]:
    source = Path(path)
    examples: list[TaskExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = TaskExample.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
            if task is not None and item.task != task.lower():
                continue
            if split is not None and item.split != split:
                continue
            examples.append(item)
    return examples


def write_examples(path: str | Path, examples: Iterable[TaskExample]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return target
