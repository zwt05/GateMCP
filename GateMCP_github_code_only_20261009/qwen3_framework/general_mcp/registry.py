"""Registry for the four paper tasks."""

from __future__ import annotations

from .adapters import BlocksworldAdapter, GSM8KAdapter, Game24Adapter, SVAMPAdapter, TaskAdapter


ADAPTERS: dict[str, type[TaskAdapter]] = {
    "game24": Game24Adapter,
    "blocksworld": BlocksworldAdapter,
    "gsm8k": GSM8KAdapter,
    "svamp": SVAMPAdapter,
}


def create_adapter(
    task: str,
) -> TaskAdapter:
    key = task.lower()
    if key not in ADAPTERS:
        raise ValueError(f"Unknown task {task!r}; available: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[key]()
