"""Whole-trajectory paper metrics and task-equal, three-seed aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
import statistics
from typing import Any


MACRO_METRICS = (
    "accuracy",
    "mean_l4_assistance_calls",
    "success_without_l4_rate",
    "mean_assistance_calls",
)
COUNT_METRICS = ("total_assistance_calls", "total_l4_assistance_calls")


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
    """Count all tool calls in each complete recorded trajectory.

    A successful row contributes to success_without_l4_rate only when the
    entire tool_calls list contains zero assistance=true verified_next_step
    entries. An autonomous success followed by an L4 call does not qualify.
    """
    samples = successes = successes_without_l4 = assistance_calls = l4_calls = 0
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {index} must be an object")
        if not isinstance(record.get("success"), bool):
            raise ValueError(f"Record {index} requires boolean success")
        calls = record.get("tool_calls")
        if not isinstance(calls, (list, tuple)):
            raise ValueError(f"Record {index} requires a complete tool_calls sequence")
        row_l4 = 0
        for call_index, call in enumerate(calls, 1):
            if not isinstance(call, Mapping) or not isinstance(call.get("name"), str) or not isinstance(call.get("assistance"), bool):
                raise ValueError(f"Record {index} tool call {call_index} requires name and boolean assistance")
            if call["assistance"] and call["name"] != "verify_answer":
                assistance_calls += 1
                if call["name"] == "verified_next_step":
                    row_l4 += 1
        l4_calls += row_l4
        samples += 1
        successes += int(record["success"])
        successes_without_l4 += int(record["success"] and row_l4 == 0)
    if samples == 0:
        raise ValueError("Cannot summarize empty input: no examples")
    return {
        "samples": samples,
        "successful_examples": successes,
        "successful_without_l4": successes_without_l4,
        "accuracy": successes / samples,
        "mean_l4_assistance_calls": l4_calls / samples,
        "success_without_l4_rate": successes_without_l4 / samples,
        "mean_assistance_calls": assistance_calls / samples,
        "total_assistance_calls": assistance_calls,
        "total_l4_assistance_calls": l4_calls,
    }


def aggregate_cells(
    cells: Iterable[Mapping[str, Any]],
    tasks: tuple[str, ...] = ("game24", "blocksworld", "gsm8k", "svamp"),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> list[dict[str, Any]]:
    """Average task-level rates/call means equally, then use sample SD.

    Absolute call counts are separately summed within each seed, never
    confused with task-equal per-example call means.
    """
    if len(tasks) != 4 or len(set(tasks)) != 4 or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Paper aggregation requires four distinct tasks and three distinct seeds")
    groups: dict[tuple[str, str], dict[tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    for cell in cells:
        model, method = cell.get("model"), cell.get("method")
        task, seed = cell.get("task"), cell.get("seed")
        if not isinstance(model, str) or not model or not isinstance(method, str) or not method:
            raise ValueError("Each cell requires model and method identifiers")
        if task not in tasks or seed not in seeds:
            raise ValueError("Cell has an unexpected task or seed")
        key = (task, seed)
        group = groups[(model, method)]
        if key in group:
            raise ValueError(f"Duplicate cell: {model}/{method}/{task}/{seed}")
        if not isinstance(cell.get("samples"), int) or cell["samples"] <= 0:
            raise ValueError("Each cell must summarize nonempty raw data")
        for metric in MACRO_METRICS + COUNT_METRICS:
            value = cell.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Cell requires finite metric {metric}")
        group[key] = cell
    if not groups:
        raise ValueError("Cannot aggregate empty input: no cells")
    required = {(task, seed) for task in tasks for seed in seeds}
    reports = []
    for (model, method), group in sorted(groups.items()):
        if set(group) != required:
            missing = sorted(required - set(group))
            raise ValueError(f"Incomplete four-task/three-seed group: {model}/{method}; missing {missing}")
        per_seed = {}
        for seed in seeds:
            task_cells = [group[(task, seed)] for task in tasks]
            values = {metric: statistics.mean(cell[metric] for cell in task_cells) for metric in MACRO_METRICS}
            values.update({metric: sum(cell[metric] for cell in task_cells) for metric in COUNT_METRICS})
            values["samples"] = sum(cell["samples"] for cell in task_cells)
            per_seed[str(seed)] = values
        summary = {
            metric: {
                "mean": statistics.mean(per_seed[str(seed)][metric] for seed in seeds),
                "sd": statistics.stdev(per_seed[str(seed)][metric] for seed in seeds),
            }
            for metric in MACRO_METRICS + COUNT_METRICS
        }
        reports.append({"model": model, "method": method, "per_seed": per_seed, "metrics": summary})
    return reports
