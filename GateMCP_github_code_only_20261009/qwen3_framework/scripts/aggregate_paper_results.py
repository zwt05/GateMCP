#!/usr/bin/env python3
"""Read complete raw JSONL cells and write task-equal three-seed paper metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qwen3_framework.general_mcp.controller import (
    LEGACY_METHOD_ALIASES,
    PAPER_METHODS,
    normalize_method,
)
from qwen3_framework.general_mcp.metrics import aggregate_cells, summarize_records
from qwen3_framework.general_mcp.schema import load_examples
from qwen3_framework.scripts.run_general_experiment import seed_partition
from qwen3_framework.scripts.run_paper_matrix import load_protocol, validate_frozen_data


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cell(path, spec, expected_ids, expected_data_sha256):
    """Reject partial, duplicated, misidentified, or incompatible raw cells."""
    seen = set()
    optional_missing = set()
    expected_protocol = {
        "task": spec["task"],
        "method": spec["method"],
        "data_sha256": expected_data_sha256,
        "cumulative_output_token_budget": 3000,
        "per_call_output_token_cap": 3000 if spec["method"] == "direct" else 600,
        "game24_cot_prompt": spec["task"] == "game24",
    }
    optional_protocol = {
        "orchestration_seed": spec["seed"],
        "seed": spec["seed"],
        "temperature": 0,
        "transport": "mcp",
        "evaluation_split": "test",
        "thinking": False,
    }

    def records():
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid raw JSONL in {path.name}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError("Raw JSONL rows must be objects")
                example_id = record.get("example_id")
                if not isinstance(example_id, str) or example_id not in expected_ids:
                    raise ValueError(f"Unexpected or missing frozen example_id: {example_id!r}")
                if example_id in seen:
                    raise ValueError(f"Duplicate raw example_id: {example_id}")
                seen.add(example_id)
                for key, value in expected_protocol.items():
                    actual = record.get(key)
                    if key == "method" and isinstance(actual, str):
                        actual = normalize_method(actual)
                    if actual != value:
                        raise ValueError(f"Raw cell protocol mismatch for {key}: {path.name}:{line_number}")
                for key, value in optional_protocol.items():
                    if key not in record:
                        optional_missing.add(key)
                    elif record[key] != value:
                        raise ValueError(f"Raw cell protocol mismatch for {key}: {path.name}:{line_number}")
                yield record

    metrics = summarize_records(records())
    if seen != expected_ids:
        raise ValueError(f"Incomplete raw cell {path.name}: {len(seen)}/{len(expected_ids)} frozen examples")
    return {
        "model": spec["model"], "task": spec["task"], "method": spec["method"], "seed": spec["seed"],
        **metrics,
        "source_file": path.name,
        "source_sha256": file_sha256(path),
        "missing_optional_protocol_fields": sorted(optional_missing),
    }


def build_parser():
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-root", type=Path, help="Output tree created by run_paper_matrix.py")
    source.add_argument("--manifest", type=Path, help="JSON cells with model/task/method/seed/path; paths relative to the manifest")
    parser.add_argument("--model-key", choices=[m["key"] for m in protocol["models"]], action="append")
    parser.add_argument("--method", choices=(*PAPER_METHODS, *LEGACY_METHOD_ALIASES), action="append")
    parser.add_argument("--data-root", type=Path, default=ROOT / "qwen3_framework/data/full_evaluation")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Write .json, .csv and .md reports; raw inputs are read-only")
    return parser


def collect_specs(args, protocol):
    if args.manifest:
        content = json.loads(args.manifest.read_text(encoding="utf-8"))
        specs = content.get("cells") if isinstance(content, dict) else content
        if not isinstance(specs, list) or not specs:
            raise ValueError("Input manifest must contain a nonempty cells list")
        base = args.manifest.resolve().parent
    else:
        specs = []
        base = args.results_root.resolve()
        for model in protocol["models"]:
            methods = PAPER_METHODS if model["matrix"] == "main" else protocol["scale_methods"]
            for task in protocol["tasks"]:
                for seed in protocol["seeds"]:
                    for method in methods:
                        relative = Path(model["key"]) / task["key"] / method / f"seed_{seed}.jsonl"
                        if not (base / relative).is_file():
                            for legacy_method, paper_method in LEGACY_METHOD_ALIASES.items():
                                legacy_path = base / model["key"] / task["key"] / legacy_method / f"seed_{seed}.jsonl"
                                if paper_method == method and legacy_path.is_file():
                                    relative = legacy_path.relative_to(base)
                                    break
                        specs.append({"model": model["key"], "task": task["key"], "method": method, "seed": seed, "path": str(relative)})
    selected = []
    models = {m["key"]: m for m in protocol["models"]}
    tasks = {task["key"] for task in protocol["tasks"]}
    selected_methods = {normalize_method(item) for item in args.method or []}
    for spec in specs:
        if not isinstance(spec, dict) or not all(key in spec for key in ("model", "task", "method", "seed", "path")):
            raise ValueError("Every source cell needs model/task/method/seed/path")
        method = normalize_method(str(spec["method"]))
        if spec["model"] not in models or spec["task"] not in tasks or spec["seed"] not in protocol["seeds"] or method not in PAPER_METHODS:
            raise ValueError("Source manifest contains a non-paper model/task/method/seed")
        if models[spec["model"]]["matrix"] == "scale" and method not in protocol["scale_methods"]:
            raise ValueError("Scale-model cells are restricted to Direct and GateMCP")
        if args.model_key and spec["model"] not in args.model_key:
            continue
        if selected_methods and method not in selected_methods:
            continue
        path = Path(spec["path"]).expanduser()
        normalized_spec = dict(spec)
        normalized_spec["method"] = method
        selected.append((normalized_spec, path if path.is_absolute() else base / path))
    if not selected:
        raise ValueError("No paper cells selected")
    return selected


def write_reports(prefix, cells, groups):
    report = {
        "schema_version": 1,
        "source": "complete raw JSONL trajectories; summary files are not inputs",
        "aggregation": "four tasks equally weighted within each seed; mean and sample SD (ddof=1) across seeds42/43/44",
        "mean_l4_assistance_calls": "whole-trajectory count of name=verified_next_step and assistance=true, divided by all task examples",
        "success_without_l4_rate": "final success=true AND zero such L4 calls anywhere in tool_calls, divided by all task examples",
        "assistance_calls": "assistance=true calls other than verify_answer; absolute totals are reported separately from task-equal per-example means",
        "model_seed_identity": "official result path or caller-supplied cell manifest; missing optional legacy row fields are disclosed per cell",
        "cells": cells,
        "groups": groups,
    }
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    md_path = Path(str(prefix) + ".md")
    for output in (json_path, csv_path, md_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing report: {output}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metric_columns = (
        ("average_score_percent", "average_score", 100),
        ("mean_l4_assistance_calls", "mean_l4_assistance_calls", 1),
        ("success_without_l4_percent", "success_without_l4_rate", 100),
        ("mean_assistance_calls", "mean_assistance_calls", 1),
        ("total_assistance_calls", "total_assistance_calls", 1),
        ("total_l4_assistance_calls", "total_l4_assistance_calls", 1),
    )
    fields = ["model", "method_id", "method_label"] + [f"{name}_{stat}" for name, _, _ in metric_columns for stat in ("mean", "sd")]
    with csv_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            row = {"model": group["model"], "method_id": group["method"], "method_label": group["method_label"]}
            for name, metric, scale in metric_columns:
                for stat in ("mean", "sd"):
                    row[f"{name}_{stat}"] = group["metrics"][metric][stat] * scale
            writer.writerow(row)
    lines = [
        "# Paper metrics from complete raw trajectories", "",
        "Four-task equal weighting, then three-seed mean ± sample SD (ddof=1).", "",
        "| Model | Method | Average Score (%) | L4 calls/example | Success with no L4 (%) | All assistance calls/example | Total assistance calls/seed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        values = []
        for _, metric, scale in metric_columns[:5]:
            stats = group["metrics"][metric]
            values.append(f"{stats['mean'] * scale:.6f} ± {stats['sd'] * scale:.6f}")
        lines.append("| " + " | ".join([group["model"], group["method_label"], *values]) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, csv_path, md_path]


def main():
    args = build_parser().parse_args()
    protocol = load_protocol()
    validate_frozen_data(protocol, args.data_root)
    specs = collect_specs(args, protocol)
    expected = {}
    for task in protocol["tasks"]:
        examples = load_examples(args.data_root / task["filename"], task["key"])
        for seed in protocol["seeds"]:
            expected[(task["key"], seed)] = ({example.id for example in seed_partition(examples, "test", seed)}, task["sha256"])
    cells = []
    for spec, path in specs:
        ids, digest = expected[(spec["task"], spec["seed"])]
        cells.append(read_cell(path, spec, ids, digest))
    groups = aggregate_cells(cells)
    method_labels = {item["key"]: item["label"] for item in protocol["methods"]}
    for cell in cells:
        cell["method_label"] = method_labels[cell["method"]]
    for group in groups:
        group["metrics"]["average_score"] = group["metrics"].pop("accuracy")
        group["method_label"] = method_labels[group["method"]]
    outputs = write_reports(args.output_prefix, cells, groups)
    print(json.dumps({"cells": len(cells), "model_method_groups": len(groups), "outputs": [str(path) for path in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
