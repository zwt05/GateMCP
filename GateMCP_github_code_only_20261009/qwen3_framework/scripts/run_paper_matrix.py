#!/usr/bin/env python3
"""Print the frozen paper matrix; execute only when --execute is explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qwen3_framework.general_mcp.controller import (
    LEGACY_METHOD_ALIASES,
    PAPER_METHODS,
    normalize_method,
)
from qwen3_framework.general_mcp.schema import load_examples
from qwen3_framework.scripts.run_general_experiment import seed_partition


def load_protocol():
    return json.loads((ROOT / "config/paper_protocol.json").read_text(encoding="utf-8"))


def build_parser():
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "qwen3_framework/data/full_evaluation")
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--model-paths", type=Path, help="JSON mapping from model key to checkpoint directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/new_reproduction")
    parser.add_argument("--model-key", choices=[m["key"] for m in protocol["models"]], action="append")
    parser.add_argument("--task", choices=[t["key"] for t in protocol["tasks"]], action="append")
    parser.add_argument(
        "--method", choices=(*PAPER_METHODS, *LEGACY_METHOD_ALIASES), action="append",
        help="Use the paper-aligned identifier; previous identifiers remain accepted",
    )
    parser.add_argument("--seed", type=int, choices=protocol["seeds"], action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run real model inference on the selected server")
    return parser


def validate_frozen_data(protocol, data_root, tasks=None, seeds=None):
    """Refuse changed, incomplete, or differently serialized test partitions."""
    checked = []
    for task in protocol["tasks"]:
        if tasks and task["key"] not in tasks:
            continue
        path = data_root / task["filename"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != task["sha256"]:
            raise ValueError(f"Frozen dataset SHA-256 mismatch: {task['key']}")
        examples = load_examples(path, task["key"])
        for seed in seeds or protocol["seeds"]:
            partition = seed_partition(examples, protocol["evaluation_split"], seed)
            if len(partition) != task["samples"] or len({e.id for e in partition}) != task["samples"]:
                raise ValueError(f"Frozen partition size/identity mismatch: {task['key']} seed {seed}")
        checked.append({"task": task["key"], "sha256": digest, "samples": task["samples"]})
    return checked


def build_cells(args):
    protocol = load_protocol()
    model_paths = {}
    if args.model_paths:
        model_paths = json.loads(args.model_paths.read_text(encoding="utf-8"))
        if not isinstance(model_paths, dict):
            raise ValueError("--model-paths must contain a JSON object")
    cells = []
    requested_methods = {normalize_method(method) for method in args.method or []}
    for model in protocol["models"]:
        if args.model_key and model["key"] not in args.model_key:
            continue
        path = Path(model_paths.get(model["key"], args.model_root / model["directory"])).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        checkpoint = path.resolve()
        allowed = list(PAPER_METHODS) if model["matrix"] == "main" else protocol["scale_methods"]
        methods = [method for method in allowed if not requested_methods or method in requested_methods]
        for task in protocol["tasks"]:
            if args.task and task["key"] not in args.task:
                continue
            for seed in args.seed or protocol["seeds"]:
                for method in methods:
                    cap = protocol["direct_per_call_output_token_cap"] if method == "direct" else protocol["assisted_per_call_output_token_cap"]
                    run_method = method
                    output = args.output_root / model["key"] / task["key"] / method / f"seed_{seed}.jsonl"
                    if args.resume and not output.is_file():
                        for legacy_method, paper_method in LEGACY_METHOD_ALIASES.items():
                            legacy_output = args.output_root / model["key"] / task["key"] / legacy_method / f"seed_{seed}.jsonl"
                            if paper_method == method and legacy_output.is_file():
                                run_method = legacy_method
                                output = legacy_output
                                break
                    command = [
                        sys.executable, str(ROOT / "qwen3_framework/scripts/run_general_experiment.py"),
                        "--task", task["key"], "--data", str((args.data_root / task["filename"]).resolve()),
                        "--method", run_method, "--model", f"local-transformers:{checkpoint}",
                        "--seed", str(seed), "--evaluation-split", "test", "--num-samples", "0",
                        "--temperature", "0", "--max-new-tokens", str(cap),
                        "--cumulative-output-token-budget", "3000", "--max-stage", "4", "--top-k", "5",
                        "--transport", "mcp", "--output", str(output.resolve()),
                    ]
                    if task["key"] == "game24":
                        command.append("--game24-cot-prompt")
                    if args.resume:
                        command.append("--resume")
                    cells.append({"model": model["key"], "checkpoint": str(checkpoint), "task": task["key"], "seed": seed, "method": run_method, "command": command})
    if not cells:
        raise ValueError("The requested filters select no paper cells")
    return cells


def main():
    args = build_parser().parse_args()
    cells = build_cells(args)
    print(json.dumps({"mode": "execute" if args.execute else "plan_only", "cells": len(cells)}, ensure_ascii=False))
    if not args.execute:
        for cell in cells:
            print(shlex.join(cell["command"]))
        return
    checks = validate_frozen_data(load_protocol(), args.data_root, args.task, args.seed)
    for checkpoint in {cell["checkpoint"] for cell in cells}:
        if not (Path(checkpoint) / "config.json").is_file():
            raise FileNotFoundError(f"Missing local checkpoint: {checkpoint}")
    print(json.dumps({"frozen_data_verified": checks}, ensure_ascii=False), flush=True)
    for cell in cells:
        subprocess.run(cell["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
