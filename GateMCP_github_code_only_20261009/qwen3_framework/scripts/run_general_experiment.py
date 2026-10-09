#!/usr/bin/env python3
"""Run one task/model/seed/method cell of the general MCP benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qwen3_framework.general_mcp.backends import build_backend
from qwen3_framework.general_mcp.adapters.game24 import Game24Adapter
from qwen3_framework.general_mcp.controller import (
    GeneralMCPController,
    LEGACY_METHOD_ALIASES,
    PAPER_METHODS,
    normalize_method,
)
from qwen3_framework.general_mcp.gateway import FastMCPTaskGateway, LocalTaskGateway
from qwen3_framework.general_mcp.metrics import summarize_records
from qwen3_framework.general_mcp.registry import create_adapter
from qwen3_framework.general_mcp.schema import load_examples


class CoTGame24Adapter(Game24Adapter):
    """Game24 prompt used by the cumulative-token diagnostics."""

    def base_prompt(self, example):
        return (
            "Solve the Game of 24 using all four supplied numbers exactly once. "
            "Only +, -, *, /, and parentheses are allowed. Think step by step "
            "briefly, check the arithmetic, and finish with exactly one separate "
            "line in the form FINAL_EXPRESSION: <expression>.\n"
            f"Numbers: {self._numbers(example)}"
        )


class CumulativeOutputBudgetBackend:
    """Share one generated-token budget across all calls for an example."""

    def __init__(self, inner, total_budget: int, per_call_cap: int):
        if total_budget <= 0 or per_call_cap <= 0:
            raise ValueError("Cumulative and per-call token budgets must be positive")
        if not hasattr(inner, "max_new_tokens"):
            raise TypeError("The selected backend does not expose max_new_tokens")
        self.inner = inner
        self.total_budget = total_budget
        self.per_call_cap = per_call_cap
        self.remaining = total_budget
        self.name = inner.name + f":cumulative-output-{total_budget}-cap-{per_call_cap}"

    def reset(self) -> None:
        self.remaining = self.total_budget

    def generate(self, prompt: str):
        if self.remaining <= 0:
            raise RuntimeError("Cumulative output-token budget exhausted")
        self.inner.max_new_tokens = min(self.per_call_cap, self.remaining)
        generation = self.inner.generate(prompt)
        self.remaining -= int(generation.output_tokens or 0)
        return generation


def seed_partition(examples, split: str, seed: int):
    return [
        item
        for item in examples
        if item.split == split and int(item.metadata.get("seed", seed)) == seed
    ]


def summarize(results, task: str, model: str, method: str, seed: int, transport: str):
    paper_metrics = summarize_records(results)
    total = max(len(results), 1)
    assistance = [
        call
        for result in results
        for call in result["tool_calls"]
        if bool(call["assistance"])
    ]
    verifier = [
        call
        for result in results
        for call in result["tool_calls"]
        if not bool(call["assistance"])
    ]
    strong = [
        call for call in assistance if call["name"] == "verified_next_step"
    ]
    stage_counts: dict[str, int] = {}
    for result in results:
        key = str(result.get("solved_stage_name") or "unsolved")
        stage_counts[key] = stage_counts.get(key, 0) + 1
    return {
        "task": task,
        "model": model,
        "method": method,
        "seed": seed,
        "transport": transport,
        "samples": len(results),
        "accuracy": sum(bool(item["success"]) for item in results) / total,
        "autonomous_solve_rate": sum(
            item.get("solved_stage_name") == "autonomous" for item in results
        )
        / total,
        "complete_answer_tool_dependency_rate": sum(
            any(bool(call.get("leakage_detected")) for call in item["tool_calls"])
            for item in results
        )
        / total,
        "strong_tool_call_rate": sum(
            any(
                call["name"] == "verified_next_step" and bool(call["assistance"])
                for call in item["tool_calls"]
            )
            for item in results
        )
        / total,
        "mean_assistance_calls": len(assistance) / total,
        "mean_verifier_calls": len(verifier) / total,
        "mean_model_calls": (
            sum(len(item["attempts"]) for item in results)
        )
        / total,
        "mean_input_tokens": (
            sum(int(item["input_tokens"]) for item in results)
        )
        / total,
        "mean_output_tokens": (
            sum(int(item["output_tokens"]) for item in results)
        )
        / total,
        "mean_latency_seconds": (
            sum(float(item["total_latency_seconds"]) for item in results)
        )
        / total,
        "stage_counts": stage_counts,
        "mean_l4_assistance_calls": paper_metrics["mean_l4_assistance_calls"],
        "success_without_l4_rate": paper_metrics["success_without_l4_rate"],
        "total_assistance_calls": paper_metrics["total_assistance_calls"],
        "total_l4_assistance_calls": paper_metrics["total_l4_assistance_calls"],
    }


async def run(args) -> None:
    data_path = args.data.expanduser().resolve()
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    all_examples = load_examples(data_path, args.task)
    test = seed_partition(all_examples, args.evaluation_split, args.seed)
    if args.num_samples > 0:
        test = test[: args.num_samples]
    if not test:
        raise ValueError(
            f"No {args.evaluation_split} examples found for "
            f"task={args.task} seed={args.seed}"
        )
    adapter = create_adapter(args.task)
    if args.game24_cot_prompt:
        if args.task != "game24":
            raise ValueError("--game24-cot-prompt is only valid for Game24")
        adapter = CoTGame24Adapter()
    student = build_backend(
        args.model,
        args.temperature,
        args.max_new_tokens,
        args.thinking,
    )
    if args.cumulative_output_token_budget is not None:
        student = CumulativeOutputBudgetBackend(
            student,
            args.cumulative_output_token_budget,
            args.max_new_tokens,
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        raise FileExistsError("Output exists; choose a new path or explicitly use --resume")
    results = []
    record_method = args.method
    if args.resume and output_path.is_file():
        results = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing_methods = {item.get("method") for item in results}
        if len(existing_methods) > 1:
            raise ValueError("Resume output mixes method identifiers")
        if existing_methods:
            record_method = str(next(iter(existing_methods)))
        for index, item in enumerate(results):
            if index >= len(test) or item["example_id"] != test[index].id:
                raise ValueError("Resume output does not match the frozen test partition")
            if item.get("data_sha256") != data_sha256:
                raise ValueError("Resume output used a different frozen task dataset")
            if normalize_method(str(item.get("method", ""))) != normalize_method(args.method):
                raise ValueError("Resume output used a different method")
            if item.get("student") != student.name:
                raise ValueError("Resume output used a different student model")
            expected = {
                "orchestration_seed": args.seed,
                "temperature": args.temperature,
                "evaluation_split": args.evaluation_split,
                "transport": args.transport,
                "max_stage": args.max_stage,
                "top_k": args.top_k,
                "thinking": bool(args.thinking),
                "cumulative_output_token_budget": args.cumulative_output_token_budget,
                "per_call_output_token_cap": args.max_new_tokens,
                "game24_cot_prompt": bool(args.game24_cot_prompt),
            }
            for key, value in expected.items():
                if item.get(key) != value:
                    raise ValueError(f"Resume output has a different {key}")

    examples_by_id = {item.id: item for item in all_examples}
    gateway = (
        FastMCPTaskGateway(data_path, args.task)
        if args.transport == "mcp"
        else LocalTaskGateway(adapter, examples_by_id)
    )
    async with gateway:
        controller = GeneralMCPController(
            student,
            adapter,
            gateway,
            top_k=args.top_k,
            orchestration_seed=args.seed,
        )
        mode = "a" if results else "w"
        with output_path.open(mode, encoding="utf-8") as sink:
            for index, example in enumerate(test[len(results) :], len(results) + 1):
                if isinstance(student, CumulativeOutputBudgetBackend):
                    student.reset()
                result = await controller.solve(
                    example,
                    record_method,
                    max_stage=args.max_stage,
                )
                payload = result.to_dict()
                payload["data_sha256"] = data_sha256
                payload["cumulative_output_token_budget"] = args.cumulative_output_token_budget
                payload["per_call_output_token_cap"] = args.max_new_tokens
                payload["game24_cot_prompt"] = bool(args.game24_cot_prompt)
                payload["orchestration_seed"] = args.seed
                payload["temperature"] = args.temperature
                payload["evaluation_split"] = args.evaluation_split
                payload["transport"] = args.transport
                payload["max_stage"] = args.max_stage
                payload["top_k"] = args.top_k
                payload["thinking"] = bool(args.thinking)
                results.append(payload)
                sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"[{index}/{len(test)}] {example.id} success={result.success} "
                    f"stage={result.solved_stage_name}",
                    flush=True,
                )

    summary = summarize(
        results,
        args.task,
        student.name,
        record_method,
        args.seed,
        args.transport,
    )
    summary["data_sha256"] = data_sha256
    summary["evaluation_split"] = args.evaluation_split
    summary["cumulative_output_token_budget"] = args.cumulative_output_token_budget
    summary["per_call_output_token_cap"] = args.max_new_tokens
    summary["game24_cot_prompt"] = bool(args.game24_cot_prompt)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["game24", "blocksworld", "gsm8k", "svamp"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--method", choices=(*PAPER_METHODS, *LEGACY_METHOD_ALIASES), required=True,
        help="Use the paper-aligned identifier; previous identifiers remain accepted for existing outputs",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--evaluation-split", choices=["test"], default="test")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-stage", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transport", choices=["local", "mcp"], default="mcp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="Defaults to 3000 for Direct and 600 for every assisted method",
    )
    parser.add_argument(
        "--cumulative-output-token-budget", type=int, default=3000,
        help="Shared generated-token budget reset for every evaluation example",
    )
    parser.add_argument(
        "--game24-cot-prompt", action="store_true",
        help="Use the exact Game24 prompt required by the paper protocol",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = 3000 if args.method == "direct" else 600
    if args.num_samples < 0:
        parser.error("--num-samples must be non-negative")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
