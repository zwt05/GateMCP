#!/usr/bin/env python3
"""Emit deterministic synthetic-fixture behavior for one isolated source root."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in {"latency_seconds", "total_latency_seconds"}
            and not (key in {"selected_protocol", "teacher_manifest_sha256"} and item is None)
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


async def probe(source_root):
    sys.path.insert(0, str(source_root))
    legacy_core = source_root / "A_progressive_mcp/src/progressive_mcp/core.py"
    if legacy_core.is_file():
        # Load the original core without executing its unrelated package
        # initializer. No core code or general-controller behavior is patched.
        package = types.ModuleType("progressive_mcp")
        package.__path__ = [str(legacy_core.parent)]
        sys.modules["progressive_mcp"] = package
    registry = importlib.import_module("qwen3_framework.general_mcp.registry")
    schema = importlib.import_module("qwen3_framework.general_mcp.schema")
    controller_module = importlib.import_module("qwen3_framework.general_mcp.controller")
    gateway_module = importlib.import_module("qwen3_framework.general_mcp.gateway")
    backend_module = importlib.import_module("qwen3_framework.general_mcp.backends")
    runner = importlib.import_module("qwen3_framework.scripts.run_general_experiment")
    bundle_root = Path(__file__).resolve().parents[1]
    fixtures = json.loads((bundle_root / "tests/fixtures/synthetic.json").read_text(encoding="utf-8"))
    protocol = json.loads((bundle_root / "config/paper_protocol.json").read_text(encoding="utf-8"))
    methods = [method["key"] for method in protocol["methods"]]
    output = {"helpers": {}, "controllers": {}, "budgets": {}}

    class ProbeBackend:
        name = "equivalence-scripted"
        max_new_tokens = 600
        def __init__(self, fixture, schedule):
            self.fixture = fixture
            self.schedule = schedule
            self.index = 0
            self.generations = []
        def generate(self, prompt):
            index = self.index
            self.index += 1
            success_from = int(self.schedule.rsplit("_", 1)[-1]) if self.schedule.startswith("success_from_") else None
            use_valid = (success_from is not None and index >= success_from) or (self.schedule == "valid_then_invalid" and index == 0)
            answer = self.fixture["valid_output"] if use_valid else self.fixture["invalid_output"]
            answer = answer or self.fixture["invalid_output"]
            self_pass = self.schedule == "self_pass_invalid" or (success_from is not None and index >= success_from)
            action = list(controller_module.GeneralMCPController.TOOL_NAMES.values())[index % 4]
            text = f"ACTION: {action}\nSELF_CHECK: {'PASS' if self_pass else 'FAIL'}\n{answer}"
            tokens = min(self.max_new_tokens, 19)
            self.generations.append({"prompt": prompt, "text": text, "input_tokens": 12, "output_tokens": tokens})
            return backend_module.Generation(text, 12, tokens, "equivalence-mock")

    class RecordingGateway(gateway_module.LocalTaskGateway):
        def __init__(self, adapter, examples):
            super().__init__(adapter, examples)
            self.observed_calls = []
        async def call_tool(self, name, arguments):
            response = await super().call_tool(name, arguments)
            self.observed_calls.append({"name": name, "arguments": arguments, "data": response.data, "error": response.error})
            return response

    for fixture in fixtures:
        example = schema.TaskExample.from_dict(fixture)
        adapter = registry.create_adapter(example.task)
        helper = {"base_prompt": adapter.base_prompt(example), "stages": {}}
        invalid = adapter.normalize_answer(fixture["invalid_output"])
        verification = adapter.verify(example, invalid)
        helper["invalid_verification"] = verification.to_dict()
        if fixture["valid_output"]:
            helper["valid_verification"] = adapter.verify(example, adapter.normalize_answer(fixture["valid_output"])).to_dict()
        if example.task == "game24":
            helper["cot_prompt"] = runner.CoTGame24Adapter().base_prompt(example)
        for stage in range(1, 5):
            helper["stages"][str(stage)] = {}
            for top_k in (1, 5, 7):
                payload = adapter.assistance(example, stage, top_k)
                helper["stages"][str(stage)][str(top_k)] = {
                    "payload": payload,
                    "leakage": adapter.leaks_complete_answer(example, payload),
                    "repair_prompt": adapter.repair_prompt(example, fixture["invalid_output"], verification, stage, payload),
                }
        helper["random_orders"] = {
            str(seed): controller_module.GeneralMCPController(ProbeBackend(fixture, "never"), adapter, None, orchestration_seed=seed)._random_stage_order(example)
            for seed in (42, 43, 44)
        }
        output["helpers"][example.id] = normalize(helper)

    primary_fixtures = [next(f for f in fixtures if f["task"] == task) for task in ("game24", "blocksworld", "gsm8k", "svamp")]
    schedules = ["never"] + [f"success_from_{index}" for index in range(5)] + ["self_pass_invalid", "valid_then_invalid"]
    cases = [(schedule, 3000) for schedule in schedules] + [("never", 19), ("never", 38)]
    for fixture in primary_fixtures:
        example = schema.TaskExample.from_dict(fixture)
        adapter = runner.CoTGame24Adapter() if example.task == "game24" else registry.create_adapter(example.task)
        for method in methods:
            for seed in (42, 43, 44):
                for schedule, budget in cases:
                    scripted = ProbeBackend(fixture, schedule)
                    backend = runner.CumulativeOutputBudgetBackend(scripted, budget, 3000 if method == "direct" else 600)
                    gateway = RecordingGateway(adapter, {example.id: example})
                    async with gateway:
                        controller = controller_module.GeneralMCPController(backend, adapter, gateway, top_k=5, orchestration_seed=seed)
                        try:
                            result = await controller.solve(example, method, max_stage=4)
                            observed = result.to_dict()
                        except RuntimeError as error:
                            if str(error) != "Cumulative output-token budget exhausted":
                                raise
                            observed = {
                                "budget_exhaustion": str(error),
                                "remaining_budget": backend.remaining,
                                "generations": scripted.generations,
                                "tool_calls": gateway.observed_calls,
                            }
                    output["controllers"][f"{example.id}/{method}/{seed}/{schedule}/budget_{budget}"] = normalize(observed)

    class MeteredBackend:
        name = "metered"
        max_new_tokens = 600
        def generate(self, prompt):
            return backend_module.Generation("", 1, self.max_new_tokens)
    metered = runner.CumulativeOutputBudgetBackend(MeteredBackend(), 1000, 600)
    tokens = [metered.generate("").output_tokens, metered.generate("").output_tokens]
    try:
        metered.generate("")
    except RuntimeError as error:
        output["budgets"]["exhaustion"] = str(error)
    metered.reset()
    output["budgets"].update({"tokens": tokens, "remaining_after_reset": metered.remaining})
    core = legacy_core if legacy_core.is_file() else source_root / "qwen3_framework/general_mcp/game24_core.py"
    output["core_sha256"] = hashlib.sha256(core.read_bytes()).hexdigest()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.source_root.resolve())), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
