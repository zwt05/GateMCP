from __future__ import annotations

import ast
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3_framework.general_mcp.backends import Generation, ScriptedBackend
from qwen3_framework.general_mcp.controller import GeneralMCPController, PAPER_METHODS
from qwen3_framework.general_mcp.gateway import FastMCPTaskGateway, LocalTaskGateway
from qwen3_framework.general_mcp.registry import create_adapter
from qwen3_framework.general_mcp.schema import TaskExample, write_examples
from qwen3_framework.scripts.run_general_experiment import CoTGame24Adapter, CumulativeOutputBudgetBackend, parse_args, run
from qwen3_framework.scripts.run_paper_matrix import build_cells, build_parser, load_protocol, validate_frozen_data

FIXTURES = json.loads((ROOT / "tests/fixtures/synthetic.json").read_text(encoding="utf-8"))
EXPECTED_METHODS = (
    "direct", "self_refine", "reflexion", "verifier_retry", "react_mcp", "adapted_verifiagent",
    "best_of_n", "gatemcp", "fixed", "random_order", "repeated_strong",
    "strongest_on_failure", "full_cascade", "gatemcp_no_gate",
)


def make_example(task="game24"):
    fixture = next(f for f in FIXTURES if f["task"] == task)
    return TaskExample.from_dict(fixture), fixture


def solve(method, responses, task="game24"):
    example, _ = make_example(task)
    adapter = CoTGame24Adapter() if task == "game24" else create_adapter(task)
    gateway = LocalTaskGateway(adapter, {example.id: example})

    async def work():
        async with gateway:
            return await GeneralMCPController(ScriptedBackend(responses), adapter, gateway, orchestration_seed=42).solve(example, method)

    return asyncio.run(work())


class PackageTests(unittest.TestCase):
    def test_exact_method_and_task_scope(self):
        self.assertEqual(PAPER_METHODS, EXPECTED_METHODS)
        protocol = load_protocol()
        self.assertEqual([m["key"] for m in protocol["methods"]], list(EXPECTED_METHODS))
        self.assertEqual([t["key"] for t in protocol["tasks"]], ["game24", "blocksworld", "gsm8k", "svamp"])
        self.assertEqual(len(protocol["models"]), 5)

    def test_defaults_and_nonpaper_method_rejection(self):
        for method in PAPER_METHODS:
            args = parse_args(["--task", "game24", "--data", "fixture.jsonl", "--method", method, "--model", "offline-static", "--output", "out.jsonl"])
            self.assertEqual(args.max_new_tokens, 3000 if method == "direct" else 600)
            self.assertEqual((args.num_samples, args.cumulative_output_token_budget, args.max_stage, args.temperature), (0, 3000, 4, 0))
        for method in ("a", "b", "ab", "critic", "vge_prompt_only", "vge_teacher_fallback"):
            with self.subTest(method=method), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--task", "game24", "--data", "fixture.jsonl", "--method", method, "--model", "offline-static", "--output", "out.jsonl"])

    def test_matrix_is_dry_run_and_has_264_cells(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.execute)
        cells = build_cells(args)
        self.assertEqual(len(cells), 264)
        self.assertEqual(len({(c["model"], c["task"], c["seed"], c["method"]) for c in cells}), 264)
        for cell in cells:
            command = cell["command"]
            self.assertEqual(command[command.index("--num-samples") + 1], "0")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "3000" if cell["method"] == "direct" else "600")
            self.assertEqual("--game24-cot-prompt" in command, cell["task"] == "game24")

    def test_copied_game24_core_is_byte_identical(self):
        digest = hashlib.sha256((ROOT / "qwen3_framework/general_mcp/game24_core.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "2bc10c982cf575c519c79980c54718542bd06cb20794f903e62f7764a0a9e9fa")

    def test_python_sources_parse_without_legacy_imports(self):
        for path in (ROOT / "qwen3_framework").rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        self.assertNotIn("progressive_mcp", module)
                        self.assertNotIn("capability", module)

    def test_frozen_data_when_present(self):
        data_root = ROOT / "qwen3_framework/data/full_evaluation"
        if not all((data_root / task["filename"]).is_file() for task in load_protocol()["tasks"]):
            self.skipTest("frozen data not supplied")
        self.assertEqual(len(validate_frozen_data(load_protocol(), data_root)), 4)


class AdapterTests(unittest.TestCase):
    def test_valid_and_invalid_answers(self):
        for fixture in FIXTURES:
            example = TaskExample.from_dict(fixture)
            adapter = create_adapter(example.task)
            with self.subTest(example=example.id):
                self.assertFalse(adapter.verify(example, adapter.normalize_answer(fixture["invalid_output"])).valid)
                if fixture["valid_output"] is not None:
                    self.assertTrue(adapter.verify(example, adapter.normalize_answer(fixture["valid_output"])).valid)

    def test_all_public_payloads_obey_existing_boundary(self):
        for fixture in FIXTURES:
            example = TaskExample.from_dict(fixture)
            adapter = create_adapter(example.task)
            for stage in range(1, 5):
                with self.subTest(example=example.id, stage=stage):
                    payload = adapter.assistance(example, stage)
                    self.assertFalse(adapter.leaks_complete_answer(example, payload))

    def test_game24_cot_prompt_exact(self):
        example, _ = make_example()
        self.assertEqual(CoTGame24Adapter().base_prompt(example), "Solve the Game of 24 using all four supplied numbers exactly once. Only +, -, *, /, and parentheses are allowed. Think step by step briefly, check the arithmetic, and finish with exactly one separate line in the form FINAL_EXPRESSION: <expression>.\nNumbers: [3, 3, 8, 8]")

    def test_math_preserves_reference_derived_skeleton(self):
        example, _ = make_example("gsm8k")
        payload = create_adapter("gsm8k").assistance(example, 4)
        self.assertEqual(payload["next_calculation"], "4*7")
        self.assertTrue(payload["result_withheld"])

    def test_blocksworld_prefix_prompt_remains_exact(self):
        example, fixture = make_example("blocksworld")
        adapter = create_adapter("blocksworld")
        payload = adapter.assistance(example, 4)
        prompt = adapter.repair_prompt(example, fixture["invalid_output"], adapter.verify(example, ""), 4, payload)
        self.assertIn("your FINAL_PLAN must begin exactly with (pick-up b); (stack b a).", prompt)
        self.assertIn("PREVIOUS_RESPONSE: " + fixture["invalid_output"], prompt)


class ControllerTests(unittest.TestCase):
    def test_every_method_runs_on_every_task_offline(self):
        for task in ("game24", "blocksworld", "gsm8k", "svamp"):
            _, fixture = make_example(task)
            for method in PAPER_METHODS:
                with self.subTest(task=task, method=method):
                    response = "ACTION: verified_next_step\nSELF_CHECK: FAIL\n" + fixture["invalid_output"]
                    result = solve(method, [response] * 8, task)
                    self.assertFalse(result.success)
                    self.assertGreaterEqual(len(result.attempts), 1)
                    self.assertLessEqual(len(result.attempts), 5)

    def test_vge_escalates_then_stops(self):
        _, fixture = make_example()
        result = solve("gatemcp", [fixture["invalid_output"], fixture["invalid_output"], fixture["valid_output"], fixture["invalid_output"]])
        self.assertTrue(result.success)
        self.assertEqual([c.name for c in result.tool_calls if c.assistance], ["decompose_task", "generate_candidates"])
        self.assertEqual(result.solved_stage, 2)

    def test_fixed_calls_strongest_even_on_autonomous_success(self):
        _, fixture = make_example()
        result = solve("fixed", [fixture["valid_output"]])
        self.assertTrue(result.success)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual([c.name for c in result.tool_calls if c.assistance], ["verified_next_step"])

    def test_full_cascade_scores_final_attempt_only(self):
        _, fixture = make_example()
        result = solve("full_cascade", [fixture["valid_output"]] + [fixture["invalid_output"]] * 4)
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 5)
        self.assertEqual(len([c for c in result.tool_calls if c.assistance]), 4)

    def test_nogate_does_not_use_strict_verifier_to_stop(self):
        _, fixture = make_example()
        result = solve("gatemcp_no_gate", ["SELF_CHECK: PASS\n" + fixture["invalid_output"]])
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 1)
        result = solve("gatemcp_no_gate", ["SELF_CHECK: FAIL\n" + fixture["valid_output"]] + ["SELF_CHECK: FAIL\n" + fixture["invalid_output"]] * 4)
        self.assertEqual(len(result.attempts), 5)
        self.assertFalse(result.success)
        for attempt in result.attempts[1:]:
            self.assertNotIn("VERIFIER_FEEDBACK:", attempt.prompt)

    def test_random_order_is_item_seed_deterministic(self):
        example, _ = make_example()
        adapter = create_adapter(example.task)
        controller = GeneralMCPController(ScriptedBackend([]), adapter, None, orchestration_seed=42)
        order = controller._random_stage_order(example)
        self.assertEqual(set(order), {1, 2, 3, 4})
        self.assertEqual(order, controller._random_stage_order(example))

    def test_cumulative_budget_and_reset(self):
        class Metered:
            name = "metered"
            max_new_tokens = 100
            def generate(self, prompt):
                return Generation("", 1, self.max_new_tokens)
        backend = CumulativeOutputBudgetBackend(Metered(), 1000, 600)
        self.assertEqual(backend.generate("").output_tokens, 600)
        self.assertEqual(backend.generate("").output_tokens, 400)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            backend.generate("")
        backend.reset()
        self.assertEqual(backend.remaining, 1000)

    def test_controller_exposes_existing_exhaustion_behavior(self):
        example, fixture = make_example()
        class Metered:
            name = "budget-mock"
            max_new_tokens = 600
            def generate(self, prompt):
                return Generation(fixture["invalid_output"], 1, self.max_new_tokens)
        adapter = create_adapter(example.task)
        async def work():
            async with LocalTaskGateway(adapter, {example.id: example}) as gateway:
                backend = CumulativeOutputBudgetBackend(Metered(), 19, 600)
                return await GeneralMCPController(backend, adapter, gateway).solve(example, "gatemcp")
        with self.assertRaisesRegex(RuntimeError, "Cumulative output-token budget exhausted"):
            asyncio.run(work())


class CLITests(unittest.TestCase):
    def test_existing_output_and_resume_protocol_guards(self):
        example, _ = make_example()
        with tempfile.TemporaryDirectory(prefix="gatemcp-resume-") as directory:
            directory = Path(directory)
            data = write_examples(directory / "synthetic.jsonl", [example])
            output = directory / "direct.jsonl"
            argv = ["--task", "game24", "--data", str(data), "--method", "direct", "--model", "offline-static", "--transport", "local", "--game24-cot-prompt", "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                asyncio.run(run(parse_args(argv)))
            original = output.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                asyncio.run(run(parse_args(argv)))
            self.assertEqual(output.read_text(encoding="utf-8"), original)
            changes = {
                "orchestration_seed": 43, "temperature": 1.0, "evaluation_split": "calibration",
                "transport": "mcp", "max_stage": 3, "top_k": 4, "thinking": True,
                "cumulative_output_token_budget": 2999, "per_call_output_token_cap": 600,
                "game24_cot_prompt": False, "data_sha256": "wrong", "method": "gatemcp", "student": "wrong",
            }
            for key, value in changes.items():
                with self.subTest(field=key):
                    record = json.loads(original)
                    record[key] = value
                    changed = json.dumps(record) + "\n"
                    output.write_text(changed, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        asyncio.run(run(parse_args(argv + ["--resume"])))
                    self.assertEqual(output.read_text(encoding="utf-8"), changed)
            record = json.loads(original)
            del record["orchestration_seed"]
            output.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                asyncio.run(run(parse_args(argv + ["--resume"])))
            output.write_text(original, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                asyncio.run(run(parse_args(argv + ["--resume"])))
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_all_fourteen_cli_methods_without_model_inference(self):
        example, _ = make_example()
        with tempfile.TemporaryDirectory(prefix="gatemcp-cli-") as directory:
            root = Path(directory)
            data = write_examples(root / "synthetic.jsonl", [example])
            for method in PAPER_METHODS:
                with self.subTest(method=method):
                    output = root / f"{method}.jsonl"
                    command = [sys.executable, str(ROOT / "qwen3_framework/scripts/run_general_experiment.py"), "--task", "game24", "--data", str(data), "--method", method, "--model", "offline-static", "--transport", "local", "--game24-cot-prompt", "--output", str(output)]
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    row = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(row["method"], method)
                    self.assertEqual(row["per_call_output_token_cap"], 3000 if method == "direct" else 600)


@unittest.skipUnless(os.environ.get("GATEMCP_TEST_MCP") == "1", "set GATEMCP_TEST_MCP=1 on the selected server")
class MCPTransportTests(unittest.TestCase):
    def test_real_stdio_matches_local_payloads_for_four_tasks(self):
        for task in ("game24", "blocksworld", "gsm8k", "svamp"):
            example, fixture = make_example(task)
            adapter = create_adapter(task)
            with self.subTest(task=task), tempfile.TemporaryDirectory(prefix="gatemcp-mcp-") as directory:
                data = write_examples(Path(directory) / "synthetic.jsonl", [example])
                async def work():
                    async with LocalTaskGateway(adapter, {example.id: example}) as local, FastMCPTaskGateway(data, task) as remote:
                        for name in ("verify_answer", "decompose_task", "generate_candidates", "local_search", "verified_next_step"):
                            arguments = {"example_id": example.id, "answer": adapter.normalize_answer(fixture["valid_output"])} if name == "verify_answer" else {"example_id": example.id, "top_k": 5}
                            expected = await local.call_tool(name, arguments)
                            observed = await remote.call_tool(name, arguments)
                            self.assertIsNone(observed.error)
                            self.assertEqual(observed.data, expected.data)
                asyncio.run(work())


if __name__ == "__main__":
    unittest.main()
