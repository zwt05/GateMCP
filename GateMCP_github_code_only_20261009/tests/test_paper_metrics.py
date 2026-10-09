from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3_framework.general_mcp.metrics import aggregate_cells, summarize_records
from qwen3_framework.scripts.aggregate_paper_results import read_cell, write_reports
from qwen3_framework.scripts.run_general_experiment import summarize


def call(name="verified_next_step", assistance=True):
    return {"name": name, "assistance": assistance}


def row(success, calls):
    return {"success": success, "tool_calls": calls, "attempts": [], "input_tokens": 0, "output_tokens": 0, "total_latency_seconds": 0, "solved_stage_name": "autonomous" if success else None}


class TrajectoryMetricTests(unittest.TestCase):
    def test_fixed_autonomous_success_still_has_zero_no_l4_success(self):
        metrics = summarize_records([row(True, [call("verify_answer", False), call()])])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["mean_l4_assistance_calls"], 1)
        self.assertEqual(metrics["success_without_l4_rate"], 0)

    def test_direct_no_l4_success_equals_accuracy(self):
        metrics = summarize_records([row(True, [call("verify_answer", False)]), row(False, [call("verify_answer", False)]), row(True, [])])
        self.assertEqual(metrics["success_without_l4_rate"], metrics["accuracy"])
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["mean_l4_assistance_calls"], 0)

    def test_repeated_strong_count_can_exceed_one(self):
        records = [row(True, [call()] * 4), row(False, [call()] * 3)]
        metrics = summarize_records(records)
        self.assertEqual(metrics["mean_l4_assistance_calls"], 3.5)
        summary = summarize(records, "game24", "mock", "repeated_strong", 42, "local")
        self.assertEqual(summary["strong_tool_call_rate"], 1)
        self.assertEqual(summary["mean_l4_assistance_calls"], 3.5)
        self.assertEqual(summary["total_assistance_calls"], 7)
        self.assertIn("stage_counts", summary)

    def test_verify_answer_is_not_assistance_and_l4_requires_true_flag(self):
        metrics = summarize_records([row(True, [call("verify_answer", True), call("verified_next_step", False), call("decompose_task", True)])])
        self.assertEqual(metrics["mean_assistance_calls"], 1)
        self.assertEqual(metrics["mean_l4_assistance_calls"], 0)
        self.assertEqual(metrics["success_without_l4_rate"], 1)

    def test_empty_and_missing_tool_calls_rejected(self):
        for records in ([], [{"success": True}], [{"success": False, "tool_calls": None}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                summarize_records(records)
        with self.assertRaises(ValueError):
            summarize_records([{"success": True, "tool_calls": [{"name": "verified_next_step"}]}])


class AggregationTests(unittest.TestCase):
    @staticmethod
    def unequal_cells():
        tasks = ("game24", "blocksworld", "gsm8k", "svamp")
        counts = (1, 3, 7, 13)
        cells = []
        for seed_index, seed in enumerate((42, 43, 44)):
            for task_index, (task, count) in enumerate(zip(tasks, counts)):
                records = [row(task_index < seed_index, [call("decompose_task")] + [call()] * (task_index + seed_index)) for _ in range(count)]
                cells.append({"model": "mock", "method": "gatemcp", "task": task, "seed": seed, **summarize_records(records)})
        return cells

    def test_four_tasks_equally_weighted_and_three_seed_sample_sd(self):
        report = aggregate_cells(self.unequal_cells())[0]
        # Per-seed macro accuracy is 0, 1/4, 1/2 despite task sizes 1/3/7/13.
        self.assertAlmostEqual(report["metrics"]["accuracy"]["mean"], 0.25)
        self.assertAlmostEqual(report["metrics"]["accuracy"]["sd"], 0.25)
        self.assertAlmostEqual(report["metrics"]["mean_l4_assistance_calls"]["mean"], 2.5)
        self.assertAlmostEqual(report["metrics"]["mean_l4_assistance_calls"]["sd"], 1)
        self.assertAlmostEqual(report["metrics"]["mean_assistance_calls"]["mean"], 3.5)
        # Absolute counts are summed (80, 104, 128), not task means times N.
        self.assertAlmostEqual(report["metrics"]["total_assistance_calls"]["mean"], 104)
        self.assertAlmostEqual(report["metrics"]["total_assistance_calls"]["sd"], 24)
        self.assertEqual(report["per_seed"]["42"]["samples"], 24)

    def test_empty_incomplete_and_duplicate_groups_rejected(self):
        complete = self.unequal_cells()
        for cells in ([], complete[:-1], complete + [complete[0]]):
            with self.subTest(count=len(cells)), self.assertRaises(ValueError):
                aggregate_cells(cells)

    def test_reports_contain_table_metrics_and_absolute_totals(self):
        cells = self.unequal_cells()
        with tempfile.TemporaryDirectory(prefix="gatemcp-report-") as directory:
            outputs = write_reports(Path(directory) / "metrics", cells, aggregate_cells(cells))
            self.assertEqual([path.suffix for path in outputs], [".json", ".csv", ".md"])
            self.assertTrue(all(path.is_file() for path in outputs))
            csv_text = outputs[1].read_text(encoding="utf-8")
            self.assertIn("mean_l4_assistance_calls_mean", csv_text)
            self.assertIn("success_without_l4_percent_mean", csv_text)
            self.assertIn("total_assistance_calls_mean", csv_text)
            original = [path.read_bytes() for path in outputs]
            with self.assertRaises(FileExistsError):
                write_reports(Path(directory) / "metrics", cells, aggregate_cells(cells))
            self.assertEqual([path.read_bytes() for path in outputs], original)

    def test_raw_cell_requires_complete_ids_and_tool_call_evidence(self):
        spec = {"model": "qwen3-1.7b", "task": "game24", "method": "direct", "seed": 42}
        def raw(example_id):
            return {"example_id": example_id, "task": "game24", "method": "direct", "data_sha256": "frozen-test-hash", "cumulative_output_token_budget": 3000, "per_call_output_token_cap": 3000, "game24_cot_prompt": True, "success": True, "tool_calls": []}
        with tempfile.TemporaryDirectory(prefix="gatemcp-raw-") as directory:
            path = Path(directory) / "cell.jsonl"
            good = [raw("a"), raw("b")]
            path.write_text("\n".join(json.dumps(item) for item in good) + "\n", encoding="utf-8")
            metrics = read_cell(path, spec, {"a", "b"}, "frozen-test-hash")
            self.assertEqual(metrics["samples"], 2)
            self.assertEqual(metrics["success_without_l4_rate"], 1)
            missing_calls = raw("a")
            del missing_calls["tool_calls"]
            for records in ([], [raw("a")], [raw("a"), raw("a")], [missing_calls, raw("b")]):
                with self.subTest(records=records):
                    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_cell(path, spec, {"a", "b"}, "frozen-test-hash")


if __name__ == "__main__":
    unittest.main()
