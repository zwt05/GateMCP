#!/usr/bin/env python3
"""Compare paper-only behavior with a read-only legacy source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_probe(source_root):
    command = [sys.executable, str(ROOT / "tools/probe_implementation.py"), "--source-root", str(source_root)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300)
    if result.returncode:
        raise RuntimeError(f"Probe failed for {source_root}:\n{result.stderr}")
    return json.loads(result.stdout)


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def differences(left, right, path="", limit=20):
    if left == right:
        return []
    found = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            current = f"{path}/{key}"
            if key not in left or key not in right:
                found.append(current + " (missing key)")
            else:
                found.extend(differences(left[key], right[key], current, limit - len(found)))
            if len(found) >= limit:
                break
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append(path + " (length differs)")
        for index, (a, b) in enumerate(zip(left, right)):
            found.extend(differences(a, b, f"{path}/{index}", limit - len(found)))
            if len(found) >= limit:
                break
    else:
        found.append(path + " (value differs)")
    return found[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--keep-probes", action="store_true")
    args = parser.parse_args()
    baseline = run_probe(args.legacy_root.resolve())
    candidate = run_probe(args.candidate_root.resolve())
    report = {
        "status": "PASS" if baseline == candidate else "FAIL",
        "scope": "current legacy default implementation versus paper-only packaging; no real model inference",
        "helper_fixtures": len(candidate["helpers"]),
        "controller_trajectories": len(candidate["controllers"]),
        "methods": 14,
        "tasks": 4,
        "controller_seeds": [42, 43, 44],
        "controller_budget_cases": [3000, 19, 38],
        "mock_output_tokens_per_call": "at most 19; clipped to the remaining budget",
        "excluded_fields": ["latency_seconds", "total_latency_seconds", "selected_protocol (only null)", "teacher_manifest_sha256 (only null)"],
        "baseline_sha256": digest(baseline),
        "candidate_sha256": digest(candidate),
        "first_difference_paths": differences(baseline, candidate),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.keep_probes:
        for label, data in (("baseline", baseline), ("candidate", candidate)):
            args.report.with_name(args.report.stem + f".{label}.json").write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if baseline != candidate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
