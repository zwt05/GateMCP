# Packaging changes and verification boundary

## Baseline

The selected paper is the 2026-10-08 Chinese revision named in
`config/paper_protocol.json`. Original source/evidence hashes are recorded in
`SOURCE_PROVENANCE.json`. Original implementation files were not edited while
building this independent package.

This name-aligned delivery is derived from the 2026-10-09 package-with-manual
delivery. It changes method identifiers and report labels only. Controller
branches, prompts, budgets, ordering, scoring, data, and experimental behavior
are unchanged. The former identifiers remain aliases for old commands and
result files.

## Changes made

1. Kept only the paper's fourteen controller methods. Removed B/A+B capability
   selection, aliases, CRITIC, prompt-only variations, and prior-fallback paths.
2. Removed prior loading, construction/abstraction helpers, registry parameters,
   gateway environment variables, and server-side prior paths. Replaced empty-
   prior ranking lookups with the equivalent fixed defaults. For Game24 the
   deleted operator-ranking component was the constant zero; stable sort order
   is otherwise unchanged.
3. Kept the default base/repair prompts and method-specific prompts unchanged.
   The original default repair body is now the sole repair implementation.
   Blocksworld's exact-prefix insertion remains in place.
4. Copied `A_progressive_mcp/src/progressive_mcp/core.py` byte-for-byte to
   `qwen3_framework/general_mcp/game24_core.py`. The adapter imports this file
   directly, avoiding the legacy package initializer and its dependencies.
5. Removed the unused API/LiteLLM backend and API credential CLI flags. The
   Transformers model-loading/rendering/generation path is unchanged. Scripted
   and offline-static backends are diagnostic-only; the latter exposes a token
   cap so it works with the existing cumulative-budget wrapper.
6. Removed obsolete null capability/prior fields from result schemas and zero
   capability-profile totals from summaries. Added explicit protocol fields to
   each new result row and data SHA-256 to summaries. Resume checks these fields
   and rejects mixed protocols; old raw results are not silently resumed into
   new-schema outputs. Existing outputs require explicit `--resume`.
7. Set paper defaults (`num_samples=0`, budget 3000, Direct cap 3000, assisted
   cap 600) and added a dry-run-first matrix launcher. The official launcher
   checks exact dataset SHA-256/counts before execution, restricts the model
   matrix, and supplies the required Game24 CoT flag.
8. Added standard-library unit/CLI checks and an optional real FastMCP stdio
   comparison. Added a subprocess-isolated old/new differential probe covering
   helper outputs and 1680 scripted controller trajectories. No real model
   inference is part of these tests.

## Intentionally unchanged

- Exact Game24 arithmetic, search, answer parsing and leakage check.
- All four task verifiers and default assistance payload semantics.
- Fixed's extra strongest-tool invocation on autonomous successes and payload
  reuse for its two guided repairs.
- Full-Cascade's unconditional stages and final-attempt scoring.
- NoGate's prompt/stopping/scoring compound change.
- Math L3/L4 calculation skeletons derived from private references.
- Best-of-N's temperature-zero, limited-diversity candidate protocol.
- The original use of per-item orchestration seed, not an added model RNG seed.
- Historical results, source manifests, audits, and paper numbers.

## Verification interpretation

The local packaging work did not run tests, start services, train, or infer.
Server verification is a separate step, recorded in a separate release receipt.
An equal differential digest means the covered default helper/controller paths
agree on the synthetic fixtures after excluding time measurements and removed
null legacy metadata. It does not certify arbitrary inputs, model checkpoints,
hardware kernels, all historical code roots, or numerical reproduction.

The final 1680 controller trajectories cover seeds42/43/44, budgets3000/19/38,
and at most19 mock output tokens per call, clipped to the remaining budget.
The standalone budget-wrapper test also exercises exhaustion/reset. This is
finite fixture coverage, not proof over all controller paths. Separate unit
and CLI tests cover summaries, all14 command-line methods, and resume guards;
the differential probe itself does not compare the complete runner.

The final package adds whole-trajectory metrics and a raw-JSONL aggregator:
mean L4 calls, final success with no L4 call anywhere in the trajectory,
all-assistance call means, and absolute call totals. Aggregation uses equal
task weights within each seed, then a mean and sample SD across three seeds.
It rejects empty, partial, duplicated, or protocol-mismatched cells and
refuses to overwrite existing reports. Optional missing legacy protocol
fields are disclosed; the caller must still audit historical model/seed
identity against source manifests. `strong_tool_call_rate` remains a per-item
indicator, not mean L4 call count. Added tests cover these metrics, CLI method
coverage, resume guards, and the inherited controller budget-exhaustion error.

## Original-workspace archive guidance

No original file is moved/deleted by this package. Cleanup should use a dated,
recoverable archive with a move manifest and hashes, after verification.

High-confidence non-paper code candidates (relative to the original root):

- `qwen3_framework/general_mcp/capability.py`.
- `qwen3_framework/teacher_mcp.py`.
- Teacher/prior construction and launcher scripts under
  `qwen3_framework/scripts/` (explicitly listed and checked by the cleanup
  coordinator; never use a broad deletion glob).
- `qwen3_framework/tests/test_teacher_mcp.py`, after the original test suite is
  preserved with its associated legacy source.
- `A_progressive_mcp/src`, `A_progressive_mcp/scripts`, and
  `A_progressive_mcp/tests`, once the original core hash is preserved.
- `B_adaptive_mcp/src`, `B_adaptive_mcp/scripts`, and
  `B_adaptive_mcp/tests`, if they are present and there are no separate active
  consumers. This is an archive candidate, not a deletion command.

The replaced original general controller/adapters/runner should also remain in
the source archive for traceability. Do not archive/delete whole A/B directories
blindly: they contain data/outputs. Preserve original `results`, source
manifests, audit reports, frozen data, paper sources, and shared project memory.
