# The fourteen paper methods

The canonical CLI identifiers are the names in the first column. Previous
identifiers remain accepted as compatibility aliases. No A/B alias,
capability-profiling, teacher-prior, CRITIC, or prompt-only experimental path
is exposed. External methods are task-specific implementations/adaptations,
not claims of byte-identical upstream author code.

| CLI identifier | Paper label | Legacy alias | Group | Exact retained behavior |
|---|---|---|---|---|
| `direct` | Direct | — | external | One autonomous call; strict verification. |
| `self_refine` | Self-Refine | — | external | Initial answer, then up to two intrinsic critique/revision rounds; stop after a verified answer. |
| `reflexion` | Reflexion | — | external | Initial answer, then up to two item-local reflection/revision rounds using verifier feedback. |
| `verifier_retry` | Verifier-Retry | — | external | Initial answer plus up to four verifier-feedback repairs; no assistance payload. |
| `react_mcp` | ReAct-MCP | `react` | external | Model chooses an ACTION from the four assistance tools; at most four follow-up turns. |
| `adapted_verifiagent` | Adapted VerifiAgent | `verifiagent` | external | Up to two meta-verification / strongest-partial-aid / revision rounds after the initial answer. |
| `best_of_n` | Best-of-N | `best_of_n_verifier` | external | At most five candidates; first uses the base prompt and later candidates a fixed independent-candidate prompt. |
| `gatemcp` | GateMCP | `vge` | proposed | Verify the autonomous attempt; escalate L1 to L4 only after failures; stop at first success. |
| `fixed` | Fixed | — | routing_control | Call L4 for every item, including autonomous successes; at most two guided repairs reuse that payload. |
| `random_order` | Random-Order | — | routing_control | Per-item seed-derived permutation of L1-L4; verify and stop at first success. |
| `repeated_strong` | Repeated-Strong | — | routing_control | After autonomous failure, call L4 again on each of at most four repairs; early stop. |
| `strongest_on_failure` | Strongest-on-Failure | — | routing_control | After autonomous failure, call L4 once for one guided repair. |
| `full_cascade` | Full-Cascade | — | routing_control | Run L1-L4 even after intermediate success; score the final attempt only. |
| `gatemcp_no_gate` | GateMCP-NoGate | `vge_no_verifier` | composite_control | Use SELF_CHECK to stop/escalate; strict verifier is hidden from routing/prompts and scores the final attempt. |

L1–L4 are `decompose_task`, `generate_candidates`, `local_search`, and
`verified_next_step`; `verify_answer` is counted separately.

This packaging preserves the original scientific design, including the
NoGate compound change (prompt, stopping and final scoring), limited
Best-of-N diversity at temperature zero, Fixed's payload reuse, and the
reference-derived math skeletons. It does not alter these limitations to
improve results. Repeated-Strong's unfavorable-to-GateMCP comparison remains
part of the reported paper evidence.
