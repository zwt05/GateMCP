# Frozen paper datasets

The reproduction launcher requires the exact four JSONL files under
`qwen3_framework/data/full_evaluation/`. It verifies SHA-256 and the test
partition count for every requested seed before launching inference. Do not
replace them with the older `data/general/` partitions or regenerate them.

| Task | Test items per seed | SHA-256 |
|---|---:|---|
| game24 | 660 | `fce419bda035bc8bc8e31bdf1f8ae851d3caa724524a6ff6c8769275a02f1c36` |
| blocksworld | 501 | `b08d515939bdc570e25283f34c4a73c011e22c8c0dbc35c92291c83067bab8b9` |
| gsm8k | 1319 | `4d63e8cbb8a95a864615be4e3de75d692bbeb639ec40c1f537eb661d95ab15e3` |
| svamp | 1000 | `e4dd563955596a5c09d1b09d47c519c71ea1e2157db1db9975530a1047beb305` |

The three archived seed42/43/44 source manifests agree on these four hashes.
The hashes identify the exact byte streams, including order and line endings.
A different serialization is not an acceptable frozen dataset substitute.

Each line uses the existing TaskExample schema: id, task, input, reference,
split and metadata. Reference data are private to task-semantic verifiers and
internal assistance construction. In GSM8K/SVAMP, L3/L4 deliberately use
reference-derived calculation skeletons, with numeric results withheld:
this is the paper's oracle-assisted controlled setting, not a newly corrected
or reference-free system.

Dataset origin and redistribution terms remain those of the original tasks:
Game24, PlanBench Blocksworld, GSM8K and SVAMP. The code package does not assign
a new license to those data. Any recovery receipt belongs in the separate
release evidence, not in the dataset JSONL itself.
