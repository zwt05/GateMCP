# GateMCP paper-aligned code snapshot

This repository snapshot contains the paper-aligned GateMCP source code, protocol configuration, tests, and helper tools. The naming-alignment snapshot preserves the existing controller logic; this GitHub package only leaves out the four frozen benchmark data files and the long operator manual.

## Contents

- `qwen3_framework/general_mcp/`: controller, task adapters, MCP interfaces, and metrics.
- `qwen3_framework/scripts/`: single-run, paper-matrix, and report aggregation commands.
- `config/`: paper protocol and an example model-path map.
- `tests/`, `tools/`: test suite, diagnostics, and differential helpers.
- `METHODS.md`, `CHANGES.md`, `SOURCE_PROVENANCE.json`: method map, change notes, and source provenance.

## Frozen evaluation data

The four frozen JSONL data files are intentionally not included in this GitHub snapshot because this bundle does not establish redistribution rights for every dataset. Required filenames, expected counts, and SHA-256 values are recorded in `DATA_REQUIREMENTS.md`. Obtain copies under their original terms, place them under `qwen3_framework/data/full_evaluation/`, and let the launcher verify them before running.

## Setup

Use Python 3.10 or newer. Install the dependencies in an environment you manage:

```sh
python -m pip install -r requirements.txt
```

Set the model directories in `config/model_paths.example.json` for the machine where you will run the experiment. First inspect a dry-run plan:

```sh
python qwen3_framework/scripts/run_paper_matrix.py --model-key qwen3-1.7b
```

Execution requires the authorized model files and the four verified data files. The launcher requires an explicit `--execute` flag.

After the paths, data hashes, model checkpoints, and run directory have been checked, a run can be started explicitly:

```sh
python qwen3_framework/scripts/run_paper_matrix.py \
  --model-paths config/model_paths.example.json \
  --output-root results/new_reproduction \
  --execute
```

## Tests

The included standard-library test suite can be run with:

```sh
python -m unittest discover -s tests -v
```

Tests that require the frozen benchmark files skip when those files are absent. This packaging task did not run the test suite.

## License and third-party material

This snapshot does not assign a project license. Review `GITHUB_UPLOAD_NOTES_CN.md` and verify project ownership and third-party terms before publishing it as an open-source repository. Data terms remain those of the original dataset sources.
