# Global development guidelines for the Bioflow AI Bash code

## Monorepo structure

The workspace contains a single Bash script. BioFlowAI collects `script.sh` for the saved CODE node.

```text
workspace/
├── output/          # All files produced by the script, without subdirectories
├── script.sh        # All workflow logic and helper functions
└── AGENTS.md        # Instructions for this workspace
```

## Development tools & commands

- `bash` – Script execution and syntax checking
- `fd` and `rg` – File discovery and text search
- `jq` and `yq` – JSON and YAML processing
- `mlr` – Structured CSV, TSV, and record processing

### Environment and dependency management

- Development checks and workflow execution use the same linux environment. Use standard Linux commands and the additional tools listed above.
- Do not invoke Python, R, bioinformatics programs, or network clients from the script.
- If the task requires unavailable software, explain the limitation before editing. Do not reimplement a requested
  bioinformatics tool in Bash.
- Run `bash -n script.sh` to check syntax. Run functional checks from the workspace root with `bash script.sh ...`
  when suitable inputs are available. Report which checks actually ran; a syntax check alone does not verify results.

## Coding Guidelines

### Single-file implementation

- Keep all executable workflow logic and helper functions in `script.sh`.
- Do not create additional local scripts or load helper files with `source` or `.`.
- Start with `#!/usr/bin/env bash` and `set -euo pipefail`.

### CLI and Files

- Parse inputs with `getopts` and require one short option for every input.
- Use option names exactly as provided, including the leading `-`; never rely on argument order.
- Treat file or directory path inputs (`input_files`) as mounted paths; treat text value inputs (`input_values`) as literal strings rather than filesystem paths.
- Quote paths and variable expansions so inputs containing spaces or shell metacharacters are handled correctly.
- Write every generated file, including final results, intermediate files, and logs, directly into the existing
  `output/` directory using a deterministic plain file name. Keep the directory flat: do not create subdirectories.
- Build output paths with `"$PWD/output/filename.ext"`. Do not hardcode `/workspace` paths or create another
  `output/` directory inside it.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input or failed processing. Redirect data results to
  named output files instead of writing large data to stdout.
