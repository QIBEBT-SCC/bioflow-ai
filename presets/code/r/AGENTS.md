# Global development guidelines for the Bioflow AI R code

## Monorepo structure

The workspace contains a single R script and its dependency declarations. BioFlowAI collects `script.R` and the
package names in `dependencies.txt` for the saved CODE node.

```text
workspace/
├── output/          # All files produced by the script, without subdirectories
├── script.R         # All workflow logic, helper functions, and classes
├── AGENTS.md        # Instructions for this workspace
└── dependencies.txt # Additional CRAN package names, one per line
```

## Development tools & commands

- `Rscript --vanilla` – Script execution in the target runtime
- Base and recommended R packages – Standard data processing and filesystem operations

### Environment and dependency management

- The saved script runs in the official `r-base:4.4.3` container. Use syntax and APIs compatible with R 4.4.3.
- Prefer base and recommended packages when practical. Declare each additional CRAN package on its own line in
  `dependencies.txt`; leave the file empty when no additional packages are needed.
- Declare package names only. Do not include version constraints, base or recommended packages, Bioconductor
  packages, GitHub repositories, URLs, local paths, installer flags, or installation commands.
- BioFlowAI installs declared dependencies before running the script. Do not install packages from `script.R`.
- The authoring `run_command` tool uses biokit-utils, which does not include R. Inspect the source and CLI contract;
  report which checks actually ran and whether execution in the target R runtime remains unverified.

## Coding Guidelines

### Single-file implementation

- Keep all executable workflow logic, helper functions, and classes in `script.R`.
- Do not create additional local R scripts or load workspace helpers with `source()`.
- Use base and recommended packages or third-party CRAN packages declared in `dependencies.txt`.
- Do not invoke Bash, Python, external programs, package installers, or network clients from the script.

### CLI and Files

- Parse `commandArgs(trailingOnly = TRUE)` and require one named option for every input.
- Use option names exactly as provided, including their leading hyphens; never rely on argument order.
- Reject unknown options, missing option values, and missing required inputs. Validate that every input exists and
  matches the expected file or directory type before reading it.
- Run from the workspace root during authoring checks and workflow execution; do not use `setwd()` to change
  into `output/`.
- Write every generated file, including final results, intermediate files, logs, and plots, directly into the
  existing `output/` directory using a deterministic plain file name. Keep the directory flat: do not create subdirectories.
- Build output paths with `file.path(getwd(), 'output', 'filename.ext')`. Do not hardcode `/workspace` paths or
  create another `output/` directory inside it.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input or failed processing.
