---
name: writing-r-code
description: Create or edit script.R and the CRAN dependency list for a BioFlowAI code_R CODE definition, preserving its CLI and R 4.4.3 runtime contract.
---

# Write an R CODE definition

## Authoring Files and Checks

- When editing an authoring workspace, deliver the script in `/workspace/script.R` and put one additional CRAN
  package name per line in `/workspace/dependencies.txt`, without installer flags or commands.
- Leave `dependencies.txt` empty when no additional dependency is needed.
- Inspect the generated source and CLI contract. Checks in biokit-utils alone do not establish compatibility with
  the target R 4.4.3 runtime and its dependencies; report which checks actually ran.

## Target Runtime

- The script runs with `Rscript --vanilla` in the official `r-base:4.4.3` container.
- Its current working directory is the writable node output directory.
- Upstream files and directories are mounted read-only and passed as named command-line options.
- Additional CRAN packages declared for the CODE definition are installed before the script runs.

## CLI and Files

- Parse `commandArgs(trailingOnly = TRUE)` and require every option in the authoritative input contract.
- Use option names exactly as provided, including their leading hyphens; do not rely on argument order.
- Validate that every input path exists before reading it.
- Write every result to the current working directory with a deterministic plain file name.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input.

## Compatibility and Dependencies

- Generate syntax and APIs compatible with R 4.4.3.
- Prefer base and recommended packages when practical.
- Declare only additional CRAN package names; do not include version constraints, base packages, Bioconductor
  packages, GitHub repositories, URLs, or local paths.
- Do not invoke Bash, Python, package installers, or network clients from the script.
