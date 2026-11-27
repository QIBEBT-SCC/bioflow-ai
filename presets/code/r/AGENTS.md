# Global development guidelines for the Bioflow AI R code

## Monorepo structure

The workspace contains a single R script. BioFlowAI collects `script.R` for the saved CODE node.

```text
workspace/
├── output/   # All files produced by the script, without subdirectories
├── script.R  # All workflow logic, package setup, helper functions, and classes
└── AGENTS.md # Instructions for this workspace
```

## Development tools & commands

- `Rscript --vanilla` – Script execution in the target runtime
- `bash` – Script execution and syntax checking
- `fd` and `rg` – File discovery and text search
- `jq` and `yq` – JSON and YAML processing
- `mlr` – Structured CSV, TSV, and record processing

### Environment and dependency management

- Development checks and workflow execution use the same environment, use syntax and APIs compatible with R 4.4.3.
- Use R package-management functions such as `install.packages()` for missing packages. For CRAN packages,
  use the CRAN repository URL supplied in the current turn's prompt as the explicit `repos` argument so
  installation works without interactive mirror selection. Load the installed package or fail clearly if
  installation did not succeed.
- Use `requireNamespace('package', quietly = TRUE)` to check availability. `require()` and `library()` load
  installed packages; they do not install missing packages.
- Check syntax with `Rscript --vanilla -e 'invisible(parse(file = "script.R"))'`.
- Run functional checks from the workspace root with `Rscript --vanilla script.R ...` when suitable inputs and
  dependencies are available. Report which checks actually ran; syntax checks alone do not verify results.

## Coding Guidelines

### Single-file implementation

- Keep all executable workflow logic, helper functions, and classes in `script.R`.
- Do not create additional local R scripts or load workspace helpers with `source()`.
- Use R APIs for data processing and package management. Do not invoke Bash, Python, or external data-processing
  programs from the script.

### CLI and Files

- Parse `commandArgs(trailingOnly = TRUE)` and require one named option for every input.
- Use option names exactly as provided, including their leading hyphens; never rely on argument order.
- Reject unknown options, missing option values, and missing required inputs. For file or directory path inputs (`input_files`), validate that every path exists and matches the expected file or directory type before reading it. Treat text value inputs (`input_values`) as literal strings rather than filesystem paths, and do not check them with `file.exists()` or `dir.exists()`.
- Run from the workspace root during authoring checks and workflow execution.
- Write every generated file, including final results, intermediate files, and plots, directly into the
  existing `output/` directory using a deterministic plain file name. Keep the directory flat: do not create subdirectories.
- Build output paths with `file.path(getwd(), 'output', 'filename.ext')`. Do not hardcode `/workspace` paths or
  create another `output/` directory inside it.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input or failed processing.
