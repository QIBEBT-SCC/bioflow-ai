---
name: using-r-node
description: Workflow-builder guidance for choosing an R code node, preparing its bindings and request, and connecting its generated outputs
---

# Skill: Using the R Code Node

Use an R code node for a bounded, deterministic file transformation whose output is required by a later workflow node and is naturally expressed in base R. The node runs one script with R 4.4.3 in a fixed container.

Prefer a tool node for bioinformatics command-line software and `downstream_summary` for terminal statistical analysis, clustering, visualization, biological interpretation, or other systematic scientific analysis. The R code node can install declared CRAN packages from the repository.

## Inputs and Request

Every upstream file or directory must pass through its own `bind_param` processor before connecting to the R node's `input_files` handle. Use a distinct named option such as `--counts` or `--metadata` for each input.

In the node manager's `request`, describe the exact transformation, the meaning and expected type of every input, every deterministic output file name, and any additional CRAN packages. Do not write the script yourself; the dedicated coding model generates R 4.4.3-compatible code that parses `commandArgs(trailingOnly = TRUE)`.

## Outputs

The script writes results into its current working directory. The R node exposes that directory through `output_folder`. Add one `select_file_in_folder` processor per result consumed by another node, with `file_name` exactly matching the script's deterministic output name.

## Construction Order

1. Confirm the task is a small R transformation rather than a tool invocation or terminal downstream analysis.
2. Add one `bind_param` node per upstream input and assign distinct named options.
3. Add the R code node with all binding outputs connected to `input_files`.
4. Inspect the returned source and dependency list for R 4.4.3 compatibility, exact option parsing, and output file names.
5. Add `select_file_in_folder` nodes for every result required later in the workflow.

Declare CRAN dependencies by package name only. Do not request version constraints, Bioconductor packages, GitHub repositories, URLs, local packages, or installation commands inside the user script.
