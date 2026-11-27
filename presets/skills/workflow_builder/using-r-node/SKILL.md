---
name: using-r-node
description: Workflow-builder guidance for choosing an R code node, preparing its bindings and request, and connecting its generated outputs
---

# Skill: Using the R Code Node

Use an R code node for a bounded, deterministic file transformation whose output is required by a later workflow node and is naturally expressed in base R. The node runs one script with R 4.4.3 in a fixed container.

Prefer a tool node for bioinformatics command-line software and `downstream_summary` for terminal statistical analysis, clustering, visualization, biological interpretation, or other systematic scientific analysis. When additional R packages are needed, the script itself checks availability and installs missing packages.

## Inputs and Request

Every upstream input must pass through its own `bind_param` processor before connecting to the R node:

- Connect file or directory inputs to `input_files`.
- Connect dynamic text value inputs (`value_string`, `read_file_content`, or a subgraph boundary input carrying text) to
  `input_values`.

Use a distinct named option such as `--counts`, `--metadata`, or `--organism` for each input; option names must be
unique across both handles. Use `input_values` only for dynamic values such as `sample:<key>` metadata, upstream
`read_file_content` values, or subgraph-exposed parameters; keep fixed constants directly in the R script.

In the node manager's `request`, describe the exact transformation, the meaning and expected type of every input
(including whether each parameter is a path or a text value), every deterministic output file name, and any additional
CRAN packages. Do not write the script yourself; the dedicated coding model generates R 4.4.3-compatible code that
parses `commandArgs(trailingOnly = TRUE)`.

## Outputs

The script runs from the workspace root and writes every result directly into its `output/` directory using a
deterministic plain file name. Keep this directory flat.

The R node's `output_folder` refers to that output directory. Add one `select_file_in_folder` processor per result
consumed by another node, with `file_name` exactly matching the plain file name written by the script, without an
`output/` prefix.

## Construction Order

1. Confirm the task is a small R transformation rather than a tool invocation or terminal downstream analysis.
2. Add one `bind_param` node per upstream input and assign distinct named options across all bindings.
3. Add the R code node with file/directory bindings connected to `input_files` and text bindings connected to `input_values`.
4. Inspect the returned source for R 4.4.3 compatibility, package setup, exact option parsing, and output file names.
5. Add `select_file_in_folder` nodes for every result required later in the workflow.

Do not request a separate dependency list or installer script. Package checks and any required installation belong
in `script.R`; the platform executes that script directly.
