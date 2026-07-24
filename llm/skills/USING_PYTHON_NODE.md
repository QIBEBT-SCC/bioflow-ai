---
name: USING_PYTHON_NODE
description: Workflow-builder guidance for choosing a Python code node, preparing its bindings and request, and connecting its generated outputs
---

# Skill: Using the Python Code Node

Use a Python code node only for a bounded, deterministic adapter or file transformation that must execute inside the workflow because its output is consumed by a later node. The entire operation must fit naturally in one Python file.

The fact that an operation can be implemented in Python is not enough reason to use this node. Prefer:

- a `text_file` resource when the content is already known while designing the workflow;
- a Bash code node when standard Linux commands, `fd`, `rg`, `jq`, `yq`, or `mlr` express the transformation clearly;
- a tool node for bioinformatics command-line software;
- `downstream_summary` for statistical analysis, clustering, visualization, data mining, biological interpretation, or other systematic scientific analysis.

The strongest signal for a Python code node is that a later workflow node cannot run until this small transformation produces a file, while implementing it in Bash would be fragile or obscure.

## Inputs and Request

Every upstream file or directory must pass through its own `bind_param` processor node before connecting to the Python
node's `input_files` handle. Give each binding a descriptive long option such as `--input-json` or
`--metadata-table`; the coding model implements the matching Click options.

The workflow-building agent does not write Python. In the node manager's `request`, describe the exact transformation,
the meaning and expected type of every bound input, every deterministic output file name, and any libraries or format
rules that are genuinely required. The manager passes the actual binding options to a dedicated coding model.

## Outputs

The request must require every result to be written under a deterministic plain file name such as `normalized.tsv`.

The Python node outputs `output_folder`, not individual files. Add one `select_file_in_folder` processor for every result that another node must consume, with `file_name` exactly matching the name written by the script.

## Construction Order

1. Decide whether the operation is truly a small Python adapter rather than Bash, a tool node, or downstream analysis.
2. Determine all inputs and their descriptive Click option names.
3. Add one `bind_param` node per input.
4. Describe the exact transformation, deterministic output file names, and any required third-party libraries in the
   Python code node request. Do not write the script yourself; a dedicated coding model generates it.
5. Add the Python code node with all `bind_param` outputs connected to `input_files`, then inspect the returned code,
   including its dependencies and output file names.
6. Add `select_file_in_folder` nodes for files consumed later in the workflow.

Avoid using this node for a terminal report or plot merely because the code is short. If the output represents the scientific conclusion rather than an intermediate artifact needed by the DAG, use `downstream_summary`.
