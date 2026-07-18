---
name: USING_PYTHON_NODE
description: Explains when and how to use the executable Python Code Node for a small deterministic file transformation inside the workflow DAG, including its Python 3.13 and Click environment, dependency declarations, named input parameters, and output handling. Consult this skill whenever a workflow step appears to require custom Python, so it is not confused with Bash, a bioinformatics tool node, or downstream scientific analysis.
---

# Skill: Using the Python Code Node

Use a Python code node only for a bounded, deterministic adapter or file transformation that must execute inside the workflow because its output is consumed by a later node. The entire operation must fit naturally in one Python file.

The fact that an operation can be implemented in Python is not enough reason to use this node. Prefer:

- a `text_file` resource when the content is already known while designing the workflow;
- a Bash code node when standard Linux commands, `fd`, `rg`, `jq`, `yq`, or `mlr` express the transformation clearly;
- a tool node for bioinformatics command-line software;
- `downstream_summary` for statistical analysis, clustering, visualization, data mining, biological interpretation, or other systematic scientific analysis.

The strongest signal for a Python code node is that a later workflow node cannot run until this small transformation produces a file, while implementing it in Bash would be fragile or obscure.

## Runtime Contract

The node runs a single script with Python 3.13 through uv. The runtime automatically provides Click for CLI parsing. Do not add Click to `dependencies`, write `pyproject.toml`, add PEP 723 metadata, create a virtual environment, or run `pip`, `uv add`, or `uv install`; the backend constructs and resolves the script environment.

Declare only additional third-party Python distribution requirements in the node's `dependencies` list, for example:

```json
["pandas>=2.2,<3", "PyYAML>=6,<7"]
```

Use distribution names rather than import names (`PyYAML`, not `yaml`; `scikit-learn`, not `sklearn`). Prefer the standard library when it keeps the script simple. Do not use URL, Git, local-path, or runtime-installed dependencies.

## Inputs and CLI

Every upstream file or directory must pass through its own `bind_param` processor node before connecting to the Python node's `input_files` handle. Give each binding a descriptive long option such as `--input-json` or `--metadata-table`, and implement the matching option with Click.

Use this structure:

```python
from pathlib import Path

import click


@click.command()
@click.option(
    '--input-json',
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def main(input_json: Path) -> None:
    """Transform one upstream result into the required intermediate file."""
    # Read input_json and perform the bounded transformation.
    Path('normalized.tsv').write_text('...', encoding='utf-8')


if __name__ == '__main__':
    main()
```

Keep option names identical across `bind_param.parameter`, the Click decorator, and the script variable. Validate paths with `click.Path`, use `pathlib.Path` for filesystem operations, and provide explicit types for other Click options when needed.

## Outputs

The current working directory is the node's writable output directory. Write every result there with a deterministic relative name such as `normalized.tsv`. Do not modify upstream inputs or write beside them.

The Python node outputs `output_folder`, not individual files. Add one `select_file_in_folder` processor for every result that another node must consume, with `file_name` exactly matching the name written by the script.

## Construction Order

1. Decide whether the operation is truly a small Python adapter rather than Bash, a tool node, or downstream analysis.
2. Determine all inputs and their descriptive Click option names.
3. Add one `bind_param` node per input.
4. Write the complete Click-based Python 3.13 script and list only its additional dependencies.
5. Add the Python code node with all `bind_param` outputs connected to `input_files`.
6. Add `select_file_in_folder` nodes for files consumed later in the workflow.

Avoid using this node for a terminal report or plot merely because the code is short. If the output represents the scientific conclusion rather than an intermediate artifact needed by the DAG, use `downstream_summary`.
