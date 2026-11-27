---
name: using-bash-node
description: Workflow-builder guidance for choosing a Bash node, preparing its bindings and request, and connecting its generated outputs
---

# Skill: Using the Bash node

Use the simplest node that matches the operation. For a small file whose name and text are already known while building the workflow, use a `text_file` resource node. Use a Bash code node only when an output must be computed at runtime, such as selecting, filtering, reshaping, renaming, or combining upstream files.

When Bash is required, the usual pattern uses a Bash code node together with the `bind_param` and `select_file_in_folder` processor nodes.

## Choosing between Text File and Bash

Use a `text_file` resource for static text such as:

- tool configuration files;
- sample or accession lists;
- small TSV, CSV, BED, JSON, or YAML inputs;
- any other text that can be written verbatim when the workflow is designed.

Set `file_name` to a plain file name such as `samples.txt` and set `content` to the exact file contents. The node writes UTF-8 text and directly outputs `file_path`, so connect that output to the downstream tool input. Do not create a bash node merely to run `echo`, `printf`, or a heredoc for known text.

Use Bash when the output depends on upstream files or directories and can be expressed clearly with the commands available in the node's Linux environment. The script runs from the workspace root and writes every result directly into its `output/` directory using a stable, explicit file name. Keep this directory flat. The Bash node exposes that directory through `output_folder`; use `select_file_in_folder` to retrieve each result.

Do not use a Bash node for:

- running bioinformatics command-line software; use an existing tool node or create a tool node so that the software version and container are explicit;
- a systematic downstream statistical analysis, data-mining procedure, or visualization workflow in Python or R; use `downstream_summary` to describe that analysis;
- invoking Python or R inside Bash. If a deterministic workflow adapter genuinely requires Python and its output feeds another node, use a Python code node; use `downstream_summary` for systematic analysis.

## Available Operations

Use Bash only when standard Linux commands or these additional commands express the transformation clearly:

| Command | Best use in a Bash node |
|---|---|
| `fd` | Fast recursive file discovery by name, extension, or type. |
| `rg` | Fast literal or regular-expression search across one or many text files. |
| `jq` | Query, filter, reshape, and format JSON data. |
| `yq` | Query, filter, modify, and convert YAML data. |
| `mlr` | Process structured TSV, CSV, and other record-oriented text: select fields, filter rows, sort, join, aggregate, and convert formats. |

If the operation needs other software, choose a tool node or Python code node as appropriate.

## Designing the Code Node

To execute native Linux commands, use a Bash code node. The workflow-building agent does not write the script. It
describes the required transformation in the node manager's `request`, and a dedicated coding model generates the
complete script.

The request must state:

- the exact deterministic transformation to perform;
- the meaning and expected type (mounted file/directory path vs literal text value) of each bound input;
- every deterministic output file name that a later workflow node will consume;
- any important validation, filtering, sorting, formatting, or preservation rules.

Before adding the code node, create one `bind_param` node per input and use a distinct short option for each one across
both `input_files` and `input_values`. Connect file or directory bindings to `input_files`, and connect dynamic text
value bindings (`value_string`, `read_file_content`, or a subgraph boundary input) to `input_values`. Use `input_values`
only for dynamic values such as `sample:<key>` metadata, upstream `read_file_content` values, or subgraph-exposed
parameters; keep fixed constants directly in the Bash script. The manager reads these bindings and gives their exact
option names and kinds to the coding model. After the node is returned, inspect the generated code before connecting
downstream nodes.

### bind_param Node

To map input files, directories, or text values to parameters defined in the Bash code, first connect the upstream node
to a `bind_param` node and set the node's `parameter` field to the corresponding option, such as `-f` or `-s`.
The output of this node is bound to that parameter and connected to the code node's `input_files` (for paths) or
`input_values` (for text values). Note that this node can bind only one parameter at a time;
if the code node requires multiple input parameters, multiple `bind_param` nodes must be used with unique options.

### select_file_in_folder Node

The processor node `select_file_in_folder` is used to select a file with a specific name from a folder. The code node's `output_folder` refers to the workspace's `output/` directory; to retrieve files from
it, this processor must be used. Use the plain file name without an `output/` prefix.
To use it, set the node's `file_name` parameter to the name of the file you want to select from the folder. The output of this node is the file with that name in the directory. Note that this node can
select only one file at a time; to expose multiple files as separate workflow outputs, use multiple instances of this node.

## Static Text Example

Suppose a downstream tool requires `samples.txt` with the following known content:

```text
sample_a
sample_b
```

Add a `text_file` resource node with `file_name` set to `samples.txt` and `content` set to the text above. Connect its `file_path` output directly to the tool. No bash or `select_file_in_folder` node is needed.

## Runtime Processing Example

Suppose you need to extract chromosome names and lengths from an upstream `.fai` file.

1. In the workflow, add a `bind_param` node and set its `parameter` field to `-f`.
2. In the workflow, request a Bash code node that extracts the first two tab-separated columns and writes
   `chrom_sizes.txt`.
3. In the workflow, add a `select_file_in_folder` node and set its `file_name` field to `chrom_sizes.txt`.

Preceding nodes must be added before the code node. Therefore, first determine the required inputs and their short options, add the corresponding `bind_param` nodes, and then add the Bash code node. Add the `select_file_in_folder` node after it for each result that another node must consume.
