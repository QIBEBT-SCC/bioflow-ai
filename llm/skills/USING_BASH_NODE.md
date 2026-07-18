---
name: USING_BASH_NODE
description: Explains when to use a text-file resource versus a Bash code node, the Linux execution environment and preinstalled commands available to Bash code, and how to safely transform upstream files into explicit workflow outputs
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

Use Bash when the output depends on upstream files or directories and can be expressed clearly with the commands available in the node's Linux environment. A Bash node outputs its working directory rather than an individual file, so write each result under a stable, explicit file name and use `select_file_in_folder` to retrieve it.

Do not use a Bash node for:

- running bioinformatics command-line software; use an existing tool node or create a tool node so that the software version and container are explicit;
- a systematic downstream statistical analysis, data-mining procedure, or visualization workflow in Python or R; use `downstream_summary` to describe that analysis;
- invoking Python or R as an escape hatch. The Bash runtime guarantees the Linux environment described below, not Python/R packages or an analysis environment.

## Execution Environment

The node runs its `code` with `/usr/bin/bash` in a Linux environment. Its current working directory is a newly created, writable output directory dedicated to that node invocation. Relative output paths therefore land in the node's output folder. Upstream inputs are not positional: each connected input must first pass through a `bind_param` node, and the runtime appends the bound option and resolved path when invoking the script.

In addition to standard Linux system commands, the environment provides these commands:

| Command | Best use in a Bash node |
|---|---|
| `fd` | Fast recursive file discovery by name, extension, or type. Prefer it over a complicated `find` expression when either is suitable. |
| `rg` | Fast literal or regular-expression search across one or many text files. Prefer it over recursive `grep`. |
| `jq` | Query, filter, reshape, and format JSON data. |
| `yq` | Query, filter, modify, and convert YAML data. |
| `mlr` | Process structured TSV, CSV, and other record-oriented text: select fields, filter rows, sort, join, aggregate, and convert formats. |

Combine them with ordinary shell tools when useful.

Do not assume that unlisted language runtimes, third-party packages, bioinformatics programs, network access, or package installation are available. If a task depends on one of those capabilities, choose the corresponding workflow node rather than trying to install software inside the Bash script.

## Node Descriptions

### Code Node

To execute native Linux commands, use a Bash code node and put the complete script in its `code` parameter.

Write scripts defensively:

- start with `set -euo pipefail` so command failures, unset variables, and failed pipelines stop the task;
- parse every bound input with `getopts`; never rely on input order;
- use a distinct short option for every input and keep it consistent with the corresponding `bind_param.parameter` value, including the leading `-`;
- quote every path variable, because resolved input paths may contain spaces or shell metacharacters;
- validate required options and expected file/directory types before processing;
- write results to the current working directory using deterministic names; do not write beside or modify upstream inputs;
- send diagnostic messages to stderr and exit non-zero on invalid input;
- avoid dumping large files to stdout. Filter them directly and redirect the desired result to a named output file.

### bind_param Node

To map input files to parameters defined in the Bash code, first connect the file to a `bind_param` node and set the node's `parameter` field to the corresponding option, such as `-f`.
The output of this node is a special definition of the file bound to that parameter, which is then connected to the input of the code node. Note that this node can bind only one parameter at a time;
if the code node requires multiple input parameters, multiple `bind_param` nodes must be used.

### select_file_in_folder Node

The processor node `select_file_in_folder` is used to select a file with a specific name from a folder. The output of the code node is always the folder of its working directory; to retrieve files from
it, this processor must be used.  
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

Suppose you need to extract chromosome names and lengths from an upstream `.fai` file. Use this script in the Bash code node:

```bash
#!/usr/bin/env bash
set -euo pipefail

fai=''

while getopts ':f:' opt; do
  case "$opt" in
    f) fai="$OPTARG" ;;
    :) echo "Option -$OPTARG requires a value" >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

if [[ -z "$fai" || ! -f "$fai" ]]; then
  echo 'A readable FAI file must be provided with -f' >&2
  exit 2
fi

cut -f1,2 "$fai" > chrom_sizes.txt
```

1. In the workflow, add a `bind_param` node and set its `parameter` field to `-f`.
2. In the workflow, add a bash `code` node and fill in the specific `code` field.
3. In the workflow, add a `select_file_in_folder` node and set its `file_name` field to `chrom_sizes.txt`.

Preceding nodes must be added before the code node. Therefore, first determine the required inputs and their short options, add the corresponding `bind_param` nodes, and then add the Bash code node. Add the `select_file_in_folder` node after it for each result that another node must consume.
