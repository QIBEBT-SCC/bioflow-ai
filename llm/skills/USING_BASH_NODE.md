---
name: USING_BASH_NODE
description: Explains when to use a text-file resource versus a bash node, and how to process files safely with bash when runtime computation is required
---

# Skill: Using the bash node

Use the simplest node that matches the operation. For a small file whose name and text are already known while building the workflow, use a `text_file` resource node. Use a bash code node only when the file content must be computed at runtime, such as transforming an upstream file or combining upstream results.

When bash is required, the usual pattern uses a bash code node together with the `bind_param` and `select_file_in_folder` processor nodes.

## Choosing between Text File and Bash

Use a `text_file` resource for static text such as:

- tool configuration files;
- sample or accession lists;
- small TSV, CSV, BED, JSON, or YAML inputs;
- any other text that can be written verbatim when the workflow is designed.

Set `file_name` to a plain file name such as `samples.txt` and set `content` to the exact file contents. The node writes UTF-8 text and directly outputs `file_path`, so connect that output to the downstream tool input. Do not create a bash node merely to run `echo`, `printf`, or a heredoc for known text.

Use bash when the output depends on upstream files or requires Linux commands at runtime. A bash node outputs its working directory rather than an individual file, so use `select_file_in_folder` to retrieve a named result.

## Node Descriptions

### Code Node

To execute native Linux commands, you must use a bash code node, and write the actual bash code to be executed in the `code` parameter.  
Since the node cannot accept input parameters in a predefined order, you must use the `getopts` syntax to specify all input files and their corresponding parameters.

### bind_param Node

To map input files to parameters defined in the bash code, first connect the file to a `bind_param` node and set the node's `parameter` field to the corresponding parameter name.  
The output of this node is a special definition of the file bound to that parameter, which is then connected to the input of the code node. Note that this node can bind only one parameter at a time;
if the code node requires multiple input parameters, multiple `bind_param` nodes must be used.

### select_file_in_folder Node

The processor node `select_file_in_folder` is used to extract files with specific names from a folder. The output of the code node is always the folder of its working directory; to retrieve files from
it, this processor must be used.  
To use it, set the node's `file_name` parameter to the name of the file you want to select from the folder. The output of this node is the file with that name in the directory. Note that this node can
extract only one file at a time; to extract multiple files, multiple instances of this node must be used.

## Static Text Example

Suppose a downstream tool requires `samples.txt` with the following known content:

```text
sample_a
sample_b
```

Add a `text_file` resource node with `file_name` set to `samples.txt` and `content` set to the text above. Connect its `file_path` output directly to the tool. No bash or `select_file_in_folder` node is needed.

## Runtime Processing Example

Suppose you need to use a bash script to extract specific columns from a `.fai` file.  
Assume you decide to implement this functionality using a script similar to the following:

```bash
#!/bin/bash
while getopts "f:" opt; do
  case $opt in
    f)
      FAI="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument value" >&2
      exit 1
      ;;
  esac
done

if [ -z "$FAI" ]; then
  echo "Error: Please provide the -fai parameter to specify the FAI file path."
  echo "Usage example: ./process.sh -fai my_data.fasta.fai"
  exit 1
fi

cut -f1,2 "$FAI" > chrom_sizes.txt
```

1. In the workflow, add a `bind_param` node and set its `parameter` field to `-f`.
2. In the workflow, add a bash `code` node and fill in the specific `code` field.
3. In the workflow, add a `select_file_in_folder` node and set its `file_name` field to `chrom_sizes.txt`.

Preceding nodes must be added before the code node. Therefore, you must first determine which parameters are needed, add the `bind_param` nodes accordingly, and then write the code and add the code
node.
The bash code node is used only for running Linux system commands, should not be used for running bioinformatics tools. Running bioinformatics command-line tools should use tool nodes or create new tool nodes.
