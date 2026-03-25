---
name: USING_BASH_NODE
description: Explains how to manually process files using the bash node
---

# Skill: Using the bash node

At times, to bridge outputs from two different tools (or to implement system operations not encapsulated within processors), bash commands are used for simple file processing.  
To implement this functionality in the workflow system, you need to use three types of nodes: a bash-type code node, and the processor nodes `bind_param` and `select_file_in_folder`.

## Node Descriptions

### Code Node

To execute native Linux commands, you must use a bash-type code node. Unlike R and Python nodes, these nodes are actively executed, so you must write the actual bash code to be executed in the `code`
parameter.  
Since the node cannot accept input parameters in a predefined order, you must use the `getopts` syntax to specify all input files and their corresponding parameters.

### bind_param Node

To map input files to parameters defined in the bash code, first connect the file to a `bind_param` node and set the node's `parameter` field to the corresponding parameter name.  
The output of this node is a special definition of the file bound to that parameter, which is then connected to the input of the code node. Note that this node can bind only one parameter at a time;
if the code node requires multiple input parameters, multiple `bind_param` nodes must be used.

### select_file_in_folder Node

The resource node `select_file_in_folder` is used to extract files with specific names from a folder. The output of the code node is always the folder of its working directory; to retrieve files from
it, this processor must be used.  
To use it, set the node's `file_name` parameter to the name of the file you want to select from the folder. The output of this node is the file with that name in the directory. Note that this node can
extract only one file at a time; to extract multiple files, multiple instances of this node must be used.

## Example

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
2. In the workflow, add a bash-type `code` node and fill in the specific `code` field.
3. In the workflow, add a `select_file_in_folder` node and set its `file_name` field to `chrom_sizes.txt`.

Preceding nodes must be added before the code node. Therefore, you must first determine which parameters are needed, add the `bind_param` nodes accordingly, and then write the code and add the code
node.