---
name: create-new-tool
description: Read this skill when a search determines that no node in the current database can meet the requirements and you are preparing to create a tool node.
---

# Is It Necessary to Create a New Tool Node?

Creating a tool node is costly, so it should only be considered as a last resort. Before calling the `create` operation of `tool_node_manager`, the following checks must be completed in order.

## 1. Confirm that no usable node actually exists in the database

A single empty search or unsatisfactory search result cannot be treated as evidence that the node does not exist in the database. Use at least three substantially different strategies to search, in sequence:

1. Use the concise, standard tool name, node name, or specific subcommand;
2. Use the tool’s recognized aliases, executable filename, package name, repository name, synonyms, or other common spellings;
3. Search using the required functionality, input/output conversion relationship, or a broader tool category.

Search terms should be as concise as possible while covering the primary names the tool may use. Merely changing capitalization, rearranging word order, or repeating the same query does not constitute a new search strategy. If the search returns potentially usable nodes, inspect their functionality, input and output handles, and editable parameters; when necessary, read the tool documentation to confirm whether a candidate node can already satisfy the requirement. Only after all of the above search strategies fail to find a usable node may you proceed to the next check.

## 2. Confirm that the required capability cannot be achieved by combining existing nodes

The absence of a single node that exactly matches the requirement does not mean that a new node must be created. Before creating one, also check the following reuse methods:

- **Extract results from an existing tool’s working directory.** An existing tool may already have generated the required file but may not have declared an independent output handle for it. In that case, connect to the working-directory output provided by the tool and use the `select_file_in_folder` processor to extract the file, rather than creating a new wrapper for the same tool command. If the complete filename cannot be determined, glob wildcards may be used in `file_name`, such as `*.bam`, `*report*.html`, or `**/*summary*.tsv`.
- **Combine multiple existing nodes to fulfill the requirement.** If no single node can complete the entire operation, check whether the same result can be achieved by connecting existing tool nodes, processor nodes, resource nodes, or simple code nodes in sequence. As long as a combination of existing nodes can fulfill the requirement, use those nodes to build the workflow; do not create a new node with composite functionality merely to reduce the number of nodes.

A new tool node may be created only after confirming that no usable node can be found in the database and that none of the reuse or combination methods above can satisfy the requirement.

# Principles for Creating a New Tool Node

## 1. Clearly describe the node’s inputs, operation, and outputs

The creation request should specify:

- Which tool will be used and what specific operation it will perform;
- The inputs expected by the node, along with their semantics and approximate types, such as sequence files, alignment files, reference databases, or directories;
- Which results the node is expected to produce for use by downstream nodes, along with the approximate semantics and types of the corresponding output handles.

There is no need to guess unconfirmed command-line parameters, subcommands, exact filenames, or other implementation details. The focus is on ensuring that the generated node has input and output handles that can connect to existing upstream outputs and pass the required results to downstream nodes. The description should be specific enough to express the workflow’s connection requirements, but should not mislead the node-generation process with unverified details.

## 2. Each node should perform one simple operation

## 2. One Simple Operation per Node

Tool nodes should remain atomic: each node should encapsulate a single command from one tool or one clear, independent operation. Do not create complex nodes that handle multiple consecutive steps, or package data transfer, formatting, preprocessing or postprocessing, or joint analysis involving multiple tools into a single node.

If a requirement contains multiple steps, split it into multiple simple nodes and pass the results through the connections between them; if it includes multiple tool operations that do not exist in the database, create separate nodes for them. Even if the plan describes using two or more tools together to complete an analysis, each node should still encapsulate only a single operation from one tool, with the combination carried out in the workflow.

# Handling Tool Creation Failures

When tool creation fails, determine from the feedback whether the failure to generate the required node was caused by an inadequate description or by the tool itself.
- If the manager returns `TOOL_NODE_CREATION_REQUIRES_HUMAN`, the tool-creation sub-agent has encountered a blocking issue and deliberately handed it back to you. Immediately call `ask_human` with the complete content inside `<human_request>`; do not silently skip the node or retry before receiving the answer. After the user responds, retry creation and include that feedback in the node requirements.
- If creation fails or the returned node is unusable due to an inaccurate description, refine your description and try generating it again.
- If the tool itself is unable to successfully package the node, report the situation to a human and ask for guidance on the next steps.
- If the human feedback requests that the process end here, proceed directly to saving the draft after completing the other wrap-up tasks, and do not process any subsequent TODOs.
