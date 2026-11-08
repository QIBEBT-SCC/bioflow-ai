---
name: create-new-tool
description: Read this skill when a search determines that no node in the current database can meet the requirements and you are preparing to create a tool node.
---

# Is It Necessary to Create a New Tool Node?

## Search before requesting creation

Before calling `tool_node_manager(action="create")`, search the catalog for the current operation with
`action="search"`. Its `patterns` are case-insensitive regular expressions over complete commands, and every pattern
must match. If a restrictive search finds nothing, remove optional flags or other restrictions and search by program
name or subcommand. Review the candidates for reusable interfaces and editable parameters before creating a node.

Every creation request must include these two required parameters alongside `name` and `request`:

- `attempted_searches`: A non-empty list of the actual search parameters already executed for this operation.
  Preserve each combination separately, for example:
  `[{"patterns": ["samtools", "view"]}, {"patterns": ["samtools"]}]`.
  Do not list planned searches or unrelated searches from another operation.
- `reason`: Explain whether no suitable node was found, or which existing candidates cannot be reused and why.
  Describe the incompatible input/output handles or fixed parameters, and why changing editable parameters or
  composing existing nodes cannot satisfy the requirement.

Search first, then retry with the actual search records and a concrete justification.

## Examples

Assume that the tool library already contains the following two tools:

### Existing tool 1: `samtools view (bam2bam)`

- Input:
  - BAM
- Output:
  - BAM

This tool represents general BAM-to-BAM processing with `samtools view`. Options such as `-q`, `-f`, `-F`, `-h`, and threading parameters can be configured as needed.

### Existing tool 2: `samtools view (bam2header)`

- Input:
  - BAM
- Output:
  - SAM header

This tool runs `samtools view -H` and extracts the SAM header from the input BAM.

---

### Positive example: create a new tool

Suppose a workflow needs to run:

```bash
samtools view input.bam chr1 chr2 chrX
```

This operation requires (BAM + BAI + region arguments) and produces (BAM).
The existing `samtools view (bam2bam)` tool is insufficient because region-based random access requires an indexed BAM and additional positional region arguments.

A new `samtools view (bam2bam with regions)` tool should therefore be created with an interface such as:

- Input:
  - BAM
  - BAI
  - regions
- Output:
  - BAM

The reason for creating the tool is not that it performs “region extraction”, but that its required interface differs from the existing BAM-to-BAM tool.

---

### Negative example 1: reuse one existing tool with different parameters

Suppose a workflow needs:

```bash
samtools view \
    -f 2 \
    -F 3844 \
    -q 20 \
    input.bam
```

This command selects properly paired reads, removes reads matching specific SAM flags, and applies a minimum mapping-quality threshold.
Although this may represent a distinct analytical task such as “strict BAM filtering”, its interface remains (BAM → BAM)

Therefore, a new tool such as `samtools view (strict filtering)` should **not** be created.
Instead, reuse `samtools view (bam2bam)` and update its parameters:

```text
-f 2 -F 3844 -q 20
```

The analytical behavior changes, but the tool interface does not.

---

### Negative example 2: compose existing tools and a glue step

Suppose a workflow needs to perform strict filtering while excluding the mitochondrial reference.

An original shell command might look like:

```bash
samtools view \
    -o output.strict.bam \
    -h \
    -b \
    -f 2 \
    -F 3844 \
    -q 20 \
    input.bam \
    $(samtools view -H input.bam \
        | awk '$1=="@SQ" && $2!="SN:MT"' \
        | cut -f2 \
        | sed 's/^SN://')
```

It may be tempting to create a dedicated tool such as `samtools view (strict filtering without MT)`
However, this is unnecessary because the task can be decomposed into existing reusable components.

First, use `samtools view (bam2header)` to obtain the SAM header.
Then use a simple bash step to extract reference names from the header while excluding `MT`:
Finally, pass those regions together with the filtering parameters to `samtools view (bam2bam with regions)` using parameters such as:

```text
-f 2 -F 3844 -q 20
```

The workflow therefore becomes:

```text
BAM
 │
 ├───────────────┐
 │               │
 ▼               │
bam2header       │
 │               │
 ▼               │
code: extract    │
non-MT regions   │
 │               │
 ▼               ▼
regions      BAM + BAI
     \          /
      \        /
       ▼      ▼
samtools view
(bam2bam with regions)
       │
       ▼
 filtered BAM
```

No new task-specific tool is required.

The general principle is that a workflow-specific operation should preferably be expressed through **parameterization and composition of reusable tools**. A new tool should be introduced only when the required interface itself is new.

In short:

> **Create tools for distinct interfaces, not for distinct parameter combinations or analytical intentions.**

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
- If the manager returns `TOOL_NODE_CREATION_HANDOFF`, the tool-creation sub-agent has deliberately stopped without creating a node. Read the complete `<handoff_context>` and do not retry the unchanged request.
  - If the handoff establishes that the requested tool cannot perform the operation but identifies alternatives, return to workflow planning and search for the suggested tools. Create a different node only after the normal catalog search shows that it is missing.
  - If the handoff establishes that the request is not atomic, split it into the reported single-command operations. Search for or create each Tool node separately and use a Bash node for simple glue, formatting, or text-processing steps where appropriate.
  - Call `ask_human` only when the handoff still requires the user to choose an alternative or restate the requirement. Include the relevant verified findings in the question, then use the answer to replan rather than retrying the original request verbatim.
- If creation fails or the returned node is unusable due to an inaccurate description, refine your description and try generating it again.
- If the tool itself is unable to successfully package the node, report the situation to a human and ask for guidance on the next steps.
- If the human feedback requests that the process end here, proceed directly to saving the draft after completing the other wrap-up tasks, and do not process any subsequent TODOs.
