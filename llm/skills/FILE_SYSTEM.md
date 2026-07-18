---
name: FILE_SYSTEM
description: How to load files and directories from plain paths, sample references, and project references; use this whenever a workflow needs input paths or cross-workflow files
---

# File System Auto-Loading Mechanism

The `file` resource node (`resource_file`) is the general path input for this system. It loads both regular files and directories. Its existing `file_path` argument and output handle use that historical name, but the resolved path may point to either kind of filesystem object.

Use the dedicated `sequence` resource only for paired R1/R2 reads. Use the general `file` resource for every other path input, including single-end or long reads, custom annotations, images, result files, and directories. Prefer dedicated database and reference-genome resource nodes when the requested resource is available through them.

## Reference Syntax

| Format            | Scope   | Description                                                   |
|-------------------|---------|---------------------------------------------------------------|
| `proj:<keyword>`  | Project | Resolves a project-level file or directory registered by `global_mark` |
| `sample:<tag>`    | Sample  | Resolves a file or directory registered on the current sample |
| Plain path string | —       | Loads an existing file or directory directly                  |

## Project-Level Reference: `proj:<keyword>`

A project maintains a keyword-to-path mapping table (`project_file_mappings`) for shared resources, such as custom annotation files or results produced by another workflow.

**Applicable Scenario**: All samples share the same resource (e.g., alignment indices, annotation files)

## Sample-Level Reference: `sample:<tag>`

Each sample can have multiple paths, each identified by a `tag`. Common tags include:

| File Type             | Common Tag         |
|-----------------------|--------------------|
| Paired-end R1         | `r1`               |
| Paired-end R2         | `r2`               |
| Single-end sequencing | `read`             |
| Mass spectrometry     | `spectrum`         |
| Image file            | `img`              |
| Other files           | `contig`, `bam`... |

**Applicable Scenario**: Different files within each sample

## Choosing the Input Node

- Use `resource_sequence` when paired R1/R2 reads should be loaded together.
- Use `resource_file` for every other existing file or directory. Set its `file_path` argument to a plain path, `sample:<tag>`, or `proj:<keyword>`.
- Use `global_mark` to register an upstream output as a project-level path. Read it later with `resource_file` and `proj:<keyword>`.

Do not create or look for separate folder, global-file, or single-read resource node types. Those inputs all use `resource_file`.

## Typical Use Case Examples

### Use Case One: Standard Single-Sample Sequencing Analysis

Use default tag names to load paired-end sequences from a sample:

```
{
  "r1": "sample:r1",
  "r2": "sample:r2"
}
```

### Use Case Two: Sample Containing Both Positive and Control Sequencing Data

Specify custom tags when uploading files:

- Positive R1 → tag: `pos_r1`, Positive R2 → tag: `pos_r2`
- Control R1 → tag: `ctrl_r1`, Control R2 → tag: `ctrl_r2`

Use two resource_sequence nodes in the workflow:

```
// Node 1 (Positive)
{
  "r1": "sample:pos_r1",
  "r2": "sample:pos_r2"
}

// Node 2 (Control)
{
  "r1": "sample:ctrl_r1",
  "r2": "sample:ctrl_r2"
}
```

### Use Case Three: Sharing a BED Annotation File

First, generate a BED file using an appropriate tool in a global workflow, then connect this file to a global_mark node to mark it as a project-level file:

```
{
  "mark_name": "exon_bed",
  "description": "..."
}
```

Note that the `mark_name` field only requires the keyword itself without the `proj:` prefix.

Reference this file in other workflows using the resource_file node:

```json
{
  "file_path": "proj:exon_bed"
}
```

### Use Case Four: Loading a Directory

Suppose an upstream workflow publishes a directory containing per-sample QC reports with `global_mark` under the keyword `qc_reports`. Load that result directory in a downstream aggregation workflow using the same resource_file node:

```json
{
  "file_path": "proj:qc_reports"
}
```

When connected to a tool input, the execution layer detects that the resolved path is a directory and mounts the directory itself.

## Important Notes

- A keyword must be unique across the entire project
- A tag must be unique within the same sample
- Tags and keywords use lowercase letters and underscores, with a maximum length of 50 characters
- You define the names of tags and keywords; users configure the corresponding files based on your definitions during usage
