---
name: FILE_SYSTEM
description: How to use the file auto-loading mechanism of this system
---

# File System Auto-Loading Mechanism

This system supports dynamic file loading in workflow nodes using **path references** without hardcoding specific paths. References are resolved during task execution and operate at two levels.

## Reference Syntax

| Format            | Level   | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `proj:<keyword>`  | Project | References project-level shared files            |
| `sample:<tag>`    | Sample  | References a specific file of the current sample |
| Plain path string | —       | Used directly without resolution                 |

## Project-Level Reference: `proj:<keyword>`

A project maintains a keyword-to-path mapping table (`project_file_mappings`) for cross-sample shared resources, such as reference genome indices or database paths.

**Applicable Scenario**: All samples share the same resource (e.g., alignment indices, annotation files)

## Sample-Level Reference: `sample:<tag>`

Each sample can have multiple files, each identified by a `tag`. Common tags include:

| File Type             | Common Tag         |
|-----------------------|--------------------|
| Paired-end R1         | `r1`               |
| Paired-end R2         | `r2`               |
| Single-end sequencing | `read`             |
| Mass spectrometry     | `spectrum`         |
| Image file            | `img`              |
| Other files           | `contig`, `bam`... |

**Applicable Scenario**: Different files within each sample

## Usage in Nodes

The system supports reading (and writing) dynamic file types through the following nodes:

- resource_sequence: Reads sample-level files
- resource_file: Reads both project-level and sample-level files
- global_mark (processor node): Marks a file as a project-level reference

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

## Important Notes

- A keyword must be unique across the entire project
- A tag must be unique within the same sample
- Tags and keywords use lowercase letters and underscores, with a maximum length of 50 characters
- You define the names of tags and keywords; users configure the corresponding files based on your definitions during usage
