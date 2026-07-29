---
name: multi-directory-input
description: How to use tools that need to scan multiple upstream tool directories, such as MultiQC and other report aggregation tools.
---

# Skill: Multi-directory tool input

Use this skill when a tool needs to scan or aggregate a variable number of upstream tool output directories. Typical examples include MultiQC and other report aggregation tools that accept one root
directory and recursively discover files inside it.

## Core Pattern

Do not connect many upstream directories directly to a normal tool node input. Tool nodes have fixed input handles, and each tool input handle accepts one logical input.

Instead, use the `collect_mount_dirs` processor:

```
Tool A workdir ┐
Tool B workdir ├─ collect_mount_dirs.mount_dir_list -> AggregationTool.input_dir
Tool C workdir ┘
```

The `collect_mount_dirs` processor creates a mount-list manifest. During tool execution, the system expands that manifest into multiple container mounts under one logical input directory.

## Processor Node

Use `utility_node_manager` to add the processor node named `collect_mount_dirs`.

Input handle:

- `dirs`: connect one or more upstream tool `workdir` outputs to this handle

Output handle:

- `mount_dir_list`: connect this output to the downstream aggregation tool's directory input

Arguments:

- No arguments are required.

When adding this processor, pass an empty arguments object.

## Downstream Tool Definition

The downstream tool should still be defined as a normal tool with a single directory input. Do not put `mountlist:` in the tool definition.

For a MultiQC-style tool, define one input file mount like this:

```json
{
  "name": "input_dir",
  "description": "Root directory containing upstream report directories to scan",
  "file_type": "INPUT",
  "file_path": "{input_dir}",
  "mount_path": "/data/multiqc_inputs"
}
```

The command can then use:

```bash
multiqc {input_dir} -o /data/output --force
```

At runtime, `{input_dir}` becomes the container path such as `/data/multiqc_inputs`, and the upstream directories are available as subdirectories under that path.

## When To Use

Use `collect_mount_dirs` when:

- The tool scans a directory tree instead of requiring one fixed input file
- The number of upstream directories may vary by workflow
- The tool should see multiple upstream tool working directories under one root directory
- You need to avoid host symlinks that may break inside the container

Common examples:

- MultiQC over FastQC, fastp, STAR, Salmon, HISAT2, featureCounts, or other QC/report output directories
- Report collection tools that recursively scan tool output folders
- Custom aggregation tools that expect one root folder containing several result directories

This processor collects directories already present in the same workflow run. For a genuine project-level aggregation
over outputs registered by separate sample runs, use the `sample_mark_collection` resource instead and connect its
`mount_collection` output directly to the aggregation tool.

## Important Notes

- Prefer connecting upstream tool `workdir` outputs, not individual report files, when the aggregation tool performs recursive discovery.
- The `mount_dir_list` output is not a normal folder path. It is intended for downstream tool inputs handled by the Docker tool execution system.
- Do not connect `mount_dir_list` to processors such as `select_file_in_folder`; use it only as a tool directory input.
- If only one known file is needed, use the normal file output handle directly instead of this processor.
- If the tool requires a fixed list of individual files rather than directory scanning, create explicit tool inputs or use a bash node instead.
