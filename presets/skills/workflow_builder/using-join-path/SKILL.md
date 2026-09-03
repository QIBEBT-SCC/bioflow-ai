---
name: using-join-path
description: Use the join_path processor when an upstream node provides a containing directory but a downstream tool requires a known nested path or multi-file index prefix.
---

# Using the Join Path Processor

Use `join_path` to append a known relative path to an upstream file or directory path. The node constructs a path value only; it does not copy data or require the joined path itself to exist.

Most tool nodes in the system already have their output prefix paths constructed in advance, so this processor is generally unnecessary. Use this processor to construct paths only when a directory is read directly through a file or database node, and the tool requires a specific file or prefix.

## Node Contract

Add the processor with `utility_node_manager` using `add_processor`:

- processor name: `join_path`
- input handle: `base_path`
- argument: `relative_path`
- output handle: `path`

Connect exactly one upstream path output to `base_path`. `relative_path` must be non-empty and relative, use `/` for nested components, and must not begin with `/` or contain `..` traversal.

The output preserves the upstream sample identity, so it can be connected directly to the downstream tool input.

## Example: BWA Index Prefix Example

The system-provided BWA Index tool node directly supplies an available prefix path. However, in this example, the user did not build the index from scratch, but instead provided a pre-built index folder directly in the file system:

```text
/project_bwa_reference/
├── genome.fa.amb
├── genome.fa.ann
├── genome.fa.bwt
├── genome.fa.pac
└── genome.fa.sa
```

The general `file` resource can loads `/project_bwa_reference/`, but the `BWA mem` tool input `ref_index_prefix` requires `/project_bwa_reference/genome.fa`.

Build this connection:

```text
file.file_path
  -> join_path.base_path  [relative_path = "genome.fa"]
  -> join_path.path
  -> BWA mem.ref_index_prefix
```

The processor returns `/project_bwa_reference/genome.fa`. At tool execution time, the containing BWA directory is mounted and BWA receives the corresponding container-side prefix.
