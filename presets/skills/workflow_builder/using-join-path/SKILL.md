---
name: using-join-path
description: Use the join_path processor when an upstream node provides a containing directory but a downstream tool requires a known nested path or multi-file index prefix, such as a BWA directory-plus-basename input.
---

# Using the Join Path Processor

Use `join_path` to append a known relative path to an upstream file or directory path. The node constructs a path
value only; it does not copy data or require the joined path itself to exist.

This distinction is important for multi-file indexes. Tools such as BWA consume one logical prefix like
`<folder>/<prefix>`, while the actual files are siblings such as `<prefix>.amb`, `<prefix>.ann`, `<prefix>.bwt`,
`<prefix>.pac`, and `<prefix>.sa`. The bare prefix may or may not exist as a file.

## When to Use It

Use `join_path` when all of the following are true:

- an upstream resource or tool output resolves to the containing directory;
- the downstream tool expects a path below that directory rather than the directory itself;
- the required relative path or prefix is known from the user request, resource metadata, or tool contract.

Do not use it when:

- the downstream tool needs the directory itself; connect the directory output directly;
- the upstream node already exposes the complete path or index prefix; connect that output directly;
- a concrete existing file must be selected by name or glob; use `select_file_in_folder` instead;
- the relative path or prefix is unknown or ambiguous. Do not guess it; inspect the available resource information or
  ask the user to identify the prefix.

## Node Contract

Add the processor with `utility_node_manager` using `add_processor`:

- processor name: `join_path`
- input handle: `base_path`
- argument: `relative_path`
- output handle: `path`

Connect exactly one upstream path output to `base_path`. `relative_path` must be non-empty and relative, use `/` for
nested components, and must not begin with `/` or contain `..` traversal.

The output preserves the upstream sample identity, so it can be connected directly to the downstream tool input.

## Example: BWA Index Prefix Example

Suppose the user provides a project resource that maps to this directory:

```text
/project/reference/bwa/
├── genome.fa.amb
├── genome.fa.ann
├── genome.fa.bwt
├── genome.fa.pac
└── genome.fa.sa
```

The general `file` resource loads `/project/reference/bwa/`, but the `BWA mem` tool input `ref_index_prefix` requires
`/project/reference/bwa/genome.fa`.

Build this connection:

```text
file.file_path
  -> join_path.base_path  [relative_path = "genome.fa"]
  -> join_path.path
  -> BWA mem.ref_index_prefix
```

The processor returns `/project/reference/bwa/genome.fa`. At tool execution time, the containing BWA directory is
mounted and BWA receives the corresponding container-side prefix. Do not select one of the `.amb`, `.ann`, `.bwt`,
`.pac`, or `.sa` files individually: together they form the index addressed by the shared prefix.

If the loaded resource is a broader directory such as `/project/reference/`, use a nested relative path such as
`bwa/genome.fa` instead.
