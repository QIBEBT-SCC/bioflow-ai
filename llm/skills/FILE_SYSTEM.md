---
name: FILE_SYSTEM
description: How to load initial and dynamically marked project/sample files and collect sample outputs for aggregation
---

# File System and Dynamic Marks

Initial files and workflow-generated marks use the same project/sample mapping system. Each mapping has an
`is_dynamic` source flag so the platform can distinguish uploaded inputs from workflow outputs.

## Reference Syntax

| Syntax           | Meaning                                           |
|------------------|---------------------------------------------------|
| `proj:<keyword>` | A project file or directory mapping               |
| `sample:<tag>`   | A file or directory mapping on the current sample |

The same syntax loads both initial and dynamic mappings. Use the general `file` resource for ordinary files and
directories. Use the dedicated `sequence` resource only for paired R1/R2 inputs.

## Discovering Existing Marks

Before choosing a new mark name or consuming an output created by another workflow, inspect the live project registry:

```text
resource_node_manager(action="view_marks", params={})
```

The operation has no filters and returns the complete marker registry. It separates initial resources, published
workflow marks, and marks planned by every workflow draft in the current builder session. Project entries use
`proj:<keyword>` and sample entries use `sample:<tag>`. Draft marks are explicitly labeled as not published yet, so they
may be referenced while designing dependent workflows without being mistaken for currently available files.

## Project Marks

Use `project_mark` only in a `project_level` workflow. Connect exactly one upstream file or directory and provide a
stable `mark_name`.

When the mark node succeeds, it immediately creates or replaces a dynamic project mapping. A later workflow loads it
with `proj:<mark_name>`.

## Sample Marks

Use `sample_mark` only in a `sample_level` workflow. Every sample stores the same semantic mark independently.

When the mark node succeeds, it immediately creates or replaces a dynamic sample-file mapping. A later sample-level
workflow for the same sample loads it with `sample:<mark_name>`.

Use a direct edge when producer and consumer are in the same workflow. Marks are intended for cross-workflow transfer.

## Collecting Sample Marks

Use `sample_mark_collection` only in a `project_level` aggregation workflow.

Arguments:

- `mark_names`: one or more dynamic sample tags to collect
- `require_all_samples`: when true, fail unless every project sample currently has every requested dynamic tag

Outputs:

- `mount_collection`: a virtual, read-only mounted root organized as `<sample>/<mark>/...`
- `collection_manifest`: a frozen JSON manifest containing sample, producer workflow/run, tag, path, and missing entries

Connect `mount_collection` directly to directory-scanning tools such as MultiQC. Files are mounted without copying.

## Lifecycle Rules

- Initial mappings have `is_dynamic=false`; mark-created mappings have `is_dynamic=true`.
- A dynamic mark cannot replace an initial mapping with the same project keyword or sample tag.
- A mark becomes available as soon as its node succeeds; the complete run does not need to finish.
- Before rerunning a workflow, dynamic mappings produced by that workflow in the same project/sample scope are removed.
- Initial mappings are never removed by workflow reruns.
- `project_mark` is project-level, `sample_mark` is sample-level, and `sample_mark_collection` is project-level.
