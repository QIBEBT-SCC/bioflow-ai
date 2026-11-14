---
name: file-system
description: Load project/sample files or sample metadata, publish outputs for later workflows, and collect sample outputs for project-level aggregation. Use when a workflow reads sample metadata as text or transfers files through project/sample mappings.
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

## Sample Metadata as Text

In a `value_string` resource, `value="sample:<key>"` reads the current sample's `meta_data` at node execution,
instead of looking up a file tag. For example, connect `value="sample:organism"` to the NCBI genome resource's
`species_name` input. Common optional keys are `organism`, `assay`, `layout`, and `platform`; custom keys work too.
Inspect the registered metadata before choosing a key. A missing sample or metadata field makes the text node fail.
Other text remains literal; project metadata lookup and inline interpolation are not supported.

Use this in a sample-level workflow. Downstream Docker tools must expose a `value:` text input to consume the output;
ordinary file inputs still expect paths. File resources continue to resolve `sample:<tag>` through sample file mappings.

## Discovering Existing Marks

Before choosing a new mark name or consuming an output created by another workflow, inspect the live project registry:

```text
resource_node_manager(action="view_marks", params={})
```

The operation has no filters and returns the complete marker registry. It separates initial resources, published
workflow marks, and marks planned by every workflow draft in the current builder session. Project entries use
`proj:<keyword>` and sample entries use `sample:<tag>`. Draft marks are explicitly labeled as not published yet, so they
may be referenced while designing dependent workflows without being mistaken for currently available files.

## Choosing a Cross-Workflow Pattern

| Transfer | Producer | Consumer |
|---|---|---|
| Project output to a later workflow | Connect one output to `project_mark` | Load `proj:<mark_name>` with a general `file` resource |
| Sample output to a later workflow for the same sample | Connect one output to `sample_mark` | Load `sample:<mark_name>` with a general `file` resource |
| Outputs from many sample runs to one aggregation workflow | Publish the same semantic `sample_mark` in every sample run | Use `sample_mark_collection` in a project-level workflow |

Use a direct edge when producer and consumer are in the same workflow. Do not use `project_mark` as a substitute for
collecting outputs belonging to separate samples.

## Project Marks

Use `project_mark` only in a `project_level` workflow. Connect exactly one upstream file or directory and provide a
stable `mark_name`.

When the mark node succeeds, it immediately creates or replaces a dynamic project mapping. Each needed output file or
directory can be marked directly with its own `project_mark`, and later workflows load it directly via `proj:<mark_name>`.

## Sample Marks

Use `sample_mark` only in a `sample_level` workflow. Every sample stores the same semantic mark independently.

When the mark node succeeds, it immediately creates or replaces a dynamic sample-file mapping. A later sample-level
workflow for the same sample loads it with `sample:<mark_name>`.

## Collecting Sample Marks

Use `sample_mark_collection` only in a `project_level` aggregation workflow.

Arguments:

- `mark_names`: one or more dynamic sample tags to collect. Only samples that currently have a requested tag are
  collected; samples without it are skipped

Outputs:

- `mount_collection`: a virtual, read-only mounted root organized as `<sample>/<mark>/...`
- `collection_manifest`: a frozen JSON manifest containing sample, producer workflow/run, tag, and path for each collected entry

Connect `mount_collection` directly to directory-scanning tools such as MultiQC. Files are mounted without copying.
When a downstream tool needs both a generated manifest and a separate folder containing the collected files, also
load the `adapt-sample-mark-collections` skill and follow its two-output connection example.

## Lifecycle Rules

- Initial mappings have `is_dynamic=false`; mark-created mappings have `is_dynamic=true`.
- A dynamic mark cannot replace an initial mapping with the same project keyword or sample tag.
- A mark name belongs to the workflow that first registers it. Another workflow cannot register the same name in the
  same project; validation reports this before the run starts. Choose a new name, or delete the owning workflow,
  which also removes its dynamic marks.
- A mark becomes available as soon as its node succeeds; the complete run does not need to finish.
- Before rerunning a workflow, dynamic mappings produced by that workflow in the same project/sample scope are removed.
- Initial mappings are never removed by workflow reruns.
- `project_mark` is project-level, `sample_mark` is sample-level, and `sample_mark_collection` is project-level.

## Workflow Outputs and Initial Resources

Connect workflow node outputs and dynamic marks (`proj:...` / `sample:...` with `is_dynamic=true`, including planned
draft marks) directly to downstream nodes as usual, and export results with `project_mark` or `sample_mark` directly
when needed by later workflows.

Initial project/sample files (`is_dynamic=false`) and shared system databases/genomes (`resource_db`, genome
resources) remain read-only; if workflow validation reports that a writable pass-through input cannot consume an
initial/external resource directly or hits a shared workspace constraint, follow the returned diagnostic to copy that
input first (e.g. with `copy2folder` and `select_file_in_folder`) or adjust the flagged edge.
