---
name: CROSS_WORKFLOW_FILE_TRANSFER
description: Register project or sample workflow outputs and consume them through normal project/sample file mappings
---

# Cross-Workflow File Transfer

Dynamic marks use the same references as initial files. The mapping's `is_dynamic` field records its source.

Before referring to an existing mark, call
`resource_node_manager(action="view_marks", params={})` to inspect the current project registry.

## Project-to-Workflow

In a project-level producer workflow:

1. Connect the output to `project_mark`.
2. Give it a stable mark name, such as `star_index`.
3. In a later workflow, load `proj:star_index` with a general `file` resource.

## Sample-to-Sample

In a sample-level producer workflow:

1. Connect the output to `sample_mark`.
2. Give every sample run the same semantic mark, such as `final_bam`.
3. In a later workflow run for the same sample, load `sample:final_bam`.

## Sample-to-Project Aggregation

Add `sample_mark_collection` to a project-level aggregation workflow and configure the required mark names. Its
`mount_collection` output preserves sample identity in a virtual `<sample>/<mark>/...` tree, while
`collection_manifest` records the exact source mappings.

Do not use `project_mark` as a substitute for a sample collection.
