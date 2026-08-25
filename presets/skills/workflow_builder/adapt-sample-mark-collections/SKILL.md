---
name: adapt-sample-mark-collections
description: Collect files made by many sample-level workflow runs and use them in one project-level workflow. Use when `sample_mark_collection` provides `mount_collection` and `collection_manifest`, especially when the next tool needs a folder, a manifest file, or both. Examples include QIIME 2 imports, MultiQC, and cohort tools.
---

# Use Files from Many Sample Runs

Use this skill when every sample run creates the same kind of output and a later project-level workflow needs all of
those outputs together.

Also load the `file-system` skill. It explains how to find existing marks and when to use sample-level or project-level
workflows.

## Understand the Two Outputs

Add `sample_mark_collection` to a `project_level` workflow. Set `mark_names` to the sample marks you need. Normally set
`require_all_samples=true` so the workflow stops when any sample is missing a required file.

The node returns two different things:

- `mount_collection` gives the next Docker tool a read-only folder containing all collected files. Connect it directly
  to a folder input on a tool node. Do not send it through `bind_param`, a CODE node, or a file processor.
- `collection_manifest` is a normal JSON file that describes what was collected. Send it to a CODE node when you need
  to create a TSV, CSV, or another manifest format for the next tool.

The JSON has an `entries` list. Useful fields in each entry are:

```text
sample_name   name of the sample
mark_name     sample mark that selected the file
mount_name    file path relative to mount_collection
source_path   original host path; keep it only as history information
kind          file or directory
```

Use `mount_name` when writing paths for the next tool. If the tool receives `mount_collection` at `/data/input`, the
path to one entry is:

```text
/data/input/<mount_name>
```

Do not build this path yourself from the sample name, mark name, or filename. The backend may clean unsafe characters
when it creates `mount_name`.

Do not ask Python code to open or copy `source_path`. The Python container can read the JSON file, but the host paths
written inside that JSON are not mounted into the Python container.

## Connect the Nodes

### The tool only needs a folder

Connect the files directly:

```text
sample_mark_collection.mount_collection -> tool.input_folder
```

This is the usual pattern for MultiQC and other tools that scan a folder.

### The tool needs both a manifest and a folder

Split the two outputs like this:

```text
collection_manifest
  -> bind_param
  -> Python or Bash CODE node
  -> select_file_in_folder
  -> tool.manifest

mount_collection
  -> tool.input_folder
```

The CODE node should create only the small manifest file. It should not copy the collected files. The original
`mount_collection` supplies those files directly to the tool.

Build this pattern in the following order:

1. Inspect the tool inputs and find the folder path used inside its container, such as `/data/reads`.
2. Add `sample_mark_collection` with every required mark.
3. Connect `collection_manifest` to `bind_param`. Use a clear Python option such as `--collection-manifest`, or a short
   Bash option such as `-m`.
4. Ask the CODE node to write one file with a fixed name.
5. Add `select_file_in_folder` for that file.
6. Connect the selected file to the tool's manifest input.
7. Connect `mount_collection` directly to the tool's folder input.

Tell the code generator all of the following in plain language:

```text
Read the BioFlow collection JSON from <input option>. Read its entries using the exact fields sample_name, mark_name,
mount_name, source_path, and kind. Write <output filename> in <required format>. For every file path in the output,
write <tool folder path>/<mount_name>. Do not open or copy source_path. Get sample names from the JSON instead of
hard-coding them. Report an error for duplicate sample/mark pairs, missing required marks, wrong file kinds, unsafe
mount_name values, or entries listed as missing.
```

After the code is generated, read it before adding the next node. Regenerate it if it:

- looks for guessed fields such as `sample`, `sample_id`, `tag`, or `path`;
- opens or copies `source_path`;
- hard-codes sample names;
- builds file paths without `mount_name`;
- omits a header required by the next tool.

### The tool reads BioFlow collection JSON directly

Connect `collection_manifest` directly only when the tool explicitly accepts BioFlow `artifact_collection` JSON. A
normal TSV or CSV input does not accept this JSON and needs the CODE step above.

## Example

For a QIIME 2 paired-end FASTQ import, call `read_skill_resource` with:

```text
skill_name = adapt-sample-mark-collections
resource_path = references/qiime2-paired-end.md
```

That file shows this same pattern with real mark names, QIIME inputs, and the required TSV header.

## Check the Finished Workflow

Before saving, confirm:

- The workflow is `project_level`.
- `mark_names` exactly match the marks created by the sample workflow.
- The generated manifest uses `mount_name` under the folder path declared by the tool.
- `mount_collection` goes directly to the tool's folder input.
- The selected generated file goes to the tool's manifest input.
- The CODE node does not copy the collected files.
- All tool inputs are connected and no edge is left unfinished.
