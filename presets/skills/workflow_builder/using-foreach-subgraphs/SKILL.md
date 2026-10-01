---
name: using-foreach-subgraphs
description: Build a workflow stage that processes an unknown number of files produced inside one run, such as annotating every bin FASTA after binning. Use a file collection and one reusable subgraph; do not use this for whole-workflow sample runs or cross-sample aggregation.
---

# Process dynamic files inside one workflow

Use this when an upstream node creates a folder containing an unknown number of matching files and each file needs the same follow-up analysis. The workflow itself still runs at its chosen sample or project scope. The number of child runs is determined when the collection node executes.

Choose the right batch boundary:

- When the same complete workflow runs for registered samples, use a sample-level workflow and select the samples at run time.
- When outputs of separate sample runs must be analyzed together, use a project-level workflow with `sample_mark_collection`. Read `adapt-sample-mark-collections` for its input contract.
- When one run dynamically creates many files for identical follow-up work, use `collect_file_collection` followed by one subgraph node.

## Build the file stage

1. Inspect the upstream producer's actual output directory and naming. `collect_file_collection` is a built-in processor: inspect it directly with `utility_node_manager.view(node_id="collect_file_collection")`, then add it with `utility_node_manager.add_processor(processor_name="collect_file_collection")`. Set its `pattern` to one relative glob for the desired regular files and a meaningful `subfolder_prefix`. A pattern with no matches fails at runtime. Do not assume every binning tool uses `.fasta`.
2. Call `subgraph_node_manager` `search` and inspect relevant results with `view`. Reuse a subgraph only when its single-file task, input handles, and outputs fit. Otherwise call `create` with `purpose="foreach"`, then `add` with the saved UID.
3. In the child task, declare external inputs with `add_subgraph_inputs` before adding tools. The builder will return the virtual source node and stable input port IDs. Connect those to the internal tools in their normal upstream-first order. Choose the outputs with `set_subgraph_outputs` and finish with `save_subgraph`.
4. Connect `collect_file_collection.collection` to exactly one subgraph input: the current file. Connect any other subgraph inputs to shared upstream sources. This connection turns the subgraph into a ForEach stage. Its outward result is `results_folder`, not the child output handles.
5. Feed `results_folder` to downstream folder analysis, or read its `manifest.json` when the analysis needs each original item and exported result. Use the manifest's mapping; do not guess child folder names. Do not create one tool node per future file or register generated files as project samples.

ForEach children cannot contain another collection/ForEach or `sample_mark` / `project_mark`. Keep cross-item aggregation after the batch stage.

## Results manifest for downstream CODE

`subgraph_node_manager.add` returns a `foreach_manifest_contract` when the collection is connected. Its `output_ports` list gives the saved subgraph's stable output IDs and display names. Use the ID, not the name, as the key in `items[].outputs`. The manifest is created only when ForEach runs; its absence during workflow construction is expected and does not require a question to the user.

When a downstream CODE node needs per-item exports, read the contract returned by `subgraph_node_manager.add`; for a subgraph already on the panel, inspect its interface with `workflow_manager.view_node`. Put the required output port ID and the manifest lookup rule in the `add_new_code.request`. Check that the generated code reads `manifest.json` through `items[].outputs[port ID]` and resolves each value relative to `results_folder` before connecting downstream nodes. If it guesses a directory layout or filename suffix, regenerate the CODE node using the contract. Do not ask the user for a runtime manifest instance to learn this schema.

The `results_folder/manifest.json` format is version 1. It has an `items` array in collection order. Every item has `key`, `name`, `relative_path` (original collection input), `subfolder`, `status`, `run_uid`, and `outputs`. For a successful item, `outputs` maps each exported subgraph output **ID** to a path relative to `results_folder`. A file export maps to its copied file; a directory export maps to its copied directory. Failed items have no exported output paths.

For example, if the subgraph interface declares `id="output_1", name="bin_record"` and the selected source file is named `bin_record.tsv`, one item can look like this:

```json
{
  "version": 1,
  "items": [
    {
      "key": "9a36da116c7910d809ccf555",
      "name": "bin.1.fa",
      "relative_path": "bins/bin.1.fa",
      "subfolder": "bin_1",
      "status": "succeeded",
      "run_uid": "00000000-0000-0000-0000-000000000001",
      "outputs": {
        "output_1": "bin_1/d8c675a41ff88f6c9253b318/bin_record.tsv"
      }
    }
  ]
}
```

This is an example, not a path template. The middle directory is an opaque port-specific directory, not `output_1`. For a TSV merge, iterate `items`, select `status == "succeeded"`, resolve `results_folder / item.outputs["output_1"]`, and validate that each resolved path is the expected file. Do not recursively search for matching basenames or require a suffix like `output_1/bin_record.tsv`. Check the actual output ID returned for the subgraph before writing the downstream CODE request, then inspect the generated code for this mapping. The file collector fails when its glob matches no files, so a zero-item manifest is not a normal runtime case.

## Example: bin FASTA annotation

```text
binning output directory
  -> collect_file_collection (glob chosen from actual bin names)
  -> saved "Annotate one bin" subgraph
       input genome_fasta -> Prokka -> selected annotation outputs
  -> results_folder -> downstream summary
```

If Prokka also needs a constant database or configuration, declare a second subgraph input and connect that input to a shared source on the parent graph. The annotation subgraph still describes one bin, independently of how many bins the producer emits.
