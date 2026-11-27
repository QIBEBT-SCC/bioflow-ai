---
name: using-text-inputs
description: >-
  Supply text values instead of file paths in workflows. Load this skill for tool inputs declared as
  `value:{name}`, or whenever a tool or resource needs literal text, text read from an upstream file.
---

# Use Text Values in Workflows

## 1. Available Text Input Methods

### Literal text: String Value

Add the `value_string` resource with `resource_node_manager` using `action="add"`,
`resource_type="value_string"`, and `arguments={"value": "human"}`. Its output handle is `value`.
The value can also contain multiple lines; use the exact text required by the consumer.

The resource type is `value_string`, not `string_value`. Text other than a leading `sample:` reference
is returned literally; this field does not interpolate placeholders or resolve project references.

### Text from a file: Read File Content

The `read_file_content` processor reads a UTF-8 file and outputs its contents through `value`:

- Add it with `utility_node_manager`, `action="add_processor"`, and
  `processor_name="read_file_content"`.
- Connect an upstream **directory** output to `folder_path`.
- Set `arguments.file_name` to a relative filename or glob within that directory, such as
  `organism.txt`. Do not pass a file path to `folder_path`.
- The file must be at most 5 MiB. A glob selects the first matching regular file in sorted path
  order; it does not concatenate files. Contents retain their whitespace and line endings.

For example, a CODE node can write `organism.txt` in its output directory. Connect that directory
to `read_file_content.folder_path`, then connect `read_file_content.value` to the text consumer.
If the consumer expects one token, ensure the producer writes that token rather than a report or
an unwanted trailing newline.

The `text_file` resource serves a different purpose: `arguments={"file_name": "config.txt",
"content": "..."}` creates a real text file and outputs its **path** through `file_path`. Use that
resource when a downstream tool needs a configuration file or another file input. Its output is
not the file's text value. Use `read_file_content` when an existing or generated file's contents
must become a text value.

### Current-sample metadata: String Value with `sample:`

Use the same `value_string` resource with `arguments={"value": "sample:organism"}` in a
`sample_level` workflow. At execution, it reads `Sample.meta_data["organism"]` for the current
sample and returns that field as a string through `value`. A workflow shared by multiple samples
can therefore supply a different organism for each sample.

Common optional metadata keys include `organism`, `assay`, `layout`, and `platform`; custom keys
also work. Inspect the registered sample metadata before choosing a key. A missing current sample,
missing key, or null metadata value fails the node rather than falling back to a literal string.
The value must be the complete reference, such as `sample:organism`; inline expressions such as
`species=sample:organism` are not expanded.

Distinguish metadata from similarly named file references:

- In `value_string`, `sample:organism` reads the metadata field `organism`.
- In a `file` or `sequence` resource, `sample:<tag>` resolves a registered sample **file** mapping.
  A file tag is not a text metadata field.
- `Sample.sample_name` is a separate model field. The current text lookup does **not** automatically
  return it for `sample:name` or `sample:sample_name`. Those references work only if the exact key
  was separately stored in `meta_data`. A registered custom field such as `sample_label` can be
  read with `sample:sample_label`; do not invent metadata that is absent from the samples.

For a species-name consumer such as the NCBI genome resource, a real connection is:

```text
value_string [value = "Homo sapiens"].value
  -> ncbi_genome.species_name
```

Change the string node's value to `sample:organism` to read the species from each sample instead.
Check that the metadata uses the spelling expected by the consumer; species names, taxonomy IDs,
and tool-specific codes such as `human` are not converted automatically.

## 2. Connect Text to Tool Input Parameters

A tool's text input is still declared in `file_mounts` with `file_type="INPUT"`, but its
`file_path` is `value:{<name>}`. For example, a tool that exposes an organism input can declare:

```json
{
  "name": "organism",
  "description": "Organism code supplied by an upstream text node",
  "file_type": "INPUT",
  "file_path": "value:{organism}",
  "mount_path": "/dev/null",
  "is_report": false,
  "is_log": false
}
```

The input handle is named **`organism`**, not `value:{organism}`. The `value:` prefix belongs to
the tool definition; the command refers to `{organism}`. At execution the input is substituted as
text without checking for a filesystem path or creating a bind mount. Do not put `value:` in the
string node's value or in an edge handle name. `mount_path` is ignored for these text inputs.
Substitution does not add shell quoting; multiword values must fit the quoting in the tool's
existing command.

Build the text source before adding its downstream tool, and connect it like this:

```text
value_string [value = "human"].value
  -> tool.organism

value_string [value = "sample:organism"].value
  -> tool.organism
```

These are alternative sources for the same handle, not two simultaneous edges. Choose literal
text for a fixed value, sample metadata for a per-sample value, or `read_file_content.value` for
text computed upstream. An ordinary file input such as `file_path="{bam}"` still requires a path;
passing text to it does not turn it into a text input.

### Catalog Example: `cooler digest`

The existing `cooler digest` tool has normal file inputs `chrom_sizes` and `fasta`, plus an
`enzyme` input declared as `file_path="value:{enzyme}"`. Its command uses the enzyme name after
the FASTA argument. To supply `HindIII`:

1. Inspect the manager help for the actions being used. Search with `tool_node_manager`,
   `action="search"`, and `params={"patterns": ["cooler digest"]}`. Read the returned input
   handles and command, and retain the selected result's tool ID.
2. Add a `value_string` resource with `arguments={"value": "HindIII"}`. Retain its returned
   **panel node ID**.
3. Add the selected tool with `tool_node_manager`, `action="add"`. Along with its two required
   file-input edges, include this text-input edge in `params.input_edges`:

   ```text
   source_node   = the returned String Value panel node ID
   source_handle = "value"
   target_handle = "enzyme"
   ```

   The manager assigns `target_node` when it creates the tool. Use the current IDs returned by
   the managers rather than hard-coded catalog or panel IDs.

The resulting connections are:

```text
chromosome-sizes file -> cooler digest.chrom_sizes
genome FASTA file     -> cooler digest.fasta
String Value [value = "HindIII"].value -> cooler digest.enzyme
```

The same wiring applies to an existing species-name input: connect `String Value.value` to its
actual handle, such as `organism`, and supply the organism spelling that tool expects. If a tool
already exposes a required text handle, supply its upstream node even when the chosen value is
constant. This does not mean ordinary static settings need new handles: keep thread counts,
thresholds, and other settings in the existing editable parameters unless the tool contract
specifically requires an upstream text value.

## 3. Connect Text to CODE Node Parameters

Python, R, and Bash CODE nodes expose a dedicated `input_values` handle alongside `input_files`:

```text
value_string [value = "sample:organism"].value
  -> bind_param [parameter = "--organism"].file
  -> bind_param.file_with_param
  -> code_python.input_values

read_file_content.value
  -> bind_param [parameter = "-g"].file
  -> bind_param.file_with_param
  -> code_bash.input_values
```

- Connect text sources (`value_string` or `read_file_content`, or a subgraph input carrying text) to `bind_param.file`,
  then connect `bind_param.file_with_param` to `input_values` (never `input_files`).
- Option names must be unique across both `input_files` and `input_values` on the same CODE node.
- Reserve `input_values` for dynamic values (such as `sample:<key>`, upstream `read_file_content`, or subgraph-exposed
  parameters); write fixed constants directly inside the script.
