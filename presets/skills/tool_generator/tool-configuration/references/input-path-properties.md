# Input Path Properties

Use these templates when an output name or command argument depends on the actual path bound to an INPUT handle.
They can appear in output `file_path`, `command_template`, and `dynamic_params[].command`.
This is the primary output pattern for tools that automatically derive output filenames from an input path. When
another output pattern applies, use this reference as an additional naming rule for input-derived paths or arguments.

## Supported Properties

The input name must match an INPUT handle in `file_mounts`. For an input handle named `fasta`:

| Template | `S1.fasta` | `S1.fasta.gz` | `S1` |
|---|---|---|---|
| `{fasta.name}` | `S1.fasta` | `S1.fasta.gz` | `S1` |
| `{fasta.stem}` | `S1` | `S1.fasta` | `S1` |
| `{fasta.suffix}` | `.fasta` | `.gz` | Empty string |

Properties follow pathlib's lexical rules. `.stem` removes only the last extension, and `.suffix` includes the
leading dot. Directories and path prefixes use the same rules for their last path component. Plain `{fasta}` still
resolves to the full container input path. Properties come from the original input path and do not change when
mount conflicts rename container directories.

## Output Definition Example

For an INPUT named `fasta` whose actual file is `S1.fasta`, declare its derived output as:

```json
{
  "name": "predict_pdb",
  "description": "Predicted protein structure",
  "file_type": "OUTPUT",
  "file_path": "{output_dir}/{fasta.stem}.pdb",
  "mount_path": "/data/output"
}
```

This declares the expected `S1.pdb`; the tool must generate that name or accept `{predict_pdb}` as its output
argument. A declaration does not rename generated files. Continue to use `{sample_name}` when output naming is
based on biological sample metadata rather than the input filename.

Keep command parameters containing these runtime properties in `dynamic_params`. For a tool whose verified
command uses an input-derived prefix, its dynamic parameter can be `--prefix {fasta.stem}`.

## Limits

Properties are unavailable on OUTPUT handles, `value:` text inputs, and `mountlist:` collections. Only `.name`,
`.stem`, and `.suffix` are supported; do not use expressions, indexing, function calls, or chained attributes.
