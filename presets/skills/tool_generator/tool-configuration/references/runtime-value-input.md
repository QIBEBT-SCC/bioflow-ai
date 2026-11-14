# Upstream-Produced Runtime Text Input

`value:` is a narrow exception for passing a pure-text result from an upstream workflow node into a tool through an
input handle. It is not a general replacement for static parameters.

## Admission Rule

Use a `value:` input only when all of the following are true:

1. The argument is a text value, not a file, directory, or path prefix.
2. Its value depends on data produced or inspected during the current workflow run, or on the current sample's registered metadata.
3. An earlier node performs that computation or metadata lookup and the tool needs an input handle connected to that node's output.

If any condition is false, do not create a `value:` file mount. Put ordinary literal arguments in
`immutable_static_params` or `modifiable_static_params` instead.

In particular, do not use `value:` merely because a parameter is textual, optional, or user-adjustable. Thread
counts, quality thresholds, flag masks, Boolean switches, species names, modes, presets, and similar tool settings
remain static parameters when they do not depend on an upstream runtime result. A value being editable between runs
is not sufficient: an upstream node must compute it or load it from the current sample during the run.

## Current-Sample Metadata

A `value_string` node with `value="sample:organism"` reads `Sample.meta_data["organism"]` for the current sample.
Its `value` output can feed a tool's `value:{organism}` input when the tool needs the sample-specific organism.
The same lookup supports common keys `assay`, `layout`, and `platform`, as well as custom metadata keys. Metadata
is optional at sample registration; the text node fails if a referenced field is missing. Do not put `sample:organism`
directly in the tool command or mount definition: the text node resolves it and passes the resulting text downstream.

## Database-Backed Example: `samtools view (bam2bam with regions)`

This tool is stored with UID `06a8cce3-b199-75ec-8000-7de2cd3eac38`. It accepts region arguments that may be derived
from the header of the BAM being processed. For example, an upstream `samtools view (bam2header)` node can emit the
SAM header, and a following glue node can inspect its `@SQ` records and produce the reference names to retain. The
result may resemble `chr1 chr2 chrX`, but its actual value depends on the BAM used in that workflow run.

That dependency makes `region_value` eligible for an input handle:

```json
{
  "name": "region_value",
  "description": "Region arguments for filter",
  "file_type": "INPUT",
  "file_path": "value:{region_value}",
  "mount_path": "/dev/null",
  "is_report": false,
  "is_log": false
}
```

The same variable is then used as a positional runtime parameter:

```json
{
  "command": "{region_value}",
  "description": "Region arguments for filter",
  "index": 1,
  "is_position": true
}
```

The relevant command settings from the stored tool are:

```json
{
  "immutable_static_params": "-b",
  "modifiable_static_params": "-h -f 2 -F 3844 -q 20 -@ 8",
  "command_template": "samtools view {optional_params} {position_params} > {log} 2>&1"
}
```

This division is intentional:

- `region_value` comes from an upstream computation over the current BAM header, so it uses `value:`.
- `-@ 8` is a thread-count setting; `-q 20`, `-f 2`, and `-F 3844` are filtering settings. They do not require an
  upstream result, so they remain in `modifiable_static_params`.
- `-b` controls the packaged tool's BAM output contract, so it remains in `immutable_static_params`.

Do not create separate `value:` handles for any of these static settings.

## Configuration Requirements

When the admission rule is satisfied:

1. Set the input mount's `file_path` to exactly `value:{<name>}` and use the same globally unique name for the input
   handle and command variable.
2. Set `file_type` to `INPUT`. Use `mount_path="/dev/null"` to make clear that no filesystem bind mount is involved.
   This is the sole exception to the normal requirement that filesystem mount roots be direct children of `/data`.
3. Reference the value from `dynamic_params` in the same way as any other runtime variable, preserving whether the
   target command expects it positionally or behind an option name.
4. Describe which upstream computation or current-sample metadata lookup produces the value and why it needs an input handle.
5. Before accepting the configuration, explicitly verify that no proposed `value:` handle is actually a thread
   count, threshold, flag, mode, preset, or other ordinary static setting.
6. When validating the draft with `execute_node`, bind each `value:` input in `input_bindings` directly to a
   literal string (e.g. `{"region_value": "chr1 chr2"}`).
