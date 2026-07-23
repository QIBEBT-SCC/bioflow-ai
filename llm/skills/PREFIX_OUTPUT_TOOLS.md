---
name: PREFIX_OUTPUT_TOOLS
description: How to configure tools whose output argument is a path prefix that produces multiple sibling files, such as RSEM, Bowtie, or other index builders.
---

# Skill: Tools with a path prefix as output

Use this pattern when a command receives one output path such as `/data/output/reference`, but creates multiple
artifacts such as `reference.grp`, `reference.idx.fa`, and `reference.seq` instead of creating `reference` itself.

Do not replace the prefix with the output directory. Downstream tools need the complete directory-plus-prefix value,
not just the directory.

## File Mounts

Taking `rsem-prepare-reference` as an example:

```json
[
  {
    "name": "gtf",
    "description": "Gene annotation in GTF format",
    "file_type": "INPUT",
    "file_path": "{gtf}",
    "mount_path": "/data/annotation",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "genome_fasta",
    "description": "Reference genome FASTA",
    "file_type": "INPUT",
    "file_path": "{genome_fasta}",
    "mount_path": "/data/genome",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "reference_prefix",
    "description": "RSEM reference index path prefix",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/{sample_name}.rsem",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "Execution log",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/11_run.log",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": true
  }
]
```

The `reference_prefix` path is intentionally allowed not to exist as a file or directory. Validation succeeds when
the same directory contains at least one different artifact whose name starts with `{sample_name}.rsem`.

## Command Settings

```json
{
  "dynamic_params": [
    {
      "index": null,
      "command": "--gtf {gtf}",
      "description": "Gene annotation in GTF format",
      "is_position": false
    },
    {
      "index": 0,
      "command": "{genome_fasta}",
      "description": "Reference genome FASTA",
      "is_position": true
    },
    {
      "index": 1,
      "command": "{reference_prefix}",
      "description": "Output path prefix for the RSEM reference index",
      "is_position": true
    }
  ],
  "immutable_static_params": "",
  "modifiable_static_params": "",
  "command_template": "rsem-prepare-reference {static_params} {dynamic_params} {position_params} > {log} 2>&1"
}
```

## Key Points

1. Define one OUTPUT handle for the logical prefix and use that handle in the command.
2. Keep the full prefix under `{output_dir}` so the platform can mount its parent directory and pass the same prefix
   to downstream tools.
3. Do not create a fake file or directory at the bare prefix path.
4. Do not declare every generated index component unless downstream nodes need those components separately.
5. Use this pattern only when generated artifact names begin with the declared prefix. Otherwise declare a real output
   file or directory.
6. Declare the execution log as a separate exact output.
7. Treat the input names in this example as illustrative; every INPUT name must still exactly reuse its logical name
   from the test-run report.
8. The test-run report lists concrete files and directories observed on disk. Collapse a related group into one
   logical prefix only in the generated `file_mounts`.
