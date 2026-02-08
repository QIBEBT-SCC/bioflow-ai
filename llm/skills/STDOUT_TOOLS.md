---
name: STDOUT_TOOLS
description: How to write configuration files when tools can only use standard output streams as output
---

# Skill: Tools with stdout as output

Taking the tool `seqtk seq` as an example, this tool uses the standard output stream as its output. The actual command is shown below:

```bash
seqtk seq -q 30 S1.fq.gz > /data/output/S1.fa 2> log.txt
```

## File Mounts

```json
[
  {
    "name": "fq_file",
    "description": "FASTQ sequence file",
    "file_path": "{fq_file}",
    "mount_path": "/data/raw",
    "file_type": "INPUT",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "fa_file",
    "description": "FASTA sequence file",
    "file_path": "{output_dir}/{sample_name}.fa",
    "mount_path": "/data/output",
    "file_type": "OUTPUT",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "log",
    "file_path": "{output_dir}/00_run.log",
    "mount_path": "/data/output",
    "file_type": "OUTPUT",
    "is_report": false,
    "is_log": true
  }
]
```

## Command Settings

Due to the involvement of redirected output, defining parameter templates using simple `dynamic_params` is challenging. Therefore, such tools do not require setting `dynamic_params` parameters.
Instead, they directly use filenames defined in file mounts as variables within the template.

```json
{
  "dynamic_params": [],
  "immutable_static_params": "",
  "modifiable_static_params": "-q 30",
  "command_template": "seqtk seq {static_params} {fq_file} > {fa_file} 2> {log}"
}
```

## Key Points

1. This mode no longer requires setting `dynamic_params`.
2. In this mode, the variables allowed in the `command_template` are:
    - The `name` value in `file_mounts` (file variable)
    - `{output_dir}` and `{sample_name}`, used directly to construct parameters related to output paths and sample names.
