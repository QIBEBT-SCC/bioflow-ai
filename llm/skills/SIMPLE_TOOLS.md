---
name: SIMPLE_TOOLS
description: Example of writing configuration files for standard command-line tools, applicable to most command-line tools.
---

# Skill: Example Configuration for Standard Tools

Taking `samtools sort` as an example, the command for this tool is:

```bash
samtools sort --write-index -@ 32 -o /data/output/A1.sorted.bam A1.bam > /data/output/11_run.log 2>&1
```

## File Mounts

```json
[
  {
    "name": "bam_in",
    "description": "Input BAM/SAM file to be sorted by genomic coordinate",
    "file_type": "INPUT",
    "file_path": "{bam_in}",
    "mount_path": "/data/raw",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "bam_sorted",
    "file_type": "OUTPUT",
    "description": "Sorted BAM file organized by genomic coordinate",
    "file_path": "{output_dir}/{sample_name}.sorted.bam",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "log",
    "file_path": "{output_dir}/11_run.log",
    "file_type": "OUTPUT",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": true
  }
]
```

## Command Settings

```json
{
  "dynamic_params": [
    {
      "index": 0,
      "command": "{bam_in}",
      "description": "Input BAM/SAM file to be sorted by genomic coordinate",
      "is_position": true
    },
    {
      "index": null,
      "command": "-o {bam_sorted}",
      "description": "Output sorted BAM file",
      "is_position": false
    }
  ],
  "immutable_static_params": "--write-index",
  "modifiable_static_params": "-@ 32",
  "command_template": "samtools sort {static_params} {dynamic_params} {position_params} > {log} 2>&1"
}
```

It can be seen that the final `command_template` contains the following variables:

- `{static_params}`: Automatically concatenated from two different static parameters
- `{dynamic_params}` and `{position_params}`: Split from the original dynamic_params according to is_position
- `{log}`: Log file, directly using the filename from the previous file mounting

## Key Points

- As long as a parameter involves the output path (`{output_dir}` variable) and sample name (`{sample_name}` variable), it should be placed in `dynamic_params`.
- Parameters such as the number of threads and species names, which are unrelated to the output path and filename, should be placed in the corresponding static parameters.
- The `{dynamic_params}` variable in command_template is not equal to the `dynamic_params` in the parameter settings; do not confuse them.