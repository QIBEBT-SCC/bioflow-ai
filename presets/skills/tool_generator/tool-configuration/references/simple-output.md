# Standard Explicit Output Pattern

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
  "command_template": "samtools sort {optional_params} {position_params} > {log} 2>&1"
}
```

It can be seen that the final `command_template` contains the following variables:

- `{optional_params}`: Non-positional dynamic parameters and the immutable/modifiable static parameters
- `{position_params}`: Positional dynamic parameters, sorted by index
- `{log}`: Log file, directly using the filename from the previous file mounting

## Key Points

- Map the target step's test output directory to `/data/output` in the packaged node. For example,
  `/data/output/04_Bowtie2/result.bam` becomes `file_path="{output_dir}/result.bam"` with `mount_path="/data/output"`.
  Drop the test step directory and keep subdirectories required by the tool itself.
- As long as a parameter involves the output path (`{output_dir}` variable) and sample name (`{sample_name}` variable), it should be placed in `dynamic_params`.
- Parameters such as the number of threads and species names, which are unrelated to the output path and filename, should be placed in the corresponding static parameters.
- `dynamic_params` is a configuration field, not a command-template placeholder; its entries are expanded through
  `{optional_params}` or `{position_params}` according to `is_position`.
