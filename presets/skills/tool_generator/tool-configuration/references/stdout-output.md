# Standard Output Redirection Pattern

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
    "file_path": "{output_dir}/11_run.log",
    "mount_path": "/data/output",
    "file_type": "OUTPUT",
    "is_report": false,
    "is_log": true
  }
]
```

## Command Settings

Define the tool's arguments in `dynamic_params` as usual. Keep stdout and stderr redirection in the command
template, using the output handles directly; redirection does not require an additional command.

```json
{
  "dynamic_params": [
    {
      "index": 0,
      "command": "{fq_file}",
      "description": "FASTQ sequence file",
      "is_position": true
    }
  ],
  "immutable_static_params": "",
  "modifiable_static_params": "-q 30",
  "command_template": "seqtk seq {optional_params} {position_params} > {fa_file} 2> {log}"
}
```

## Key Points

1. Use `{position_params}` and `{optional_params}` for the tool's arguments, as in other output patterns.
2. Reference output handles such as `{fa_file}` and `{log}` directly for redirection.
3. Keep the node to one target-tool invocation. Any processing of its stdout beyond writing it to a file belongs
   in a separate workflow node.
