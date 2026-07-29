# In-Place Modification Pattern

For example, the tool `cooler balance` directly modifies the original file. Commands for this tool look like this:

```
cooler balance --force --nproc 32 H1.cool > /data/output/11_run.log 2>&1
```

## File Mounts

Since the tool directly modifies the source file, both input and output are the same file. However, to ensure file independence, we need to copy the file to a separate output path before making
modifications.
Therefore, we must prepare both the original file (input) and the copied file (output) separately.

```json
[
  {
    "name": "cool_path",
    "description": "Origin .cool Hi-C contact matrix (Cooler format)",
    "file_type": "INPUT",
    "file_path": "{cool_path}",
    "mount_path": "/data/raw",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "balanced_cool_path",
    "description": "Balanced .cool Hi-C contact matrix (Cooler format)",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/{sample_name}.cool",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "Execution log capturing stdout and stderr from cooler balance.",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/11_run.log",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": true
  }
]
```

### Command Settings

Since command structures are no longer composed of single commands and have become more complex, it is now permissible to omit dynamic parameters and directly use file variables within command
templates:

```json
{
  "dynamic_params": [],
  "immutable_static_params": "--force",
  "modifiable_static_params": "--nproc 32",
  "command_template": "cp {cool_path} {balanced_cool_path} && cooler balance {static_params} {balanced_cool_path} > {log} 2>&1"
}
```

## Key Points

1. Input files in the mount correspond to: the source file of the input type and the actual operational file located in the output path (which is also the output file)
2. The command needs to copy the source file to the output file first, and then the tool runs on the output file
3. In this mode, the variables allowed in the template string are:
    - The `name` value in `file_mounts` (file variable)
    - `{output_dir}` and `{sample_name}`, used directly to construct dynamic parameters (output paths, file prefixes, etc.)
    - `{static_params}`: Static parameters; the system will automatically merge and fill in both modifiable and non-modifiable parameters.
