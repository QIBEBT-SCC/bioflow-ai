# In-Place Modification Pattern

Use `tool_type=1` (`DOCKER_PASSTHROUGH`) when the command modifies an input file or directory directly.
The modified input is writable and automatically becomes a same-name output at its original path. This also
applies to a single Tool; it does not require a preceding pass-through Tool.

For example, `cooler balance` updates the input `.cool` file:

```bash
cooler balance --force --nproc 32 H1.cool > /data/output/11_run.log 2>&1
```

The test-run report keeps `cool_path` in `input_files` and describes the verified modification in `note`.
If no new result files are created, `output_files` contains only the log. Give the INPUT a stable, unique
logical name and bind it explicitly during validation; the name below is illustrative.

## File Mounts

```json
[
  {
    "name": "cool_path",
    "description": "Hi-C contact matrix updated in place with balancing weights",
    "file_type": "INPUT",
    "file_path": "passthrough:{cool_path}",
    "mount_path": "/data/raw",
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

Do not declare another OUTPUT for the updated `.cool` file. The automatic `cool_path` output points to the input
file after balancing; `log` is an ordinary newly generated OUTPUT.

## Command Settings

Call `set_tool_info` with `tool_type=1`. Keep the tool invocation atomic and define the file argument normally:

```json
{
  "dynamic_params": [
    {
      "description": "Hi-C contact matrix to balance in place",
      "command": "{cool_path}",
      "is_position": true,
      "index": 0
    }
  ],
  "immutable_static_params": "--force",
  "modifiable_static_params": "--nproc 32",
  "command_template": "cooler balance {optional_params} {position_params} > {log} 2>&1"
}
```

## Key Points

1. Connect outputs produced within the same workflow run and dynamic project/sample marks (`is_dynamic=true`) directly.
   Only initially registered project/sample files (`is_dynamic=false`), `resource_db`/`resource_genome`, unmanaged
   literal paths, or `value_string` paths need an upstream `copy2folder` node (followed by `select_file_in_folder`)
   before connecting to `cool_path`. Do not put the copy inside the Tool command.
2. Outputs reference mutable data. If the workflow also needs the pre-modification version, make an independent
   copy before the modifying node and keep that version on a separate path.
3. Mark only inputs that need writable access or must continue with their original path as `passthrough:`.
   Other inputs remain ordinary read-only INPUTs; newly created results and the log remain ordinary OUTPUTs.
   A companion file that must sit beside a pass-through input, such as an index written by `samtools index` or
   `samtools faidx`, is an OUTPUT with the same `mount_path` as that input (e.g. `/data/raw`) and
   `file_path="{output_dir}/{bam.name}.bai"`; the runtime writes it into the input's directory.
   If the tested command supplies an explicit path for the companion, reference the OUTPUT handle (e.g.
   `{bam_index}`) so the command receives the resolved container path. Record the companion's actual path in
   the test report, even when it is outside the process working directory.
4. Non-workspace in-place/sidecar tools mounted outside `/data/output` may feed downstream tools (including multiple
   predecessors into one downstream `tool_type=1` tool) without workspace conflicts. When tools explicitly inherit a
   shared `/data/output` workspace across nodes, that workspace chain remains linear (one predecessor and one
   successor; collections and ForEach cannot export shared `/data/output` workspaces). Exporting pass-through outputs
   through `project_mark` or `sample_mark` is supported.
5. When validating a standalone in-place modification node with `execute_node`, bind the `passthrough:` input
   directly to its trial-relative path; validation isolates a copy per attempt so the trial data stays untouched.
6. When the command also needs a configured workspace or fixed dependencies across nodes, use the
   [pass-through workspace pattern](passthrough.md) for those mount and connection requirements.
