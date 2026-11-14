# Pass-Through Workspace Pattern

Use this pattern when a command modifies an input file/directory in place, or several Tool nodes must keep using
the same file or directory, including its original container path. Typical cases are a configuration command that
records absolute input paths, and a later command that executes or updates the configured workspace in place.
For a single-file modification example, see [in-place modification](in-place-modification.md).

## How Pass-Through Works

Set `tool_type=1` (`DOCKER_PASSTHROUGH`) for every Tool in a workspace chain. A first node creates its
working directory by the usual Tool rules. When a later pass-through Tool receives a `passthrough:` INPUT
from a pass-through Tool, it uses that predecessor's actual working directory instead of creating a new one.
The directory may itself have been inherited from an earlier node.

The host workspace is mounted at `/data/output` throughout the chain. Commands use this same default working
directory, and ordinary OUTPUT declarations keep using `{output_dir}`. New files and directories are declared
as ordinary OUTPUTs; there is no separate pass-through OUTPUT mode or shared mount root.

For configuration validation, use `search_node` and `view_node` to find and inspect an equivalent upstream configure Tool, then run it with `execute_node(node_id=..., input_bindings=...)`. Bind the configured directory and each dependency to that run's outputs using `{"run_id": "...", "output": "..."}` (and `"relative_path"` when selecting a file inside a directory output). Inspect generated directories with `inspect_files(..., root="validation")` if needed, then call `execute_node(input_bindings=...)` without `node_id` to validate the draft using the predecessor's workspace and mount paths. If no suitable configure node exists in the catalog, report the missing prerequisite instead of inventing inputs or skipping validation. The trial directory is read-only, and the validation run isolates each attempt under its own temporary directory. A test path such as `/data/output/03_strelka_config/strelka_run/...` becomes `{output_dir}/strelka_run/...` in the draft. When a first pass-through Tool uses trial files as writable inputs, bind them to trial-relative paths; validation copies those declared paths into the attempt automatically.

Mark each file or directory that should continue downstream as an INPUT with `file_path="passthrough:{name}"`.
It automatically exposes a same-name output pointing to the original host path. Do not manually declare a second
OUTPUT with that name. Initial input mount mappings are retained so absolute paths written into configurations
remain accessible. Connect each configured dependency separately.

For an in-place operation, call the tool directly on the marked INPUT. Do not prepend `cp` or declare a separate
OUTPUT for the modified input. The report keeps that path in `input_files` and records the modification in `note`;
newly created results and the log belong in `output_files` at their actual test paths. The report's `working_dir`
is the process working directory. Package native results according to their locations within the inherited
workspace or input bindings, even when the runner was tested from a separate step directory.

A pass-through Tool may have inputs from ordinary nodes and at most one distinct pass-through predecessor.
Several file connections from that same predecessor count as one source. A pass-through predecessor continues
into at most one pass-through Tool, so the chain stays linear and the workspace is written by one node at a time.
Ordinary nodes may still read its outputs. Path selectors and ordinary subgraph interfaces preserve the original
pass-through source identity.

All pass-through inputs are writable. Copying is required only for data that comes from a resource node
(`resource_file`, `resource_db`, `resource_ncbi_genome`, and other `resource_*` nodes except
`resource_text_file`) or a `value_string` path: route it through `copy2folder` first so the original project
data is never modified.
Outputs of Tool nodes and other workflow processing nodes, including path selectors over them, are already
workflow-produced; connect them directly and do not add copy nodes for them. Each workflow run starts its own
workspace.
Outputs reference mutable data: later nodes can change files already exposed by earlier nodes.

## Example: Configure and Run Strelka2

Strelka2 separates configuration from execution:

1. `configureStrelkaGermlineWorkflow.py` reads the input paths and creates a run directory containing
   `runWorkflow.py` and its serialized configuration.
2. `runWorkflow.py` reads that configuration and writes results and execution state into the configured directory.

Passing only the generated script to a new node is insufficient: the configured input paths and the complete run
directory must remain accessible. Use two Tool nodes, with pass-through mounts for both the run directory and its
external file dependencies.

The example below uses the germline configuration options documented in the repository's Strelka2 2.9.10 help.
It illustrates a single BAM without optional call regions. The stored call-regions variant also needs its BED and
tabix index passed through by the same principle. Give each INPUT a stable, unique logical name when creating a node
and bind it explicitly during validation; the example names and file names are illustrative.

In this example the BAM and the reference genome come from project resource nodes, so they are copied first.
Assume earlier copying nodes have produced these independent workflow files:

```text
copied_bam/
  sample.bam
  sample.bam.bai
copied_reference/
  genome.fa
  genome.fa.fai
```

Declare four separate file INPUTs: `bam`, `bai`, `reference`, and `fai`. The copying step keeps each primary file
and its index in the same host directory, with the companion names shown above. File pass-through mounts preserve
the parent-directory binding, so each pair uses the same mount target. Use stable, unique logical INPUT names
and the companion file names established by the current test run.

`copy2folder` copies all connected files into one folder and only exposes `folder_path`. Copy each primary file
together with its index in one `copy2folder` node, then pick each file with its own `select_file_in_folder`
(`file_name="sample.bam"`, `file_name="sample.bam.bai"`, and so on). Path selectors are transparent, so the
configuration node sees both files in the same copied directory. Copying the BAM and its index in two separate
`copy2folder` nodes would put them in different directories and fail with a fixed mount conflict at `/data/bam`.

If the BAM and its index are instead produced by upstream Tools in the same workflow (for example an alignment and
`samtools index` writing the index beside the BAM), connect those outputs directly without `copy2folder`. A primary
file and its index must still live in one directory to share a mount target.

### 1. Configuration Node: File Mounts

```json
[
  {
    "name": "bam",
    "description": "Input BAM file",
    "file_type": "INPUT",
    "file_path": "passthrough:{bam}",
    "mount_path": "/data/bam",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "bai",
    "description": "BAM index adjacent to the BAM",
    "file_type": "INPUT",
    "file_path": "passthrough:{bai}",
    "mount_path": "/data/bam",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "reference",
    "description": "Reference FASTA file",
    "file_type": "INPUT",
    "file_path": "passthrough:{reference}",
    "mount_path": "/data/reference",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "fai",
    "description": "FASTA index adjacent to the reference",
    "file_type": "INPUT",
    "file_path": "passthrough:{fai}",
    "mount_path": "/data/reference",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "run_dir",
    "description": "Strelka run directory containing the generated driver and configuration",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/strelka_run",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "Configuration command stdout and stderr",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/11_run.log",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": true
  }
]
```

The four INPUT declarations automatically produce the `bam`, `bai`, `reference`, and `fai` outputs. Do not add
separate OUTPUT entries for them. Each primary file and index pair shares one identical parent-directory binding,
which the executor merges.

### Configuration Node: Command Settings

```json
{
  "tool_type": 1,
  "dynamic_params": [
    {
      "command": "--bam={bam}",
      "description": "Input BAM; record its fixed path in the configuration",
      "is_position": false,
      "index": null
    },
    {
      "command": "--referenceFasta={reference}",
      "description": "Reference FASTA; its FAI remains adjacent",
      "is_position": false,
      "index": null
    },
    {
      "command": "--runDir={run_dir}",
      "description": "Create the configured workflow at its stable shared path",
      "is_position": false,
      "index": null
    }
  ],
  "immutable_static_params": "",
  "modifiable_static_params": "",
  "command_template": "configureStrelkaGermlineWorkflow.py {optional_params} > {log} 2>&1"
}
```

For the copied files above, the expanded command is:

```bash
configureStrelkaGermlineWorkflow.py --bam=/data/bam/sample.bam --referenceFasta=/data/reference/genome.fa --runDir=/data/output/strelka_run > /data/output/11_run.log 2>&1
```

The `bai` and `fai` inputs have no separate command-line arguments here. Strelka discovers them beside the BAM and
FASTA. Declare and connect them explicitly to preserve these dependencies throughout the pass-through chain.

### 2. Execution Node: File Mounts

```json
[
  {
    "name": "run_dir",
    "description": "Complete configured Strelka directory, updated in place",
    "file_type": "INPUT",
    "file_path": "passthrough:{run_dir}",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "bam",
    "description": "Same copied BAM file and original binding used during configuration",
    "file_type": "INPUT",
    "file_path": "passthrough:{bam}",
    "mount_path": "/data/bam",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "bai",
    "description": "Same BAM index and original binding used during configuration",
    "file_type": "INPUT",
    "file_path": "passthrough:{bai}",
    "mount_path": "/data/bam",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "reference",
    "description": "Same copied FASTA file and original binding used during configuration",
    "file_type": "INPUT",
    "file_path": "passthrough:{reference}",
    "mount_path": "/data/reference",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "fai",
    "description": "Same FASTA index and original binding used during configuration",
    "file_type": "INPUT",
    "file_path": "passthrough:{fai}",
    "mount_path": "/data/reference",
    "is_report": false,
    "is_log": false
  },
  {
    "name": "log",
    "description": "Workflow driver stdout and stderr, separate from native Strelka logs",
    "file_type": "OUTPUT",
    "file_path": "{output_dir}/11_run.log",
    "mount_path": "/data/output",
    "is_report": false,
    "is_log": true
  }
]
```

The execution node inherits the configuration node's entire working directory. Both use the same host directory
at `/data/output`, and `{run_dir}` resolves to `/data/output/strelka_run`. Strelka's native logs and state stay
inside `strelka_run`. Both nodes declare the usual `11_run.log`. The configuration node starts the workspace and
keeps that name; because the execution node writes its log through `{log}` in an inherited workspace, the runtime
redirects it to `/data/output/<node directory>.log` so the two logs never collide. Use the same Strelka
image/version for both nodes.

### Execution Node: Command Settings

```json
{
  "tool_type": 1,
  "dynamic_params": [
    {
      "command": "{run_dir}/runWorkflow.py",
      "description": "Execute the driver inside the original configured run directory",
      "is_position": true,
      "index": 0
    }
  ],
  "immutable_static_params": "--mode local",
  "modifiable_static_params": "--jobs 4 --memGb 8",
  "command_template": "python {position_params} {optional_params} > {log} 2>&1"
}
```

The expanded command is:

```bash
python /data/output/strelka_run/runWorkflow.py --mode local --jobs 4 --memGb 8 > /data/output/3_strelka2-runworkflow.log 2>&1
```

The log name comes from the node's own directory (`<position>_<tool name>`), here assumed to be
`3_strelka2-runworkflow`. A test run does not rename the log, so the tested command still writes `11_run.log`;
keep that declaration and let the runtime rename it.

The BAM and reference paths are already in the serialized configuration, so this command does not repeat them.
The four file INPUT handles are still required to restore both files and their indexes at the same paths. The
example uses the Python interpreter provided by the matching Strelka image, as in the existing Strelka runner
definition.

### Workflow Connections and Results

Connect the configuration node's outputs to the execution node's inputs individually:

| Configuration output | Execution input | Restored container path |
|---|---|---|
| `run_dir` | `run_dir` | `/data/output/strelka_run` |
| `bam` | `bam` | `/data/bam/sample.bam` |
| `bai` | `bai` | `/data/bam/sample.bam.bai` |
| `reference` | `reference` | `/data/reference/genome.fa` |
| `fai` | `fai` | `/data/reference/genome.fa.fai` |

The execution node automatically republishes these same five handles. Its `run_dir` output points to the original
configured directory, now containing the results of execution. For the germline example, the variant output is
located at `results/variants/variants.vcf.gz` inside that directory, with its `.tbi` companion.

Use `select_file_in_folder` on the execution node's `run_dir` output with
`file_name="results/variants/variants.vcf.gz"`, or declare an ordinary OUTPUT at
`{output_dir}/strelka_run/results/variants/variants.vcf.gz`. Both resolve inside the original workspace.

## Key Points

1. `tool_type=1` selects workspace inheritance; `tool_type=0` keeps ordinary per-node workspaces. Include the type
   when calling `set_tool_info` and when saving or exporting the definition.
2. `passthrough:` is only an INPUT marker. Its automatic output keeps the original path. Newly generated files
   use ordinary OUTPUT declarations under `{output_dir}`. An OUTPUT whose `mount_path` equals a pass-through
   INPUT's `mount_path` is written into that input's host directory instead of the workspace: declare
   `{output_dir}/{bam.name}.bai` at the BAM's mount (e.g. `/data/raw`) to create the index beside the BAM.
   When the tested command takes an explicit output path, use the OUTPUT handle (e.g. `{bam_index}`) to get the
   resolved container path. Keep `/data/output` for results that belong in the workspace.
3. The inherited workspace is both the working directory and the output directory, mounted at `/data/output`.
   The standard `workdir` output exposes that actual inherited directory.
4. Restore each configured input dependency, including BAM/BAI and FASTA/FAI, at its original mount position.
   Preserve companion names and adjacency before entering the chain. Conflicting fixed mounts fail.
5. Accept at most one distinct pass-through predecessor across the node's inputs. Multiple handles from the same
   predecessor are allowed, as are ordinary data-producing predecessors. Each pass-through predecessor continues into
   at most one pass-through Tool; the chain cannot branch or rejoin.
6. Keep the usual `log` output written through `{log}`; in an inherited workspace the runtime renames it per node,
   so successive commands never share a log file. Existing files do not skip a
   pass-through command. A failed command retains its partial changes and blocks dependent nodes.
7. Copy only resource-node data (`resource_*` except `resource_text_file`, and `value_string` paths) with
   `copy2folder`; connect Tool and other workflow-produced outputs directly. Pass-through references cannot be exported through mountlist collections, ForEach,
   marks, or independent workflow runs. Ordinary subgraphs preserve the same source and workspace.
8. Build each node from its actual test-run report. The JSON illustrates the pattern; it does not establish that
   a Strelka analysis has been run or that an existing stored node has been migrated.
