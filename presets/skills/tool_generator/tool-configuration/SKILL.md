---
name: tool-configuration
description: Generate a BioFlow Docker tool configuration from a real execution report. Use when classifying a tool's input/output behavior and producing its file mounts, parameters, and command template.
---

# Tool Configuration

Use this skill for every tool-configuration task. First check whether the command modifies an input in place or
must preserve a shared writable workspace across nodes, then classify how its primary output is produced.
Read exactly one matching primary output-pattern reference before configuring the tool. When output names or
command arguments depend on an INPUT filename, also read `references/input-path-properties.md`; this includes
input-derived output prefixes. If, and only if, a non-file text parameter must receive a value dynamically produced
by an upstream node during workflow execution, also read `references/runtime-value-input.md`.

## Choose the Output Pattern

Apply the first matching primary output pattern in this order:

| Pattern | Use when | Reference |
|---|---|---|
| Pass-through workspace | The command requires the same writable workspace across Tool nodes | `references/passthrough.md` |
| In-place modification | The command updates an input file or directory directly; use writable pass-through instead of copying inside the command | `references/in-place-modification.md` |
| Output prefix | An output argument is a path prefix and the tool creates multiple sibling artifacts whose names begin with that prefix | `references/prefix-output.md` |
| Standard-output redirection | The primary result exists only because stdout is redirected to an output file | `references/stdout-output.md` |
| Input-derived output | The tool automatically determines output filenames from an INPUT path instead of accepting an explicit output filename | `references/input-path-properties.md` |
| Standard explicit output | The command writes its primary result to an explicit file or directory path | `references/simple-output.md` |

Stderr or ordinary log redirection does not make a tool a standard-output tool. Use the standard-output pattern only
when stdout itself carries a primary result.

Both pass-through workspace and in-place modification patterns use `tool_type=1` (`DOCKER_PASSTHROUGH`).
Mark modified or continuing INPUTs with `file_path="passthrough:{name}"`; the runtime exposes their same-name
outputs automatically. Ordinary independent Tool workspaces use `tool_type=0` (`DOCKER`). Select the node mode
explicitly.

Each Tool node must execute exactly one actual invocation of the target tool, with optional result/log redirection.
Do not combine commands using separators, pipelines, subshells, or command substitutions, including in parameter
fragments. Copying, preprocessing, and postprocessing belong in separate workflow nodes; this also applies to
pass-through Tools.

Fixed read-only paths, including references inside list/configuration files, can use ordinary directory INPUTs.
They do not require pass-through. Test with the final mount layout so those references resolve.

## Runtime Text Input Exception

File mounts normally represent files, directories, or path prefixes. Most literal command arguments belong in
`immutable_static_params` or `modifiable_static_params`, including user-adjustable values such as thread counts,
thresholds, flags, modes, and other algorithm settings.

Do not turn such static parameters into input handles. A `value:` file-mount input is allowed only for a pure-text
argument whose value must be computed by an earlier node from data available during the current workflow run. The
need to edit a value between runs does not make it runtime-produced. When this exception genuinely applies, read
`references/runtime-value-input.md` in addition to the selected output-pattern reference.

## Workflow

1. Inspect the test-run command, input mapping, observed output files, and notes about modified inputs together.
2. Select the first matching pattern from the table; shared workspace/path identity takes precedence over the
   ordinary output shape.
3. Call `read_skill_resource` with this skill name and the corresponding output-pattern reference path.
4. If output names or command arguments depend on an INPUT filename, also read
   `references/input-path-properties.md`, even when the primary pattern is Output prefix or another pattern.
5. If a proposed non-file input can only come from an upstream runtime computation, also read
   `references/runtime-value-input.md`; otherwise keep literal arguments in the appropriate static-parameter field.
6. Follow the selected references together with the agent's file-mount and command-setting requirements.
7. Treat names and concrete files in references as illustrative. Give each INPUT a stable, unique logical name and bind it explicitly during validation; outputs must reflect the current command's actual behavior.

Do not load unrelated pattern references speculatively. If validation shows that the initial classification was wrong,
reclassify the tool and then read the correct reference.
