---
name: tool-configuration
description: Generate a BioFlow Docker tool configuration from a real execution report. Use when classifying a tool's input/output behavior and producing its file mounts, parameters, and command template.
---

# Tool Configuration

Use this skill for every tool-configuration task. First classify the tool by how its primary output is produced, then
read exactly one matching output-pattern reference before configuring the tool. If, and only if, a non-file text
parameter must receive a value dynamically produced by an upstream node during workflow execution, also read
`references/runtime-value-input.md`.

## Choose the Output Pattern

Apply the first matching pattern in this order:

| Pattern | Use when | Reference |
|---|---|---|
| In-place modification | The command changes an input file directly instead of producing an independent result path | `references/in-place-modification.md` |
| Output prefix | An output argument is a path prefix and the tool creates multiple sibling artifacts whose names begin with that prefix | `references/prefix-output.md` |
| Standard-output redirection | The primary result exists only because stdout is redirected to an output file | `references/stdout-output.md` |
| Standard explicit output | The command writes its primary result to an explicit file or directory path | `references/simple-output.md` |

Stderr or ordinary log redirection does not make a tool a standard-output tool. Use the standard-output pattern only
when stdout itself carries a primary result.

## Runtime Text Input Exception

File mounts normally represent files, directories, or path prefixes. Most literal command arguments belong in
`immutable_static_params` or `modifiable_static_params`, including user-adjustable values such as thread counts,
thresholds, flags, modes, and other algorithm settings.

Do not turn such static parameters into input handles. A `value:` file-mount input is allowed only for a pure-text
argument whose value must be computed by an earlier node from data available during the current workflow run. The
need to edit a value between runs does not make it runtime-produced. When this exception genuinely applies, read
`references/runtime-value-input.md` in addition to the selected output-pattern reference.

## Workflow

1. Inspect the test-run command, input mapping, and observed output files together.
2. Select one primary output pattern from the table.
3. Call `read_skill_resource` with this skill name and the corresponding output-pattern reference path.
4. If a proposed non-file input can only come from an upstream runtime computation, also read
   `references/runtime-value-input.md`; otherwise keep literal arguments in the appropriate static-parameter field.
5. Follow the selected references together with the agent's file-mount and command-setting requirements.
6. Treat names and concrete files in references as illustrative. Input names must still match the current test-run
   report, and outputs must reflect the current command's actual behavior.

Do not load multiple pattern references speculatively. If validation shows that the initial classification was wrong,
reclassify the tool and then read the correct reference.
