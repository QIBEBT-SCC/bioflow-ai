---
name: tool-configuration
description: Generate a BioFlow Docker tool configuration from a real execution report. Use when classifying a tool's input/output behavior and producing its file mounts, parameters, and command template.
---

# Tool Configuration

Use this skill for every tool-configuration task. First classify the tool by how its primary output is produced, then
read exactly one matching reference before configuring the tool.

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

## Workflow

1. Inspect the test-run command, input mapping, and observed output files together.
2. Select one primary output pattern from the table.
3. Call `read_skill_resource` with this skill name and the corresponding reference path.
4. Follow that example together with the agent's file-mount and command-setting requirements.
5. Treat names and concrete files in references as illustrative. Input names must still match the current test-run
   report, and outputs must reflect the current command's actual behavior.

Do not load multiple pattern references speculatively. If validation shows that the initial classification was wrong,
reclassify the tool and then read the correct reference.
