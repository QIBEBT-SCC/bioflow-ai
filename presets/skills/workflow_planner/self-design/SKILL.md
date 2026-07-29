---
name: self-design
description: Use this skill when the user wants the agent to autonomously design a bioinformatics workflow from scratch, with little or no pre-existing plan, protocol, reference workflow, tool list, or literature-derived analytical route.
---

# Self Design Workflow

## Task Scenarios

Users usually provide brief and non-professional analysis requests (e.g., "I want to assemble this genome" or "Help me run a single-cell alignment"). Depending on the nature of the user's
requirements, your tasks differ.

### User's request is a pure pipline analysis task

1. Derive all necessary upstream steps for this analysis (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Indicate which step the files needed by the user come from, and what type of file they are (possibly multiple files, same logic)

### User's request includes scripted analysis tasks

1. Derive all necessary upstream steps for this analysis type (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Identify key files generated after pipline analysis ends, which will serve as input for scripted analysis
3. Based on user requirements, provide professional downstream data mining suggestions and recommend programming languages and toolkits

Do not classify every use of Python as downstream analysis. A bounded, deterministic single-file transformation whose output is required by a later pipeline node may be included in the pipeline as a `Python Code Node` when Bash is not a clear solution. Statistical analysis, clustering, visualization, data mining, and biological interpretation remain downstream scripted analysis.

## Workflow

You must strictly adhere to the following interaction workflow; do not submit the plan before obtaining confirmation:

1. Check user input. If task planning requires additional meta information (e.g., sequencing type, species origin, etc.), call the `ask_human` tool to ask follow-up questions
2. Based on collected information, call the `write_plan` tool to write a complete Markdown format analysis plan
3. Call `ask_human` to ask if the user approves the plan (no need to rewrite it, the user can directly see the content written by `write_plan`)
    - If the user requests modifications, return to step 2
    - If the user gives an affirmative response, do not call tools again, summarize and end the task

## Tool Selection Constraints

- Do NOT recommend tools that require a graphical user interface (GUI).
- Do NOT recommend tools that depend on a specific directory structure or file naming convention to function (e.g., CellRanger, HiC-Pro).
- When multiple tools are equally applicable, prioritize in this order:
    1. Better accuracy or performance
    2. More recently published or actively maintained
    3. Broader compatibility and community adoption
