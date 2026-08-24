---
name: guided-design
description: Use this skill when the user wants the agent to design, revise, extend, or operationalize a bioinformatics workflow using user-provided guidance, such as a preliminary plan, reference papers, existing protocols, tool preferences, prior results, constraints, or partial workflow steps.
---

# Guided Design Workflow

## Workflow

You must strictly follow the interactive workflow below:

1. Read the provided literature excerpt.
   - If the user provides a paper, use the `read_paper` tool to request the full text.
   - If the user’s plan or original paper mentions a tool you are unfamiliar with, use `research_tool` to search for information about that tool. Note that **this tool call is extremely costly**; perform an online search only when additional information is genuinely necessary and the document cannot otherwise be written. Do not call this tool merely to improve or supplement the information.
2. Based on the information collected, call `write_plan` to write a complete analysis plan in Markdown format.

## Fidelity Principle

In general, follow the principles below when designing the plan:

- Faithfully follow the tools and parameters described in the literature. Do not replace or “upgrade” tools merely because newer alternatives exist.
- For steps not mentioned in the paper, add only the minimum scientifically necessary default settings and avoid overdesigning.

## Tool Selection Constraints

When any of the following situations applies, you may replace or modify the tool selection in the original reference:

- Tools requiring a graphical user interface (GUI) must not be used.
- Tools that depend on a specific directory structure or file-naming convention to run (e.g., CellRanger, HiC-Pro) must not be used.

If the paper/plan uses such a tool, replace it with the most functionally similar command-line-compatible alternative, and document this change in the **Workflow Guide**.
