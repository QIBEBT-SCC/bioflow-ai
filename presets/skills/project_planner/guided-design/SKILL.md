---
name: guided-design
description: Use this skill when the user wants the agent to design, revise, extend, or operationalize a bioinformatics workflow using user-provided guidance, such as a preliminary plan, reference papers, existing protocols, tool preferences, prior results, constraints, or partial workflow steps.
---

# Guided Design Workflow

## Workflow

You must strictly follow the interactive workflow below:

1. Read the provided literature excerpt.
   - If the user provides a paper, use `view_pdf` to read its full text or the relevant Methods paragraphs.
   - If the user’s plan or original paper mentions a tool you are unfamiliar with, use `web_run` to search for information about that tool. Perform an online search only when additional information is genuinely necessary and the plan cannot otherwise be written. Do not search merely to improve or supplement the information.
2. Based on the information collected, write the analysis plan to `PLAN.md` with `apply_patch`, then request approval with `confirm_plan`.

## Fidelity Principle

In general, follow the principles below when designing the plan:

- Faithfully follow the tools and parameters described in the literature. Do not replace or “upgrade” tools merely because newer alternatives exist.
- For steps not mentioned in the paper, add only the minimum scientifically necessary default settings and avoid overdesigning.

## Tool Selection Constraints

When any of the following situations applies, you may replace or modify the tool selection in the original reference:

- Tools requiring a graphical user interface (GUI) must not be used.
- Tools that depend on a specific directory structure or file-naming convention to run (e.g., CellRanger, HiC-Pro) must not be used.
- Do not directly use the R, Python, Bash, or Perl scripts from third-party pipelines. For the relevant functions, choose suitable tools or rewrite the scripts yourself.

If the paper/plan uses such a tool, replace it with the most functionally similar command-line-compatible alternative, and document this change in the **Workflow Guide**.
