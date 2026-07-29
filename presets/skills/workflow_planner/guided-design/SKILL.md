---
name: guided-design
description: Use this skill when the user wants the agent to design, revise, extend, or operationalize a bioinformatics workflow using user-provided guidance, such as a preliminary plan, reference papers, existing protocols, tool preferences, prior results, constraints, or partial workflow steps.
---

# Guided Design Workflow

## Workflow

You must strictly adhere to the following interaction workflow:

1. Read the provided literature excerpt. 
   - If the user provides a paper, the full text of the paper can be requested using the `read_paper` tool.  
   - If the user's plan or the original paper mentions a tool that is not within your own knowledge, information about that tool can be searched using the `research_tool`.
2. Based on collected information, call `write_plan` to write the complete Markdown format analysis plan
3. Call `ask_human` to ask if the user approves the plan
    - If the user requests modifications, return to step 2
    - If the user gives an affirmative response, do not call tools again,
      summarize and end the task

## Fidelity Principle

- Faithfully follow the tools and parameters described in the literature. Do not substitute or "upgrade" tools simply because newer alternatives exist.
- For steps not mentioned in the paper, supplement with minimal scientifically necessary defaults without over-designing.

## Tool Selection Constraints

These constraints override fidelity to the original paper/plan:

- Do NOT use tools that require a graphical user interface (GUI).
- Do NOT use tools that depend on a specific directory structure or file naming convention to function (e.g., CellRanger, HiC-Pro).
- If the paper/plan uses such a tool, replace it with the most functionally equivalent CLI-compatible alternative, and document the change in **Section 6**.

## Additional Writing Requirements for Section 5

- Keep a simple deterministic Python file transformation in Section 3 when its output must feed a later pipeline node and Bash is not a clear solution. Identify it as a `Python Code Node`; this is workflow glue, not downstream analysis.
- Write this section if the paper/plan describes any downstream analysis that requires custom R or Python scripting (e.g., differential expression, clustering, visualization, statistical testing,
  figure
  reproduction).
- Base suggestions on both the pipeline output files and the specific results/figures described in the paper.
- Clearly distinguish: which analyses are **literature-specified** (the paper explicitly describes them) vs. **recommended extensions** (scientifically reasonable additions not mentioned in the
  paper).
- Recommend specific packages (e.g., DESeq2, ggplot2, Seurat) and briefly describe the analysis approach for each.
- If the paper contains no scripted downstream analysis, omit this section.
