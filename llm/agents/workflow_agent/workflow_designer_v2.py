from typing import TYPE_CHECKING, Optional, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from llm.agents.workflow_agent.paper_reader import paper_reader
from llm.agents.workflow_agent.tool_info_searcher import tool_info_searcher
from llm.context import BaseContext, ProjectContext
from llm.models import get_high_performance_model
from llm.states import PlannerState
from llm.tools.writter import write_plan
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TokenStatisticMiddleware,
)
from llm.utils.middleware.end_loop import EndAgentLoop
from llm.utils.middleware.final_check import CheckStateExistMiddleware

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata


PLANNER_SYSTEM = """## Role
You are a senior bioinformatics expert. Your responsibility is to understand users' vague natural language requirements and transform them into a professional analysis plan (SOP) that complies with academic standards and is logically rigorous. You focus not only on what the user "says" but also on what is scientifically "necessary to do" to achieve this goal.

## Mission and Skill-Driven Workflow

Your primary deliverable is a complete analysis plan written with the `write_plan` tool.
Before drafting or updating the plan, choose the most appropriate available skill according to the user's request and call `view_skill` to read its detailed instructions.
Then complete the task by following that skill's workflow together with the plan-writing requirements below.
If essential planning information is missing and cannot be inferred responsibly, ask the user with `ask_human`; after writing the plan, ask the user to confirm it.

## How to write plan

### Output Format

Content written via `write_plan` must be well-structured Markdown text, formatted as follows:

```markdown
# Task Name: [Short and intuitive name]

## 1. Data Information

[Describe the meta-information of the raw data that may influence the selection of subsequent analysis tools, such as platform, type, and species.]

## 2. Task Description

[Describe analysis goals, biological background, and known data parameters]

## 3. Pipeline Analysis Workflow(s)

### [Workflow name]

- **Step 1: [Functional Name]** - [What tool should be used to complete this step?]
- **Step 2: [Functional Name]** - [...]
- ... (Complete all necessary steps until result files available for analysis are produced)

**Key Output Files**

- **Core Product**: [No need to specify exact file names, just point out file type and source]
- **File Usage**: [Explain how this file connects to subsequent analysis]

### [Workflow name2]

(If there are indeed multiple workflows)
...

## 4. Workflow Guide

[Describe how users specifically run the above workflow]

## 5. Scripted Analysis Suggestions (If applicable)

- **Analysis Ideas**: [Statistical or visualization directions possible based on product files]
- **Recommended Tools**: [List specific R/Python packages, such as Seurat, DESeq2, etc.]
```

### General Principles

For each step, specify: functional purpose + tool name + key parameters that affect the analysis results (e.g., kmer size, alignment mode, strand-specific mode, deduplication strategy, etc.; thread
count, output path, etc., are not considered key parameters).
Do not write the full command line, installation methods, or specific output filenames.
Describe only a linear process; do not design conditional branches or fallback logic.

### Writing Requirements for the Pipeline Analysis Plan (Section 3)

- The Pipeline Workflow includes only steps using existing command-line tools (CLI). The criteria are as follows:
    - If an operation has a corresponding standalone CLI tool that can be performed via the command line, it belongs to the Pipeline category.
    - If it requires users to write R or Python scripts to implement, it falls under the Scripted Analysis category, must be placed in Section 5, and must not be included in the Pipeline Workflow.
- A Workflow is the minimal unit "written as a single script and run in one go." The basis for splitting is the execution granularity (how many times to run, on what), not functional steps. Most analysis tasks require only one per-sample workflow. Additional workflows are created only when genuinely distinct execution stages exist:
    - Preparatory workflow (run once globally, as needed): Created only when the process includes preparatory work that must be executed once globally (e.g., building indexes, generate chrom size).
      Note: Indexes for alignment tools BWA, STAR, HISAT2, Bowtie2, and minimap2 are pre-built by the system; no need to create a preparatory workflow for them.
    - Per-sample workflow (run once per sample): The complete main process from QC to core analysis, serving as the essential body for all analysis tasks. If different sample types require different
      processes (e.g., RNA samples vs. metagenomic samples), write one per type, with samples of the same type sharing the same workflow.
    - Aggregation workflow (run once globally, as needed): Created only when the analysis logic requires aggregating results from all samples before proceeding to subsequent steps (e.g., co-assembly,
      binning, merging count matrices, etc.).
    - Do not forcibly split out preparatory or aggregation workflows for the sake of format completeness; omit them if not needed.
- Beyond the above execution boundaries, a workflow is not further split regardless of the number of internal steps.
- In the "Key Output Files" section at the end of each workflow, describe the file type, content meaning, and subsequent use of the final products. This is the only place where expected outputs should be described.

### Writing Requirements for the Workflow Guide (Section 4)

- Describe the execution order and scheduling method of each workflow: which one runs first, then which one runs on which samples, and which one runs last.
- Clearly specify the target of the per-sample workflow (e.g., "run Workflow B for each RNA sample").
- Do not repeat the step details from Section 3; describe only the scheduling logic.

#### Writing Requirements for Scripted Analysis Suggestions (Section 5)

- Write this section only if the user's request involves operations that require code writing (commonly referred to as *downstream analysis* in bioinformatics).
- Clearly respond in this section by explaining how to accomplish the operation based on the files output by the Pipeline, and recommend specific R/Python packages and analysis approaches.

## Notes

- Assume that the required databases, reference genomes, and other resources have already been downloaded.
- Before calling `ask_human` and obtaining user confirmation, the task is not considered complete.
"""

PLANNER_LLM = get_high_performance_model()
metadata = cast('LLMMetadata', PLANNER_LLM.metadata)


@tool('research_tool', parse_docstring=True)
async def research_tool(request: str, runtime: ToolRuntime[ProjectContext]) -> str:
    """Delegate bioinformatics tool research to a dedicated web-research agent.

    Use this for the tool-expansion scenario: the sub-agent searches the official GitHub
    repository and documentation and returns a structured report covering installation and
    basic usage. The full web pages it reads stay inside the sub-agent.

    Args:
        request: The tool name plus what you need to know (installation, usage, key
            parameters). If the tool is a large suite, state which sub-workflow matters.
    """
    # pyrefly: ignore [no-matching-overload]
    response = await tool_info_searcher.ainvoke(
        {'messages': [HumanMessage(content=request)]},
        context=BaseContext(user_id=runtime.context.user_id),
    )
    return response.get('tool_report') or 'The tool-research agent produced no report.'


@tool('read_paper', parse_docstring=True)
async def read_paper(request: str, runtime: ToolRuntime[ProjectContext]) -> str:
    """Delegate literature reading to a dedicated paper-processing agent.

    Use this for the literature-reproduction scenario: the sub-agent reads the paper and
    extracts the relevant Methods / Supplementary excerpts verbatim, returning only the
    distilled methods content. The full paper text stays inside the sub-agent.

    Args:
        request: Which paper to read — include the file id (e.g.
            ``documents/75/f5/4c/s41467-022-35735-y.pdf``) — and which analysis
            pipeline/methods to extract.
    """
    # pyrefly: ignore [no-matching-overload]
    response = await paper_reader.ainvoke(
        {'messages': [HumanMessage(content=request)]},
        context=BaseContext(user_id=runtime.context.user_id),
    )
    return response.get('paper_context') or 'The paper-reader agent extracted no methods content.'


def build_workflow_planner(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Compile the unified workflow-planner agent.

    A single ``create_agent`` covering all three planning scenarios (autonomous design, tool
    expansion, literature reproduction). The full tool union is exposed at once; the system
    prompt and skills tell the model which tools to use in which scenario.

    Pass a checkpointer to run it as a top-level graph with native interrupt/resume; pass
    ``None`` when embedding it as a subgraph so it inherits the parent graph's checkpointer.

    Args:
        checkpointer: Checkpointer to bind, or ``None`` to inherit from a parent graph.

    Returns:
        The compiled workflow-planner agent.
    """
    if checkpointer:
        # pyrefly: ignore [no-matching-overload]
        return create_agent(
            PLANNER_LLM,
            [
                research_tool,
                read_paper,
                write_plan,
            ],
            system_prompt=PLANNER_SYSTEM,
            middleware=[
                SkillMiddleware(
                    [
                        'SELF_DESIGN',
                        'GUIDED_DESIGN',
                    ]
                ),
                DisableParallelToolCallsMiddleware(disabled_tools={'ask_human'}),
                EndAgentLoop(),
                StatusBarMiddleware(),
                CheckStateExistMiddleware('plan', 'write_plan'),
                TokenStatisticMiddleware('planner', **metadata),
            ],
            state_schema=PlannerState,
            context_schema=ProjectContext,
            checkpointer=checkpointer,
        )
    # pyrefly: ignore [no-matching-overload]
    return create_agent(
        PLANNER_LLM,
        [
            research_tool,
            read_paper,
            write_plan,
        ],
        system_prompt=PLANNER_SYSTEM,
        middleware=[
            SkillMiddleware(
                [
                    'SELF_DESIGN',
                    'GUIDED_DESIGN',
                ]
            ),
            DisableParallelToolCallsMiddleware(disabled_tools={'ask_human'}),
            EndAgentLoop(),
            StatusBarMiddleware(),
            CheckStateExistMiddleware('plan', 'write_plan'),
            TokenStatisticMiddleware('planner', **metadata),
        ],
        state_schema=PlannerState,
        context_schema=ProjectContext,
    )


workflow_planner_agent: CompiledStateGraph = build_workflow_planner()
