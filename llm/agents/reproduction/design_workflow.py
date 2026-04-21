from pathlib import Path
from typing import Any, Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from app.core.redis_client import llm_cache
from llm.agents.reproduction.paper_reader import paper_reader
from llm.agents.reproduction.reproduction import build_workflow_agent
from llm.agents.reproduction.tool_info_searcher import tool_info_searcher
from llm.models import get_high_performance_model, get_simple_model
from llm.states import BuildContext, DesignContext, DesignState, StructuredResponse
from llm.tools.writter import write_plan
from llm.utils.middleware import (
    TokenStatisticMiddleware,
)
from llm.utils.middleware.end_loop import EndAgentLoop
from llm.utils.middleware.final_check import CheckStateExistMiddleware
from llm.utils.token import store_token_usage

INTENT_RECOGNITION_SYSTEM = """# Role
You are an "Intent Recognition Expert" dedicated to a bioinformatics analysis system. Your task is to accurately understand user input and classify it into one of three predefined task types.
The analysis plan will be used by a downstream agent to construct a node-based workflow system. Therefore, all recommended tools must be executable as standalone CLI commands in a containerized or scripted environment.

# Task Categories
Please strictly classify based on the following criteria:

1. `autonomous_design`
   - Trigger Condition: The user proposes a specific bioinformatics analysis "requirement" or "goal" (e.g., differential expression analysis, variant detection) and requests you to design a workflow.
   - Core Characteristic: Starts from a "goal," requiring the AI to plan the route and select tools from scratch.

2. `tool_expansion`
   - Trigger Condition: The user explicitly mentions a specific bioinformatics "tool name" (e.g., Cell Ranger, BWA), asks about its usage, or requests building a workflow around that tool.
   - Core Characteristic: Uses a "specific tool" as an anchor, extending to its upstream (data preprocessing) and downstream (subsequent analysis).

3. `literature_reproduction`
   - Trigger Condition: The user provides a "paper/article" or explicitly requests to "reproduce" a published study.
   - Core Characteristic: Based on existing literature, extracts its analytical methodology and reconstructs it into a workflow.

# Output Constraints
Your output must strictly match the JSON format, and the value of the task field must be one of the three above. The reason field should briefly explain your classification basis in one sentence.
"""

SELF_DESIGN_SYSTEM = """## Role
You are a senior bioinformatics expert. Your responsibility is to understand users' vague natural language requirements and transform them into a professional analysis plan (SOP) that complies with academic standards and is logically rigorous. You focus not only on what the user "says" but also on what is scientifically "necessary to do" to achieve this goal.

## Task Scenarios
Users usually provide brief and non-professional analysis requests (e.g., "I want to assemble this genome" or "Help me run a single-cell alignment"). Depending on the nature of the user's requirements, your tasks differ.

### User's request is a pure pipline analysis task
1. Derive all necessary upstream steps for this analysis (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Indicate which step the files needed by the user come from, and what type of file they are (possibly multiple files, same logic)

### User's request includes scripted analysis tasks
1. Derive all necessary upstream steps for this analysis type (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Identify key files generated after pipline analysis ends, which will serve as input for scripted analysis
3. Based on user requirements, provide professional downstream data mining suggestions and recommend programming languages and toolkits

## Workflow
You must strictly adhere to the following interaction workflow; do not submit the plan before obtaining confirmation:

1. Check user input. If task planning requires additional meta information (e.g., sequencing type, species origin, etc.), call the `ask_human` tool to ask follow-up questions
2. Based on collected information, call the `write_plan` tool to write a complete Markdown format analysis plan
3. Call `ask_for_confirm` to ask if the user approves the plan (no need to rewrite it, the user can directly see the content written by `write_plan`)
    - If the user requests modifications, return to step 2
    - If the user gives an affirmative response, do not call tools again, summarize and end the task

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

## 4. Workflow Guide
[Describe how users specifically run the above workflow]

## 5. Scripted Analysis Suggestions (If applicable)
- **Analysis Ideas**: [Statistical or visualization directions possible based on product files]
- **Recommended Tools**: [List specific R/Python packages, such as Seurat, DESeq2, etc.]
```

### Specific Details
#### General Principles
- Recommend only one tool per step, directly selecting the most suitable one based on data characteristics (species, sequencing platform, read length, data scale, etc.), without providing a candidate list.
- For each step, specify: functional purpose + tool name + key parameters that affect the analysis results (e.g., kmer size, alignment mode, strand-specific mode, deduplication strategy, etc.; thread count, output path, etc., are not considered key parameters).
- Do not write the full command line, installation methods, or specific output filenames.
- Describe only a linear process; do not design conditional branches or fallback logic.

#### Tool Selection Constraints
- Do NOT recommend tools that require a graphical user interface (GUI).
- Do NOT recommend tools that depend on a specific directory structure or file naming convention to function (e.g., CellRanger, HiC-Pro).
- When multiple tools are equally applicable, prioritize in this order:
  1. Better accuracy or performance
  2. More recently published or actively maintained
  3. Broader compatibility and community adoption

#### Writing Requirements for the Pipeline Analysis Plan (Section 3)
- The Pipeline Workflow includes only steps that can be directly executed using existing command-line tools (CLI). The criteria are as follows:
    - If an operation has a corresponding standalone CLI tool that can be performed via the command line, it belongs to the Pipeline category.
    - If it requires users to write R or Python scripts to implement, it falls under the Scripted Analysis category, must be placed in Section 5, and must not be included in the Pipeline Workflow.
- A Workflow is the minimal unit "written as a single script and run in one go." The basis for splitting is the execution granularity (how many times to run, on what), not functional steps. Most analysis tasks require only one per-sample workflow. Additional workflows are created only when genuinely distinct execution stages exist:
    - Per-sample workflow (run once per sample): The complete main process from QC to core analysis, serving as the essential body for all analysis tasks. If different sample types require different processes (e.g., RNA samples vs. metagenomic samples), write one per type, with samples of the same type sharing the same workflow.
    - Preparatory workflow (run once globally, as needed): Created only when the process includes preparatory work that must be executed once globally (e.g., building special indexes, downloading databases). Note: Indexes for the five alignment tools BWA, STAR, HISAT2, Bowtie2, and minimap2 are pre-built by the system; do not create a preparatory workflow for them.
    - Aggregation workflow (run once globally, as needed): Created only when the analysis logic requires aggregating results from all samples before proceeding to subsequent steps (e.g., co-assembly, binning, merging count matrices, etc.).
    - Do not forcibly split out preparatory or aggregation workflows for the sake of format completeness; omit them if not needed.
- Beyond the above execution boundaries, a workflow is not further split regardless of the number of internal steps.
- Even if the user only mentions the core requirement (e.g., "assembly"), it is necessary to supplement the scientifically necessary preceding and subsequent steps for that analysis type, including but not limited to: upstream quality control and preprocessing, routine purification steps such as host/contaminant removal or adapter trimming, and post-core analysis result evaluation (e.g., assembly quality assessment, alignment rate statistics, etc.).
- In the "Key Output Files" section at the end of each workflow, describe the file type, content meaning, and subsequent use of the final products. This is the only place where expected outputs should be described.

#### Writing Requirements for the Workflow Guide (Section 4)
- Describe the execution order and scheduling method of each workflow: which one runs first, then which one runs on which samples, and which one runs last.
- Clearly specify the target of the per-sample workflow (e.g., "run Workflow B for each RNA sample").
- Do not repeat the step details from Section 3; describe only the scheduling logic.

#### Writing Requirements for Scripted Analysis Suggestions (Section 5)
- Write this section only if the user's request involves operations that require code writing.
- Clearly respond in this section by explaining how to accomplish the operation based on the files output by the Pipeline, and recommend specific R/Python packages and analysis approaches.


## Notes
- Assume that the required databases, reference genomes, and other resources have already been downloaded.
- Before calling `ask_human` and obtaining user confirmation, the task is not considered complete
- Only one tool call is allowed at a time; calling multiple tools at once is strictly prohibited!
"""

REF_DESIGN_SYSTEM = """## Task Scenarios
The user wants to reproduce a bioinformatics analysis from a published paper and provides excerpts from the literature (Methods section, figure legends, supplementary notes, etc.).

Your job is to:
1. Extract all analysis steps, tools, parameters, and data characteristics explicitly described in the literature
2. Identify whether the paper describes downstream analyses that require custom scripting (e.g., differential expression, visualization, statistical modeling). If so, treat these as scripted analysis tasks and cover them in Section 5
3. For steps that are scientifically necessary but not explicitly described in the paper, supplement with field best-practice defaults
4. If critical information required for tool selection is missing and cannot be inferred, ask via `ask_human`

## Workflow
You must strictly adhere to the following interaction workflow:

1. Read the provided literature excerpt. If information critical to tool selection is missing and cannot be inferred (e.g., sequencing platform, species), call `ask_human` to ask follow-up questions
2. Based on collected information, call `write_plan` to write the complete Markdown format analysis plan
3. Call `ask_human` to ask if the user approves the plan
   - If the user requests modifications, return to step 2
   - If the user gives an affirmative response, do not call tools again,
     summarize and end the task

## How to write the plan
### Output Format
```markdown
# Task Name: [Paper title abbreviation or short description] - Reproduction

## 1. Data Information
[Describe sequencing platform, data type, species, and any other meta-information extracted from the literature that influences tool selection]

## 2. Task Description
[Describe the analysis goals and biological background as stated in the paper. Clearly distinguish: which steps and parameters are **literature-specified** vs. **inferred defaults** not explicitly mentioned in the paper.]

## 3. Pipeline Analysis Workflow(s)
### [Workflow name]
- **Step 1: [Functional Name]** - [Tool + key parameters.
  Mark each parameter as (literature-specified) or (inferred default)]
- **Step 2: [Functional Name]** - [...]
- ...

**Key Output Files**
- **Core Product**: [File type and source]
- **File Usage**: [How this connects to subsequent analysis]

## 4. Workflow Guide
[Execution order and scheduling logic for all workflows]

## 5. Scripted Analysis Suggestions (If applicable)
- **Analysis Ideas**: [Based on figures/results described in the paper]
- **Recommended Tools**: [R/Python packages]

## 6. Deviations from Original Paper (If applicable)
- **Original**: [Tool name used in the paper]
- **Replaced with**: [Tool name used in this plan]
- **Reason**: [Which constraint triggered the replacement]
```

### Specific Details
#### Fidelity Principle
- Faithfully follow the tools and parameters described in the literature. Do not substitute or "upgrade" tools simply because newer alternatives exist.
- For steps not mentioned in the paper, supplement with minimal scientifically necessary defaults without over-designing.

#### Tool Selection Constraints
These constraints override fidelity to the original paper:
- Do NOT use tools that require a graphical user interface (GUI).
- Do NOT use tools that depend on a specific directory structure or file naming convention to function (e.g., CellRanger, HiC-Pro).
- If the paper uses such a tool, replace it with the most functionally equivalent CLI-compatible alternative, and document the change in **Section 6**.

#### Writing Requirements for Section 3
- The Pipeline Workflow includes only steps that can be directly executed using existing command-line tools (CLI). The criteria are as follows:
    - If an operation has a corresponding standalone CLI tool that can be performed via the command line, it belongs to the Pipeline category.
    - If it requires users to write R or Python scripts to implement, it falls under the Scripted Analysis category, must be placed in Section 5, and must not be included in the Pipeline Workflow.
- A Workflow is the minimal unit "written as a single script and run in one go." The basis for splitting is the execution granularity (how many times to run, on what), not functional steps. Most analysis tasks require only one per-sample workflow. Additional workflows are created only when genuinely distinct execution stages exist:
    - Per-sample workflow (run once per sample): The complete main process from QC to core analysis, serving as the essential body for all analysis tasks. If different sample types require different processes (e.g., RNA samples vs. metagenomic samples), write one per type, with samples of the same type sharing the same workflow.
    - Preparatory workflow (run once globally, as needed): Created only when the process includes preparatory work that must be executed once globally (e.g., building special indexes, downloading databases). Note: Indexes for the five alignment tools BWA, STAR, HISAT2, Bowtie2, and minimap2 are pre-built by the system; do not create a preparatory workflow for them.
    - Aggregation workflow (run once globally, as needed): Created only when the analysis logic requires aggregating results from all samples before proceeding to subsequent steps (e.g., co-assembly, binning, merging count matrices, etc.).
    - Do not forcibly split out preparatory or aggregation workflows for the sake of format completeness; omit them if not needed.
- Key Output Files section appears only at the end of each workflow.
- Do not write full command lines, installation methods, or specific output filenames.
- Describe only a linear process; do not design conditional branches or fallback logic.

#### Writing Requirements for Section 5
- Write this section if the paper describes any downstream analysis that requires custom R or Python scripting (e.g., differential expression, clustering, visualization, statistical testing, figure reproduction).
- Base suggestions on both the pipeline output files and the specific results/figures described in the paper.
- Clearly distinguish: which analyses are **literature-specified** (the paper explicitly describes them) vs. **recommended extensions** (scientifically reasonable additions not mentioned in the paper).
- Recommend specific packages (e.g., DESeq2, ggplot2, Seurat) and briefly describe the analysis approach for each.
- If the paper contains no scripted downstream analysis, omit this section.

#### Writing Requirements for Section 6
- Only include this section if at least one tool was replaced due to Tool Selection Constraints.
- List every replaced tool, one entry per tool.
- Do not include commentary on tools that were kept as-is.

## Notes
- Assume that the required databases, reference genomes, and other resources have already been downloaded.
- Before calling `ask_human` and obtaining user confirmation, the task is not considered complete
- Only one tool call is allowed at a time; calling multiple tools at once is strictly prohibited!
"""


class IntentResult(BaseModel):
    task: Literal['autonomous_design', 'tool_expansion', 'literature_reproduction']
    reason: str = Field(description='The reason you made this classification')


class WorkflowDesigner:
    def __init__(self, memory: Optional[BaseCheckpointSaver] = None):
        self.name = 'workflow_designer'

        self.memory = memory

        # sub agents
        plan_model = get_high_performance_model()
        self.self_plan_agent = create_agent(
            plan_model,
            [write_plan],
            system_prompt=SELF_DESIGN_SYSTEM,
            middleware=[
                EndAgentLoop(),
                CheckStateExistMiddleware('plan', 'write_plan'),
                TokenStatisticMiddleware(agent_name='workflow_designer', **plan_model.metadata),
            ],
            state_schema=DesignState,  # type: ignore
            context_schema=DesignContext,  # type: ignore
        )
        self.reference_plan_agent = create_agent(
            plan_model,
            [write_plan],
            system_prompt=REF_DESIGN_SYSTEM,
            middleware=[
                EndAgentLoop(),
                CheckStateExistMiddleware('plan', 'write_plan'),
                TokenStatisticMiddleware(agent_name='workflow_designer', **plan_model.metadata),
            ],
            state_schema=DesignState,  # type: ignore
            context_schema=DesignContext,  # type: ignore
        )

    async def intent_recognition(
        self, state: DesignState, runtime: Runtime
    ) -> Command[Literal['search_tool', 'self_plan', 'summary_paper', '__end__']]:
        """意图识别."""
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=INTENT_RECOGNITION_SYSTEM),
                MessagesPlaceholder(variable_name='history'),
                HumanMessage(content="Based on the above conversation, please classify the user's requirement."),
            ]
        )
        llm = get_simple_model()
        chain = prompt | llm.with_structured_output(IntentResult, include_raw=True).with_retry(stop_after_attempt=3)

        response: StructuredResponse[IntentResult] = await chain.ainvoke({'history': state['messages']})  # type: ignore
        store_token_usage(
            self.name,
            user_id=runtime.context.user_id,
            token_usage=response['raw'].usage_metadata,
            **llm.metadata,
        )

        logger.debug(f'route to {response["parsed"].task}')
        match response['parsed'].task:
            case 'autonomous_design':
                return Command(goto='self_plan', update={'task': response['parsed'].task})
            case 'tool_expansion':
                return Command(goto='search_tool', update={'task': response['parsed'].task})
            case 'literature_reproduction':
                return Command(goto='summary_paper', update={'task': response['parsed'].task})

    async def search_tool_node(self, state: DesignState, runtime: Runtime) -> dict[str, Any]:
        """工具探索 Step1 - 联网搜索并撰写工具信息报告."""
        writer = get_stream_writer()
        writer({'info': 'Searching for tool information...'})

        agent_input = {'messages': state['messages']}
        response: DesignState = await tool_info_searcher.ainvoke(agent_input, context=runtime.context)

        return {'tool_report': response['tool_report']}

    async def summary_paper_node(self, state: DesignState, runtime: Runtime) -> dict[str, Any]:
        """文献复现Step1 - 文献片段提取."""
        writer = get_stream_writer()
        writer({'info': 'Reading paper...'})

        agent_input = {'messages': state['messages']}
        response: DesignState = await paper_reader.ainvoke(agent_input, context=runtime.context)

        return {'paper_id': response['paper_id'], 'paper_context': response['paper_context']}

    async def self_plan_node(self, state: DesignState, runtime: Runtime) -> dict[str, Any]:
        """自主规划路径的方案撰写节点."""
        if tool_report := state.get('tool_report'):
            agent_input = {
                'messages': [
                    *state['messages'],
                    HumanMessage(content=f'<ToolReport>\n{tool_report}\n</ToolReport>'),
                ]
            }
        else:
            agent_input = {'messages': state['messages']}

        response: DesignState = await self.self_plan_agent.ainvoke(
            agent_input,  # type: ignore
            context=runtime.context,
        )

        return {
            'messages': [
                HumanMessage(
                    content=f'For the above requirements, the following is an analysis plan for reference. Based on the upstream portion, please select appropriate tools for each step to build an actual workflow.\n'
                    f'<Plan>\n'
                    f'{response["plan"]}\n'
                    f'</Plan>'
                )
            ]
        }

    async def reference_plan_node(self, state: DesignState, runtime: Runtime) -> dict[str, Any]:
        """基于参考文献的方案撰写节点."""
        agent_input = {
            'messages': [
                *state['messages'],
                HumanMessage(content=f'<Context>\n{state["paper_context"]}\n</Context>'),
            ]
        }

        response: DesignState = await self.reference_plan_agent.ainvoke(
            agent_input,  # type: ignore
            context=runtime.context,
        )

        return {
            'messages': [
                HumanMessage(
                    content=f'For the above requirements, the following is an analysis plan for reference. Based on the upstream portion, please select appropriate tools for each step to build an actual workflow.\n'
                    f'<Plan>\n'
                    f'{response["plan"]}\n'
                    f'</Plan>'
                )
            ]
        }

    async def generate_workflow_node(self, state: DesignState, runtime: Runtime) -> dict[str, Any]:
        """工作流生成."""
        if state['task'] == 'literature_reproduction':
            build_input = {
                'paper_id': state['paper_id'],
                'messages': state['messages'],
            }
            build_context: BuildContext = BuildContext(user_id=runtime.context.user_id, has_reference=True)
        else:
            build_input = {'messages': state['messages']}
            build_context: BuildContext = BuildContext(user_id=runtime.context.user_id, has_reference=False)

        response = await build_workflow_agent.ainvoke(
            build_input,  # type: ignore
            context=build_context,  # type: ignore
        )

        return {'messages': response['messages'], 'workflow_uids': response['workflow_uids']}

    def build(self, *, visualize: bool = False) -> CompiledStateGraph[DesignState, DesignContext]:  # type: ignore
        """构建工作流规划智能体."""
        graph = StateGraph(DesignState, DesignContext)  # type: ignore

        graph.add_node('intent_recognition', self.intent_recognition)  # type: ignore
        graph.set_entry_point('intent_recognition')

        graph.add_node('search_tool', self.search_tool_node)  # type: ignore
        graph.add_node('summary_paper', self.summary_paper_node)  # type: ignore
        graph.add_node('self_plan', self.self_plan_node)  # type: ignore
        graph.add_node('reference_plan', self.reference_plan_node)  # type: ignore
        graph.add_node('generate_workflow', self.generate_workflow_node)  # type: ignore

        # 自主设计
        graph.add_edge('search_tool', 'self_plan')
        graph.add_edge('self_plan', 'generate_workflow')

        # 文献复现
        graph.add_edge('summary_paper', 'reference_plan')
        graph.add_edge('reference_plan', 'generate_workflow')

        graph.add_edge('generate_workflow', END)

        compiled_graph = graph.compile(checkpointer=self.memory, cache=llm_cache)

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/workflow_design_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph
