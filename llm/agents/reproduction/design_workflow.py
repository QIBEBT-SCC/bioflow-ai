from pathlib import Path
from typing import Any, Literal, Optional

from langchain.agents import create_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from loguru import logger
from pydantic import BaseModel, Field

from app.core.redis_client import llm_cache
from llm.agents.reproduction.reproduction import build_workflow_agent
from llm.models import get_high_performance_model, get_long_context_model, get_simple_model
from llm.states import BuildContext, DesignState, StructuredResponse, UserContext
from llm.tools.storage import view_pdf
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    TokenStatisticMiddleware,
)
from llm.utils.middleware.todo import BuildTodoMiddleware
from llm.utils.token import store_token_usage

INTENT_RECOGNITION_SYSTEM = """# Role
You are an "Intent Recognition Expert" dedicated to a bioinformatics analysis system. Your task is to accurately understand user input and classify it into one of three predefined task types.

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

### User's request is a pure upstream analysis task
1. Derive all necessary upstream steps for this analysis (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Indicate which step the files needed by the user come from, and what type of file they are (possibly multiple files, same logic)

### User's request includes downstream analysis tasks
1. Derive all necessary upstream steps for this analysis type (Quality Control, Filtering, Decontamination, Core Analysis, Result Evaluation)
2. Identify key files generated after upstream analysis ends, which will serve as input for downstream analysis
3. Based on user requirements, provide professional downstream data mining suggestions and recommend programming languages and toolkits

## Workflow
You must strictly adhere to the following interaction workflow; do not submit the plan before obtaining confirmation:

1. Check user input. If task planning requires additional meta information (e.g., sequencing type, species origin, etc.), call the `ask_human` tool to ask follow-up questions
2. Based on collected information, call the `write_plan` tool to write a complete Markdown format analysis plan
3. Call `ask_human` to ask if the user approves the plan (no need to rewrite it, the user can directly see the content written by `write_plan`)
    - If the user requests modifications, return to step 2
    - If the user gives an affirmative response, do not call tools again, summarize and end the task

## Output Format
Content written via `write_plan` must be well-structured Markdown text, formatted as follows:

```markdown
# Task Name: [Short and intuitive name]

## 1. Data Information
[Describe the meta-information of the raw data that may influence the selection of subsequent analysis tools, such as platform, type, and species.]

## 2. Task Description
[Describe analysis goals, biological background, and known data parameters]

## 3. Standardized Analysis Workflow (Upstream)
### Workflow Plan
- **Step 1: [Functional Name]** - [Detailed description of the scientific necessity and operational logic of this step]
- **Step 2: [Functional Name]** - [Describe logic...]
- ... (Complete all necessary steps until result files available for analysis are produced)

### Key Output Files
- **Core Product**: [No need to specify exact file names, just point out file type and source]
- **File Usage**: [Explain how this file connects to subsequent analysis]

## 4. Downstream Analysis Suggestions (If applicable)
- **Analysis Ideas**: [Statistical or visualization directions possible based on product files]
- **Recommended Tools**: [List specific R/Python packages, such as Seurat, DESeq2, etc.]
```

## Notes
1. In the upstream analysis workflow description, your task is only to provide the general analysis step planning; specific tools to be used will be selected by subsequent work based on your planning, **you do not need to decide on specific tools**
2. The plan must include "invisible but necessary" steps, such as: BUSCO/QUAST evaluation after assembly, repeat sequence marking after alignment, mitochondrial filtering for single cells, etc.
3. Before calling `ask_human` and obtaining user confirmation, the task is not considered complete
4. Upstream planning is for subsequent tool selection and workflow boundaries, requiring only clear logic; downstream suggestions are for the user to see and must be actionable
"""

SUMMARY_PAPER_SYSTEM = """You are a bioinformatics scholar whose work involves reading published biological literature and summarizing the information contained within.

## Output Task
Based on the type of literature, invoke the appropriate tool to read the full text and summarize it. Your output must include the following two parts:
1. Full Summary: What is the article about? What are the main research content and conclusions?
2. Bioinformatics Summary: Focus on the method section of the article, summarizing the bioinformatics analysis operations performed in the study.

## Notes
- If the uploaded literature is not related to bioinformatics, politely inform the user.
- The language of your generated summary must match the user's question language.
"""

SPLIT_WORKFLOW_SYSTEM = """You are a Pipeline Architect proficient in designing bioinformatics automation workflows. Your task is to break down literature methods into independent, automated engineering workflows.

# How to Divide Workflows?
## Core Definition: What is a "workflow"?
- **A workflow = a complete analysis pipeline for a specific type of omics data.**
- It generally covers the entire process from data preprocessing and upstream analysis to downstream biological statistical analysis.

## Workflow Separation Standards
- If the article includes multi-omics analysis, and the user does not specify a particular type, then the analysis processes for different types of data need to be extracted into different workflows.
- If two different omics datasets have completely identical analysis methods, they can be written as one. However, if even one step is different, they must be separated into two. We need to write an automated process based on your list; it's impossible to manually distinguish cases.
- Do not split "data cleaning", "normalization", "dimensionality reduction", and "enrichment analysis" into multiple tasks. If they are all performed on the same RNA-seq data, they belong to the same workflow.

## Negative Constraints (Important)
- **Strictly no fragmentation**: Strictly prohibit extracting different stages (steps) of the same omics analysis into multiple Tasks.
- **Exclude wet lab experiments**: Ignore sampling and library preparation.
- **Exclude multi-omics cross-referencing**: If the task is "integrating RNA and protein data for modeling", this cross-referencing part is not extracted; only their respective independent standard analysis chains are extracted.


# Operational Procedures
## Step 1: Initialize List
1. Call appropriate tools to obtain the full text of the literature and read it
2. Based on the rules above, first determine how many workflows the article generally contains
3. For each workflow, call `add_todos` to add two tasks, one formatted as "Initialize Workflow #n XXX Design" and the other as "Save Workflow #n XXX"

## Step 2: Fill in Actual Steps
Building on the previous step, continue calling the `add_todos` tool to insert detailed analysis steps after the initial task of each workflow
- The format for specific analysis steps is "using XXX to ...", each corresponding to an analysis step in the original text
- No need to record specific parameter details; simply state which tool is used for what
- If the original text clearly states that multiple tools can be used for a certain step, choose the one you deem most appropriate. Pointing to multiple tools is not allowed in the task.
- If the analysis data for a step comes from another workflow (generally common at the beginning of a new workflow), add a note specifying which file from which workflow is being used
- Since inserting into an existing list alters the TODO-list's index, call the `view_todos` tool to inspect the current list before populating each workflow.

# Precautions
- Only extract operations related to bioinformatics analysis; do not extract preliminary work such as file downloads and database preparation.
- Only extract the workflow for omics data analysis in the text. Do not extract wet-lab design or other data analyses (e.g., imaging).
- The to-do list is for subsequent build tasks, do not write your own decomposition workflow related content into it. If you accidentally added it, please delete it.

# Output Format
Please answer directly according to the following structure, without superfluous content:
```text
**Identification Result: [Number] independent workflows in total**

**Workflow Details:**
1. **Workflow #1 [Brief Name]**
   - **Source Data**: [Briefly describe the input, e.g., Raw Sequencing Data (FastQ)]
   - **Main Tools and Steps**: [List core toolchain, connected by arrows]
   - **Function Description**: [One-sentence summary]

2. **Workflow #2 [Brief Name]**
   - **Source Data**: [e.g., Merged Contigs or BAM Files from the previous step]
   - **Main Tools and Steps**: ...
   - **Function Description**: ...
```

The language of your generated summary must match the user's question language.
"""


class IntentResult(BaseModel):
    task: Literal['autonomous_design', 'tool_expansion', 'literature_reproduction']
    reason: str = Field(description='The reason you made this classification')


class AutonomousDesignAgent:
    def __init__(self, memory: Optional[BaseCheckpointSaver] = None, *, use_cache: bool = False) -> None:
        self.name = 'autonomous_design'

        self.tools = [wait_for_confirm, write_plan]
        self.llm = get_high_performance_model().bind_tools(self.tools, parallel_tool_calls=False)

        self.memory = memory
        self.cache = llm_cache if use_cache else None

    async def llm_node(self, state: DesignState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        """Agent node."""
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=SELF_DESIGN_SYSTEM),
                MessagesPlaceholder(variable_name='history'),
            ]
        )
        chain = prompt | self.llm

        response = await chain.ainvoke({'history': state['messages']})  # type: ignore
        store_token_usage(
            self.name,
            user_id=runtime.context.user_id,
            token_usage=response.usage_metadata,
            **self.llm.metadata,
        )

        return {'messages': [response]}

    @staticmethod
    def model_router(state: DesignState) -> Command[Literal['tools', 'model', '__end__']]:
        """处理未调用工具直接结束的情况."""
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return Command(goto='tools')

        # 实际上理论来说不会走到这里, 工具直接退出了
        if isinstance(state['messages'][-2], ToolMessage) and state['messages'][-2].name == 'ask_human':
            return Command(goto=END)

        return Command(
            goto='model',
            update={
                'messages': [
                    HumanMessage(
                        content='Call the `ask_human` tool to ask a human for information instead of outputting information directly.'
                    )
                ]
            },
        )

    def build(self) -> CompiledStateGraph:  # noqa: D102
        graph = StateGraph(state_schema=DesignState, context_schema=UserContext)  # type: ignore

        graph.add_node('model', self.llm_node)  # type: ignore
        graph.add_node('router', self.model_router)  # type: ignore
        graph.add_node('tools', ToolNode(self.tools))

        graph.add_edge('model', 'router')
        graph.add_edge('tools', 'model')

        graph.set_entry_point('model')

        return graph.compile(cache=self.cache, checkpointer=self.memory, name=self.name)


class WorkflowDesigner:
    def __init__(self, history: BaseChatMessageHistory, memory: Optional[BaseCheckpointSaver] = None):
        self.name = 'workflow_designer'

        self.history = history
        self.memory = memory

        # sub agents
        self.autonomous_design_agent = AutonomousDesignAgent(self.memory).build()

        self.summary_paper_agent = create_agent(
            get_long_context_model(),
            [view_pdf],
            system_prompt=SUMMARY_PAPER_SYSTEM,
            middleware=[TokenStatisticMiddleware(agent_name=self.name, **get_long_context_model().metadata)],
            state_schema=DesignState,
            cache=llm_cache,
        )

        self.split_workflow_agent = create_agent(
            get_long_context_model(),
            [view_pdf],
            system_prompt=SPLIT_WORKFLOW_SYSTEM,
            middleware=[
                BuildTodoMiddleware(),
                DisableParallelToolCallsMiddleware(),
                TokenStatisticMiddleware(agent_name=self.name, **get_long_context_model().metadata),
            ],
            state_schema=DesignState,
        )

    async def intent_recognition(
        self, state: DesignState, runtime: Runtime[UserContext]
    ) -> Command[Literal['autonomous_design', 'summary_paper', '__end__']]:
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

        response: StructuredResponse[IntentResult] = await chain.ainvoke(  # type: ignore
            {'history': state['messages']}
        )
        store_token_usage(
            self.name,
            user_id=runtime.context.user_id,
            token_usage=response['raw'].usage_metadata,
            **llm.metadata,
        )

        logger.debug(f'route to {response["parsed"].task}')
        match response['parsed'].task:
            case 'autonomous_design':
                return Command(goto='autonomous_design', update={'task': 'autonomous_design'})
            case 'tool_expansion':
                return Command(goto='__end__', update={'task': 'tool_expansion'})
            case 'literature_reproduction':
                return Command(goto='summary_paper', update={'task': 'literature_reproduction'})
            case _:
                return Command(goto='__end__')

    async def autonomous_design_node(self, state: DesignState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        """自主设计Step1 - 设定方案."""
        response: DesignState = await self.autonomous_design_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )

        return {'messages': response['messages'], 'workflow_description': response['workflow_description']}

    async def summary_paper_node(self, state: DesignState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        """从文献复现Step1 - 文献总结."""
        # response = await self.summary_paper_agent.ainvoke(
        #     {'messages': state['messages']},
        #     context=UserContext(user_id=runtime.context.user_id),
        # )
        # await self.history.aadd_messages([response['messages'][-1]])

        # Mocked response for demonstration
        response = {
            'paper_id': 'sample_paper_id',
            'messages': [HumanMessage(content='This is a sample paper summary.')],
        }

        return {'paper_id': response['paper_id'], 'paper_summary': response['messages'][-1].content}

    async def split_workflow_node(self, state: DesignState, runtime: Runtime[UserContext]) -> dict[str, Any]:  # type: ignore
        """文献复现Step2 - 工作流拆解."""
        response = await self.split_workflow_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )
        await self.history.aadd_messages([response['messages'][-1]])

        return {'messages': [response['messages'][-1]], 'todos': response['todos'], 'paper_id': response['paper_id']}

    async def plan_confirmation_node(
        self, state: DesignState
    ) -> Command[Literal['generate_workflow', 'split_workflow']]:
        """方案确认."""
        confirm = interrupt({'info': 'plz confirm the plan above, whether to proceed?'})

        if isinstance(confirm, bool) and confirm:
            return Command(goto='generate_workflow', update={'workflow_description': state['workflow_description']})

        match state['task']:
            case 'autonomous_design':
                return Command(goto='autonomous_design', update={'messages': [HumanMessage(content=confirm)]})
            case 'literature_reproduction':
                return Command(goto='split_workflow', update={'messages': [HumanMessage(content=confirm)]})
        return Command(goto='__end__')

    async def generate_workflow_node(self, state: DesignState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        """工作流生成."""
        if state['task'] == 'literature_reproduction':
            build_input = {
                'paper_id': state['paper_id'],
                'messages': [
                    HumanMessage(
                        content=f'The following workflows are available in the literature, please build them:\n{state["workflow_description"]}'
                    )
                ],
                'todos': state['todos'],
            }
            build_context: BuildContext = BuildContext(user_id=runtime.context.user_id, has_reference=True)
        else:
            build_input = {
                'messages': [
                    HumanMessage(
                        content=f'For the above requirements, the following is an analysis plan for reference. Based on the upstream portion, please select appropriate tools for each step to build an actual workflow.\n'
                        f'<Plan>\n'
                        f'{state["workflow_description"]}\n'
                        f'</Plan>'
                    )
                ],
            }
            build_context: BuildContext = BuildContext(user_id=runtime.context.user_id, has_reference=False)

        response = await build_workflow_agent.ainvoke(
            build_input,  # type: ignore
            context=build_context,  # type: ignore
        )

        return {'messages': response['messages'], 'workflow_uids': response['workflow_uids']}

    def build(self, *, visualize: bool = False):
        graph = StateGraph(DesignState, UserContext)  # type: ignore

        graph.add_node('intent_recognition', self.intent_recognition)  # type: ignore
        graph.set_entry_point('intent_recognition')

        # 自主设计路径
        graph.add_node('autonomous_design', self.autonomous_design_node)  # type: ignore

        graph.add_edge('autonomous_design', 'generate_workflow')
        graph.add_edge('generate_workflow', END)

        # 文献参考路径
        graph.add_node('summary_paper', self.summary_paper_node)  # type: ignore
        graph.add_node('split_workflow', self.split_workflow_node)  # type: ignore
        graph.add_node(
            'plan_confirmation',
            self.plan_confirmation_node,  # type: ignore
            destinations=('generate_workflow', 'split_workflow'),
        )
        graph.add_node('generate_workflow', self.generate_workflow_node)  # type: ignore

        graph.add_edge('summary_paper', 'split_workflow')
        graph.add_edge('split_workflow', 'plan_confirmation')
        graph.add_edge('generate_workflow', END)

        compiled_graph = graph.compile(checkpointer=self.memory, cache=llm_cache)

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/reproduction_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph


@tool
async def write_plan(plan: str, runtime: ToolRuntime) -> Command:
    """Write or update your plan.

    Args:
        plan: Specific content of the plan
    """
    if runtime.state.get('workflow_description'):
        return Command(
            update={
                'workflow_description': plan,
                'messages': [ToolMessage(content='Updated the plan content.', tool_call_id=runtime.tool_call_id)],
            }
        )
    return Command(
        update={
            'workflow_description': plan,
            'messages': [ToolMessage(content='The plan content has been saved.', tool_call_id=runtime.tool_call_id)],
        }
    )


@tool('ask_human', parse_docstring=True)
async def wait_for_confirm(question: str, runtime: ToolRuntime) -> Command:
    """Consult humans for advice.

    Args:
        question: Issues to consult with humans
    """
    answer = interrupt({'question': question})
    if isinstance(answer, bool) and answer:
        return Command(
            goto='__end__',
            update={
                'messages': [
                    ToolMessage(content='The user agreed with your point.', tool_call_id=runtime.tool_call_id)
                ],
            },
        )
    return Command(
        goto='model',
        update={'messages': [ToolMessage(content=f'Human feedback is: "{answer}"', tool_call_id=runtime.tool_call_id)]},
    )
