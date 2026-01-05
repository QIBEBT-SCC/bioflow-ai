from typing import Literal, Optional

import rich
from langchain.agents import create_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from llm.agents.reproduction.reproduction import build_workflow_agent
from llm.models import get_agent_model, get_long_context_model, get_simple_model
from llm.states import BuildSchema, DesignState, StructuredResponse, UserContext
from llm.tools.storage import view_pdf
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.token import store_token_usage

INTENT_RECOGNITION_SYSTEM = """You are a user requirements analyst. You need to classify the current user's requirement into one of the following three categories based on a conversation context:
1. **reproduce_paper**: The user's requirement is to reproduce a method or experiment from a certain academic paper (from a PDF, txt, markdown, etc.)
2. redesign: The user's requirement is to design a new workflow or method
3. from_script: The user's requirement is to reproduce a workflow based on an existing script or code (a sh or nextflow script, etc.)

Please select the category that best matches the user's requirement based on the conversation content, and briefly explain the reason for this classification.
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

# Core Definition: How to divide workflows?
Extract workflows based on the following logic tree:
1. First, analyze the different types of raw wet-lab data in the article (e.g., scRNA-seq, meta, ATAC). Each different type of raw data corresponds to an independent workflow.
2. A workflow extends until a single analysis for the current raw data is completed. For example, for simple scRNA-seq data, the workflow should include quality control, alignment, and expression matrix generation from Raw Data.
3. For joint analysis of multiple data or different types of data, create new independent workflows, also extending until the current data analysis is completed.
4. Downstream statistical analysis and plotting based on R or Python should be treated as a separate downstream analysis workflow, not to be confused with upstream data processing workflows.

In simple terms, imagine you are writing bash scripts or nextflow scripts, where each workflow corresponds to an independent script file, and each script file completes a clear data processing task.
Avoid vague workflow divisions like "integrated analysis" or "multi-omics joint analysis."
However, do not over-split workflows, for example, separating quality control and alignment into two workflows, as this would lead to workflow fragmentation and hinder engineering implementation.


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

3. **Workflow #3 Downstream Analysis** (if any)
   - **Source Data**: [e.g., Gene Abundance Matrix (from Workflow #2)]
   - **Main Tools and Steps**: [e.g., R (DESeq2, clusterProfiler, ggplot2)]
   - **Function Description**: [Summarize the statistical analysis and plot generation content included]
```

The language of your generated summary must match the user's question language.
"""


class IntentResult(BaseModel):
    task: Literal['reproduce_paper', 'redesign', 'from_script'] = Field(description='')
    reason: str = Field(description='The reason you made this classification')


class WorkflowDesigner:
    def __init__(self, history: BaseChatMessageHistory):
        self.name = 'workflow_designer'
        self.summary_llm = get_long_context_model()
        self.agent_llm = get_agent_model()

        self.history = history

        self.summary_paper_agent = create_agent(
            self.summary_llm,
            [view_pdf],
            system_prompt=SUMMARY_PAPER_SYSTEM,
            middleware=[TokenStatisticMiddleware(agent_name=self.name, **self.summary_llm.metadata)],  # type: ignore
            state_schema=DesignState,  # type: ignore
        )
        self.split_workflow_agent = create_agent(
            self.summary_llm,
            [view_pdf],
            system_prompt=SPLIT_WORKFLOW_SYSTEM,
            middleware=[TokenStatisticMiddleware(agent_name=self.name, **self.summary_llm.metadata)],  # type: ignore
            state_schema=DesignState,  # type: ignore
        )

    async def intent_recognition(self, state: DesignState, runtime: Runtime[UserContext]):  # type: ignore
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

        return {'task': response['parsed'].task}

    @staticmethod
    def route_path(state: DesignState) -> Literal['summary_paper', '__end__']:
        match state['task']:
            case 'reproduce_paper':
                return 'summary_paper'
            case _:
                return '__end__'

    async def summary_paper_node(self, state: DesignState, runtime: Runtime[UserContext]):  # type: ignore
        """从文献复现Step1 - 文献总结"""

        response = await self.summary_paper_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )
        await self.history.aadd_messages([response['messages'][-1]])

        return {'paper_id': response['paper_id']}

    async def split_workflow_node(self, state: DesignState, runtime: Runtime[UserContext]):  # type: ignore
        """文献复现Step2 - 工作流拆解"""

        response = await self.split_workflow_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )
        await self.history.aadd_messages([response['messages'][-1]])

        return {'messages': [response['messages'][-1]]}

    async def plan_confirmation_node(self, state: DesignState):
        """文献复现Step3 - 方案确认"""
        confirm = interrupt({'info': 'plz confirm the plan above, whether to proceed?'})

        if isinstance(confirm, bool) and confirm:
            return Command(goto='generate_workflow', update={'workflow_description': state['messages'][-1].content})
        else:
            return Command(goto='split_workflow', update={'messages': [HumanMessage(content=confirm)]})

    async def generate_workflow_node(self, state: DesignState, runtime: Runtime[UserContext]):
        """文献复现Step4 - 工作流生成"""

        build_input: BuildSchema = {
            'paper_id': state['paper_id'],
            'messages': [
                HumanMessage(
                    content=f'The following workflows are available in the literature, please build them:\n{state["workflow_description"]}'
                )
            ],
        }
        response = await build_workflow_agent.ainvoke(
            build_input,  # type: ignore
            context=UserContext(user_id=runtime.context.user_id),  # type: ignore
        )

        return {'workflow_uids': response['workflow_uids']}

    def build(self, memory: Optional[BaseCheckpointSaver] = None, visualize: bool = False):
        graph = StateGraph(DesignState, UserContext)  # type: ignore

        graph.add_node('intent_recognition', self.intent_recognition)  # type: ignore
        graph.add_node('summary_paper', self.summary_paper_node)  # type: ignore
        graph.add_node('split_workflow', self.split_workflow_node)  # type: ignore
        graph.add_node(
            'plan_confirmation',
            self.plan_confirmation_node,  # type: ignore
            destinations=('generate_workflow', 'split_workflow'),
        )
        graph.add_node('generate_workflow', self.generate_workflow_node)  # type: ignore

        graph.set_entry_point('intent_recognition')
        graph.add_conditional_edges('intent_recognition', self.route_path)
        graph.add_edge('summary_paper', 'split_workflow')
        graph.add_edge('split_workflow', 'plan_confirmation')
        graph.add_edge('generate_workflow', END)

        if memory:
            compiled_graph = graph.compile(checkpointer=memory)
        else:
            compiled_graph = graph.compile()

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/reproduction_agent.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph
