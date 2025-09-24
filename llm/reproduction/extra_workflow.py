from typing import TypedDict, Optional, cast, Literal

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

from llm.states import ExtraWorkflowState, ExtractOutput1, ExtractOutput2

EXTRA_SYSTEM_1 = """你是一名生物信息学工程师，你的任务是根据用户提供的资料，将其中描述的生物信息学分析流程总结出来，并以固定的格式输出。

<ThinkingProcess>
为了达成这一目的，你可以将整个任务分解为以下步骤：

1. 首先简要的阅读用户提供的资料，先对其进行总结。同时提取出材料中所使用到的生物信息分析工具
2. 详细阅读材料，提取出其中的生物信息分析工作流并输出

每一步都需要遵循固定的格式进行输出，详见下文<OutputFormat>部分
</ThinkingProcess>

<OutputFormat>
当前正在进行的是步骤一，你的输出需要包含以下内容：
- `overall_summary`：对用户所提供资料的简单描述
- `major_finding`：如果用户提供的是一篇完整的学术论文，总结这篇论文的主要结论/发现。否则留空
- 提取的工具。具体而言，包括：
    - `tools`：生物信息学分析工具，一般指的是通过conda或者二进制包的形式安装的分析软件
    - `r_packages`和`python_packages`：在下游分析中使用的特定编程语言中的库函数
    - 注意，区分tools和packages主要是看此软件（算法）是直接可以通过命令行调用，还是需要编写代码调用相关模块。不能因为有的软件是用python编写的（如spades）就觉得他是packages
    - 我们只需要直接进行分析的软件和包，因为我们将编写自己的工作流编排软件。类似nextflow这样的编排工具不需要提取

特别的，如果用户提供的资料中没用可用的生物信息学分析工作流，则直接在`overall_summary`中指出，其它词条全部为空。
</OutputFormat>
"""

EXTRA_SYSTEM_2 = """你是一名生物信息学工程师，你的任务是根据用户提供的资料，将其中描述的生物信息学分析流程总结出来，并以固定的格式输出。

<ThinkingProcess>
为了达成这一目的，你可以将整个任务分解为以下步骤：

1. 首先简要的阅读用户提供的资料，先对其进行总结。同时提取出材料中所使用到的生物信息分析工具
2. 详细阅读材料，提取出其中的生物信息分析工作流并输出
    - 基于上一步提取到的工具，如果根据文中的描述可以将它们归入同一个有向无环图之中，则我们认为这些工具可以组成一个工作流
    - 有多少个有向无环图，就有多少个工作流

每一步都需要遵循固定的格式进行输出，详见下文<OutputFormat>部分
</ThinkingProcess>

<OutputFormat>
当前正在进行的是步骤二，你的输出需要包含一个列表`workflows`，每一个元素对应一个工作流。每个工作流对象需要包含如下元素：
- `name`：基于功能、分析数据类型等为这个工作流起一个名字
- `workflow_summary`：简要描述这个工作流的生物信息学作用
- `conclusion_support`：如果用户提供的是一篇完整的学术论文，基于上一步得到的`major_finding`的内容，描述这个工作流运行得到的结果在得到分析结果的过程中起了什么作用。否则可以留空
- `workflow_detail`：工作流的详细描述
    - 使用有序列表描述工作流的每一步，格式为：`1. (工具类型) 使用**(工具名称)**...`
        - 工具类型只允许在tool、r_package和python_package中三选一
        - 如果文中有与工具参数选择相关的描述，使用无序列表将相关的原文作为子列表附加
    - 这一步的样式可以参考下文的<WorkflowDetailExample>部分
- `mermaid_code`：使用mermaid语法绘制的工作流结构示意图，布局为LR
    - 以 "```mermaid" 开头，以 "```" 结尾（即标准markdown代码块格式），任务中不包含任何其他文本。
    - 使用默认节点表示分析工具 (`id`)
    - 使用不对称形状节点表示中间文件 (`id>文件描述]`)
    - 使用子例程形状节点表示 R 或 Python 包 (`id[[工具/包名称]]`)
    - 请注意，节点显示**只需要**最基本的工具/包名称和文件描述。版本号、参数和其他详细信息已在`workflow_detail`中描述，**不应**在节点中显示。
    - 为了Mermaid正确渲染，节点显示内容中**不要使用任何特殊符号**（尤其是括号、引号等），除了语法要求之外。

特别的，如果用户提供的资料中没用可用的生物信息学分析工作流，则直接返回一个空的workflows列表即可
</OutputFormat>

<WorkflowDetailExample>
1.  (tool): Perform quality control on raw data using `fastp`
    * Filter out reads shorter than 35bp
2.  (tool): Align high-quality reads to the reference genome using `STAR`
3.  (tool): Quantify gene expression using `featureCounts`
    * Count at the gene level, summarizing reads aligned to exons
4.  (r package): Perform differential expression analysis using the `DESeq2` R package
5.  (r package): Perform functional enrichment analysis using the `clusterProfiler` R package
</WorkflowOverviewExample>
"""


class ExtraWorkflowAgent:
    def __init__(self, llm: ChatOpenAI, need_interrupt: bool = False):
        self.llm = llm

        self.need_interrupt = need_interrupt

    async def summary_paper_node(self, state: ExtraWorkflowState):
        writer = get_stream_writer()
        writer({'info': 'Summarizing literature...'})

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ExtractOutput1]
            parsing_error: Optional[BaseException]

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=EXTRA_SYSTEM_1), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.with_structured_output(ExtractOutput1, include_raw=True)
        output = cast(StructureOutput, await chain.ainvoke({'history': state.messages}))

        if output['parsed'].major_finding:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Major Finding\n'
                f'{output["parsed"].major_finding}\n\n'
                f'## Tools used in the article\n'
                f'{_tool_formatter(output["parsed"])}'
            )
        else:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Tools used in the article\n'
                f'{_tool_formatter(output["parsed"])}'
            )

        return {
            'tools': output['parsed'].tools,
            'r_packages': output['parsed'].r_packages,
            'python_packages': output['parsed'].python_packages,
            'formatted_msg': formatted_msg,
            'messages': [output['raw']],
        }

    async def extract_workflow_node(self, state: ExtraWorkflowState):
        writer = get_stream_writer()
        writer({'info': 'Extracting workflow...'})

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ExtractOutput2]
            parsing_error: Optional[BaseException]

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=EXTRA_SYSTEM_2), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.with_structured_output(ExtractOutput2, include_raw=True)
        output = cast(StructureOutput, await chain.ainvoke({'history': state.messages}))

        if output['parsed'].workflows:
            step1_msg = state.formatted_msg
            step2_msg = _workflow_formatter(output['parsed'])

            return {'messages': [AIMessage(content=f'{step1_msg}\n\n## Workflows\n{step2_msg}')]}
        else:
            return {
                'messages': [
                    AIMessage(
                        content='It seems that the file you provided does not contain a bioinformatics analysis workflow.'
                    )
                ]
            }

    def step_router(self, state: ExtraWorkflowState) -> Literal['human_interrupt', 'auto_decision']:
        if self.need_interrupt:
            return 'human_interrupt'
        else:
            return 'auto_decision'

    @staticmethod
    async def interrupt_node(
        state: ExtraWorkflowState,
    ) -> Command[Literal['__end__', 'summary_paper', 'extract_workflow']]:
        """人工干预"""
        answer = interrupt({'confirm': f"{state.formatted_msg}\n\nDo you think it's okay? Can we continue?"})
        if isinstance(answer, bool) and answer:
            if state.analyse_step == 0:
                filtered_history = [
                    m if isinstance(m, HumanMessage) else AIMessage(content='Blocked history messages')
                    for m in state.messages[1:-1]
                ]
                return Command(
                    goto='extract_workflow',
                    update={
                        'analyse_step': 1,
                        'messages': [
                            *[RemoveMessage(id=m.id) for m in state.messages[1:]],
                            *filtered_history,
                            ('ai', state.formatted_msg),
                            ('human', 'Very good, now continue to complete the remaining steps.'),
                        ],
                    },
                )
            else:
                return Command(goto=END)

        next_node = 'summary_paper' if state.analyse_step == 0 else 'extract_workflow'
        return Command(goto=next_node, update={'messages': [HumanMessage(content=answer)]})

    @staticmethod
    async def auto_decision_node(state: ExtraWorkflowState) -> Command[Literal['__end__', 'extract_workflow']]:
        if state.analyse_step == 0:
            if len(state.tools) == 0:
                return Command(
                    goto=END,
                    update={
                        'messages': [
                            AIMessage(
                                content='It seems that the file you provided does not contain a bioinformatics analysis workflow.'
                            )
                        ]
                    },
                )
            else:
                return Command(
                    goto='extract_workflow',
                    update={
                        'analyse_step': 1,
                        'messages': [
                            RemoveMessage(id=state.messages[-1].id),
                            ('ai', state.formatted_msg),
                            ('human', 'Very good, now continue to complete the remaining steps.'),
                        ],
                    },
                )
        return Command(goto=END)

    def build(self, visualize: bool = False, memory: Optional[BaseCheckpointSaver] = None):
        graph = StateGraph(ExtraWorkflowState)
        graph.add_node('summary_paper', self.summary_paper_node)
        graph.add_node('human_interrupt', self.interrupt_node)
        graph.add_node('auto_decision', self.auto_decision_node)
        graph.add_node('extract_workflow', self.extract_workflow_node)

        graph.set_entry_point('summary_paper')
        graph.add_conditional_edges('summary_paper', self.step_router)

        compiled_graph = graph.compile(checkpointer=memory) if memory else graph.compile()
        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/extra_workflow.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')
        return compiled_graph


def _tool_formatter(llm_result: ExtractOutput1) -> str:
    tool_list = []
    if llm_result.tools:
        tool_list.append(f'- **tools**\n{"\n".join([f"    - `{t}`" for t in llm_result.tools])}')

    if llm_result.r_packages:
        tool_list.append(f'- **r packages**\n{"\n".join([f"    - `{t}`" for t in llm_result.r_packages])}')

    if llm_result.python_packages:
        tool_list.append(f'- **python packages**\n{"\n".join([f"    - `{t}`" for t in llm_result.python_packages])}')

    return '\n'.join(tool_list)


def _workflow_formatter(llm_result: ExtractOutput2) -> str:
    wf_list = []

    for wf in llm_result.workflows:
        if wf.conclusion_support:
            wf_list.append(
                f'### {wf.name}\n'
                f'#### Summary\n'
                f'{wf.workflow_summary}\n\n'
                f'#### Conclusion Support\n'
                f'{wf.conclusion_support}\n\n'
                f'#### Workflow Detail\n'
                f'{wf.workflow_detail}\n\n'
                f'{wf.mermaid_code}'
            )
        else:
            wf_list.append(
                f'### {wf.name}\n'
                f'#### Summary\n'
                f'{wf.workflow_summary}\n\n'
                f'#### Workflow Detail\n'
                f'{wf.workflow_detail}\n\n'
                f'{wf.mermaid_code}'
            )

    return '\n'.join(wf_list)
