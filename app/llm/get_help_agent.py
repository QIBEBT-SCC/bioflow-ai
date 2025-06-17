import operator
from typing import Literal, Annotated

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import ToolException
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.core.llm_chat import gpt_41_mini
from app.llm.tools.container import RunDocker


GLOBAL_SYSTEM_PROMPT = """You are an expert specializing in the analysis of bioinformatics tools. The user wants to obtain the help documentation for a specific tool.

<goal>  
Your task is to:  
1. Analyze the user's input to determine the correct command to run.  
2. Use the `run_command` tool to execute the appropriate command to retrieve help information.  
3. Analyze the command output to determine whether it includes subcommands.  
4. If subcommands exist, retrieve detailed help information for each of them as follows:  
   * For the **first** subcommand, invoke the tool separately and continue trying until you get a valid result.  
   * Once the first subcommand succeeds, proceed to invoke the tool **multiple times in parallel** to retrieve help information for the **remaining subcommands**.  
5. When you believe enough help information has been collected, say **"I have finished collecting all the help information"** to indicate the end of the collection phase.  
</goal>

<tool_calling>
* The tool is already running in a Docker container. You only need to provide the bare command-line instructions — do **not** include prefixes like `docker run`.
* If the tool has subcommands, follow the above procedure to retrieve help documentation.
</tool_calling>

<error_handling>
If the tool returns information that is not valid software help documentation, it means your command was incorrect and should be revised.

* If you receive an error like `command not found`, it indicates the tool name is incorrect. Try a new, correct tool name based on your knowledge.
* If you receive an error like `unrecognized arguments`, it means the parameters are incorrect. In this case:
  * First, try to determine the correct command for displaying the help documentation based on the returned message.
  * If that fails, try the following options in order:
    * `tool_name --help` or `tool_name -h`
    * `tool_name help`
    * `tool_name` (some tools show help when run with no arguments)
</error_handling>

Please choose the most appropriate command based on the tool name and progressively retrieve the complete help documentation.
"""

ERROR_SYS_PROMPT = """You are an expert specializing in the analysis of bioinformatics tools. The user wants to obtain the help documentation for a specific tool.

现在，我们要尝试获取一个工具的子命令的帮助文档，你的任务是根据系统返回的结果判断返回的内容是否是我们想要的子命令的帮助信息。
你会得到工具的主帮助信息，此次执行的指令，以及该指令的返回值。
"""

ERROR_USER_PROMPT = """<main_doc>
{main_doc}
</main_doc>

Now we try the command:
```
{command}
```

and the return is:
<sub_command_return>
{sub_command_return}
</sub_command_return>
"""


class HelpDocument(BaseModel):
    title: str
    content: str


class ErrorStructure(BaseModel):
    """这里得到的输出是否是期望的子命令的帮助文档？"""

    is_doc: bool = Field(description='返回的文本是否是子命令的帮助文档')


class HelpDocumentState(MessagesState):
    """智能体状态管理"""

    docker_repo: str
    docker_tag: str
    registry_host: str

    documents: Annotated[list[HelpDocument], operator.add]
    final_document: str
    tool_name: str


class HelpDocumentOutput(MessagesState):
    final_document: str
    tool_name: str


class SmartHelpDocumentAgent:
    """智能帮助文档获取智能体"""

    def __init__(self):
        self.llm = gpt_41_mini()
        self.structure_llm = gpt_41_mini()
        self.docker_tool = RunDocker()

        # 绑定工具到LLM
        self.llm_with_tools = self.llm.bind_tools([self.docker_tool], tool_choice='auto', parallel_tool_calls=True)

    def call_llm(self, state: HelpDocumentState) -> dict:
        """LLM节点：决定是否调用工具"""
        messages = state.get('messages', [])

        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=GLOBAL_SYSTEM_PROMPT), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm_with_tools
        response = chain.invoke({'history': messages})

        return {'messages': [response]}

    @staticmethod
    def should_continue(state: HelpDocumentState) -> Literal['run_command', 'run_command_parallel', 'format_document']:
        """判断是否继续调用工具、收集文档或格式化文档"""
        messages = state.get('messages', [])
        last_message = messages[-1]

        # 如果最后一条消息有工具调用，继续执行工具
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if len(last_message.tool_calls) > 1:
                return 'run_command_parallel'
            return 'run_command'

        # 否则格式化文档
        return 'format_document'

    def run_command(self, state: HelpDocumentState):
        messages = state.get('messages', [])
        tool_call: ToolCall = messages[-1].tool_calls[0]
        tool_call_id = tool_call['id']

        try:
            command_output = self.docker_tool.invoke(
                {
                    **tool_call['args'],
                    'state': {
                        'docker_repo': state.get('docker_repo'),
                        'docker_tag': state.get('docker_tag'),
                        'registry_host': state.get('registry_host', 'docker.io'),
                    },
                }
            )
            if len(state.get('documents', [])) == 0:
                return {
                    'messages': [ToolMessage(command_output, tool_call_id=tool_call_id)],
                    'documents': [HelpDocument(title=tool_call['args']['command'], content=command_output)],
                }
            return {
                'messages': [
                    ToolMessage('Successful get the subcommand help doc, add to documents', tool_call_id=tool_call_id)
                ],
                'documents': [HelpDocument(title=tool_call['args']['command'], content=command_output)],
            }
        except ToolException as e:
            # 第一个子命令
            if state.get('documents'):
                prompt = ChatPromptTemplate.from_messages(
                    [SystemMessage(content=ERROR_SYS_PROMPT), ('human', ERROR_USER_PROMPT)]
                )
                chain = prompt | self.structure_llm.with_structured_output(ErrorStructure)
                response: ErrorStructure = chain.invoke(
                    {
                        'main_doc': state.get('documents')[0].content,
                        'command': tool_call['args'],
                        'sub_command_return': str(e),
                    }
                )

                if response.is_doc:
                    return {
                        'messages': [
                            ToolMessage(
                                'Successful get the subcommand help doc, add to documents. Now you can call the tool for the remaining subcommands in parallel.',
                                tool_call_id=tool_call_id,
                            )
                        ],
                        'documents': [HelpDocument(title=tool_call['args']['command'], content=str(e))],
                    }

            return {'messages': [ToolMessage(str(e), tool_call_id=tool_call_id)]}

    def run_command_parallel(self, state: HelpDocumentState):
        last_messages = state.get('messages', [])[-1]
        tool_messages = []
        new_documents = []

        for tool_call in last_messages.tool_calls:
            try:
                command_output = self.docker_tool.invoke(
                    {
                        **tool_call['args'],
                        'state': {
                            'docker_repo': state.get('docker_repo'),
                            'docker_tag': state.get('docker_tag'),
                            'registry_host': state.get('registry_host', 'docker.io'),
                        },
                    }
                )

                new_documents.append(HelpDocument(title=tool_call['args']['command'], content=command_output))
            except ToolException as e:
                new_documents.append(HelpDocument(title=tool_call['args']['command'], content=str(e)))
            finally:
                tool_messages.append(
                    ToolMessage(
                        'Successful get the subcommand help doc, add to documents', tool_call_id=tool_call['id']
                    )
                )

        return {'messages': tool_messages, 'documents': new_documents}

    @staticmethod
    def format_document(state: HelpDocumentState) -> dict:
        """格式化最终文档"""
        documents = state.get('documents', [])

        if not documents:
            return {
                'messages': [AIMessage(content='我无法获取到有效的帮助文档，请检查相关的信息是否正确。')],
                'final_document': '',
            }

        # 使用收集到的文档自动拼接
        tool_name = documents[0].title.split(' ')[0]
        document_parts = [f'# {tool_name} 帮助文档']

        # 添加每个文档
        for doc in documents:
            document_parts.append(f'\n## {doc.title}\n')
            document_parts.append('```')
            document_parts.append(doc.content)
            document_parts.append('```')

        final_document = '\n'.join(document_parts)

        return {
            'messages': [AIMessage(content=f'已生成完整的帮助文档，包含 {len(documents)} 个部分。')],
            'final_document': final_document,
            'tool_name': tool_name,
        }

    def build_graph(self, visualize: bool = False) -> CompiledStateGraph:
        """构建LangGraph工作流"""
        graph = StateGraph(HelpDocumentState, output=HelpDocumentOutput)

        graph.add_node('call_llm', self.call_llm)
        graph.add_node('run_command', self.run_command)
        graph.add_node('run_command_parallel', self.run_command_parallel)
        graph.add_node('format_document', self.format_document)

        graph.add_edge(START, 'call_llm')
        graph.add_conditional_edges('call_llm', self.should_continue)
        graph.add_edge('run_command', 'call_llm')
        graph.add_edge('run_command_parallel', 'format_document')
        graph.add_edge('format_document', END)

        compiled_graph = graph.compile()
        if visualize:
            compiled_graph.get_graph(xray=True).draw_mermaid_png(
                output_file_path='help_doc_agent.png', draw_method=MermaidDrawMethod.API, curve_style=CurveStyle.BASIS
            )
        return compiled_graph
