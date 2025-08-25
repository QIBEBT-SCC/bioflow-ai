from typing import Annotated, Literal, Optional

from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, RemoveMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

from sqlmodel import Session

import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.crud.tools import image as image_crud
from llm.utils.formatters import handle_formatter
from llm.generate_tool_config import tool_generator, ToolConfigSchema
from llm.models import qwen3_embedding, gemini_25_flash, gpt_41_mini
from llm.states import (
    CreateToolState,
    CreateToolOutput,
    SearchImageState,
    GetDocState,
    TestRunState,
    SelectGroupState,
)
from llm.tools.container import run_command, build_container, run_command_in_container, stop_and_rm_container
from llm.tools.database import (
    search_local_image,
    pull_and_save_image,
    list_docs,
    view_doc,
    save_doc,
    view_groups,
    add_group,
    search_biodb,
)
from llm.tools.human import ask_human
from llm.tools.images import search_image
from llm.tools.testdata import view_testdata
from llm.tools.web_search import google_search
from app.models.tools import ToolType, ToolDefine
from app.schemas.tools import ToolNodePublic

SELECT_IMAGE_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, analysis data, etc., into individual nodes. Now, what we need to do is encapsulate a new node.
To achieve this, your first task is to **find available Docker images** based on the user's description, which will serve as the foundation for encapsulation.

<ThinkingProcess>
To find them, you just need to follow these steps:
First, use the `search_local_image` tool to search the local database for existing tool images.
    - If an image for the tool exists, use `select_image` to select the corresponding image.
    - Otherwise, call `search_image` to search for images on Docker Hub, and based on the results, use the `pull_and_save_image` tool to pull and save the image.

If you encounter any unresolvable issues at any point, you can call the `ask_human` tool to ask for human assistance.
</ThinkingProcess>
"""

GET_DOC_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is encapsulate a new node.
To achieve this, we have already found the corresponding image for the tool in previous work. Your task now is to **obtain the help documentation for the corresponding tool** based on the user's description, as a basis for encapsulation.

<ThinkingProcess>
To achieve your goal, you can follow these steps:
1. First, call the `list_docs` tool to view the tool documentation already recorded under the corresponding image.
2. If there are records in the returned results that you think might be useful, you can call the `view_doc` tool to view the specific content of that record.
    - If this record is suitable, call the `select_doc` tool to select this document and complete your task.
    - Otherwise, view other records as appropriate or proceed to the next step.
3. If the result of step 1 is empty or there are no suitable records, then you need to call the `run_command` tool to try to run commands within the corresponding container to get help documentation (see the detailed explanation in the <RunCommand> section).
4. `save_doc` will save the output from the last `run_command` execution to the database. After you get the desired help documentation, remember to call this tool to save the document and complete your task.

If you encounter any unresolvable issues at any point, you can call the `ask_human` tool to ask for human assistance.
</ThinkingProcess>

<RunCommand>
When there is no corresponding help document in the database, you need to try running commands to get the tool's help document. In this step, you need to pay attention to:
- The tool is already running in a Docker container. You only need to provide the bare command-line instructions — do **not** include prefixes like `docker run`.
- If you receive an error like `command not found`, it indicates the tool name is incorrect. Try a new, correct tool name based on your knowledge.
    - If you can't find the name of the tool in your knowledge, try to remember if you have to add a suffix such as `.py`, `.sh`, etc.
</RunCommand>
"""

GET_REPORT_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
To achieve this, your task is to **obtain the output structure of the corresponding tool** as the basis for encapsulation.

<ToolCalling>
Actually, the input files required by the tool can be clearly seen from its help documentation. However, the node also needs to know the specific output files of the tool to construct output handles. To achieve this, you can use the following tools:
- We have some predefined test data. If the required input file (type) for the tool exists in the test data, you can directly call the `test_run` tool to run the tool and see what the output looks like.
- If the required data is not available in the test data, call the `google_search` tool to search online for information about the tool's output.
</ToolCalling>
"""

TEST_RUN_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
To achieve this, your current task is to attempt to run actual commands within a container based on existing information, and **provide a tool execution report** based on the results, offering the necessary information for node encapsulation.


<ThinkingProcess>
To achieve your goal, you can follow these steps:
1. First, according to the tool's documentation, determine if an additional database is required. If so, call the `search_biodb` tool to find the addresses of available databases.
2. Call the `build_container` tool to create a container for testing. By default, the test data has been mounted into the container; you can add other mounts as needed.
    - Since this is just a test run, the tool's output does not need to be mapped out; just mount the necessary external files.
3. Call `run_command_in_container` to execute commands within the container and obtain the tool's output.
    - It is recommended to use `ls -la` to view the output files generated by the tool.
    - If the output is too long, the content will not be displayed directly. You can call the `view_output` tool to view part of the output.
4. When the requirements have been run, call the `stop_and_rm_container` tool to delete the temporary container.
5. Based on the feedback during the tool's operation, provide a report.

If you encounter any unresolvable issues at any point, you can call the `ask_human` tool to ask for human assistance.
</ThinkingProcess>

<OutputFormat>
Once you have completed all analyses and gathered all the required information, you must stop calling tools and produce a **Final Report**. This report must strictly adhere to the following text format, without any JSON, code blocks, or other extraneous explanatory text.

```text
**Tool Analysis Report**
1.  **Does it require a database?**
      * [Yes/No].

2.  **Output File Composition Analysis:**
      * /path/to/file_A: [Input for subsequent analysis/Report file/Temporary file/Unknown] - [Briefly describe the file's purpose.]
      * /path/to/file_B: [Input for subsequent analysis/Report file/Temporary file/Unknown] - [Briefly describe the file's purpose.]
      * ...

3.  **Does it require manual creation of the output folder?**
      * [Yes/No].

4.  **Is a log file generated automatically?**
      * [Yes/No].
```
</OutputFormat>
"""

REPORT_STRUCTURE = """<OutputFormat>
Once you have completed all analyses and gathered all the required information, you must stop calling tools and produce a **Final Report**. This report must strictly adhere to the following text format, without any JSON, code blocks, or other extraneous explanatory text.

```text
**Tool Analysis Report**
1.  **Does it require a database?**
      * [Yes/No].

2.  **Output File Composition Analysis:**
      * /path/to/file_A: [Input for subsequent analysis/Report file/Temporary file/Unknown] - [Briefly describe the file's purpose.]
      * /path/to/file_B: [Input for subsequent analysis/Report file/Temporary file/Unknown] - [Briefly describe the file's purpose.]
      * ...

3.  **Does it require manual creation of the output folder?**
      * [Yes/No].

4.  **Is a log file generated automatically?**
      * [Yes/No].
```
</OutputFormat>
"""

SELECT_GROUP_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
Now your task is to **assign an appropriate group** to this tool based on the information available about it.

<ThinkingProcess>
To complete this task, you can follow these steps:
1. Call the `view_groups` tool to see the existing groups.
2. If there is a suitable group, directly call the `select_group` tool to select the corresponding group.
3. If there is no suitable group, call the `add_group` tool to create a new group. The creation of new groups should follow these principles:
    - The creation of top-level groups should be extremely cautious, only creating them when it is certain that the tool belongs to a completely new major category.
    - Sub-groups are generally used for tools that have multiple sub-commands, and the group name is usually the main tool name.
    - According to the above rules, groups should not be too deep, with a maximum of two to three levels.
4. After creating a new group, you also need to call the `select_group` tool to select the corresponding group.
</ThinkingProcess>
"""


class CreateToolAgent:
    def __init__(self):
        self.struct_llm = gpt_41_mini()
        self.think_llm = gemini_25_flash()

        # tool
        self.web_search_tool = google_search

        # sub agents
        self.search_image_agent = create_react_agent(
            model=self.think_llm,
            tools=[search_local_image, search_image, select_image, pull_and_save_image, ask_human],
            prompt=SELECT_IMAGE_SYSTEM,
            state_schema=SearchImageState,
        )
        self.get_doc_agent = create_react_agent(
            model=self.think_llm,
            tools=[list_docs, view_doc, select_doc, run_command, save_doc, ask_human],
            prompt=GET_DOC_SYSTEM,
            state_schema=GetDocState,
        )
        self.test_run_agent = create_react_agent(
            model=self.think_llm,
            tools=[
                search_biodb,
                build_container,
                run_command_in_container,
                view_output,
                stop_and_rm_container,
                ask_human,
            ],
            prompt=TEST_RUN_SYSTEM,
            state_schema=TestRunState,
        )
        self.select_group_agent = create_react_agent(
            model=self.think_llm,
            tools=[view_groups, add_group, select_group],
            prompt=SELECT_GROUP_SYSTEM,
            state_schema=SelectGroupState,
        )

    def select_image_node(self, state: CreateToolState):
        """首先搜索得到此工具的对应镜像"""
        search_input = SearchImageState(messages=state.messages)
        response = self.search_image_agent.invoke(search_input)

        return {'image_id': response['image_id'], 'image': response['image']}

    def get_doc_node(self, state: CreateToolState):
        """然后尝试获取指令对应的帮助文档"""
        doc_input = GetDocState(messages=state.messages, image_id=state.image_id, image=state.image)
        response = self.get_doc_agent.invoke(doc_input)

        with Session(engine) as session:
            doc = doc_crud.get(session, response['doc_id'])

        return {
            'messages': [
                ('ai', 'First I need the help documentation for this tool'),
                ('human', f'The following is the help documentation: \n\n{doc.content}'),
            ],
            'doc_id': response['doc_id'],
            'doc': doc.content,
        }

    def get_output_report_node(self, state: CreateToolState):
        """接下来尝试获取工具的输出结果（因为文档通常不会说输出是什么）"""
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=GET_REPORT_SYSTEM),
                MessagesPlaceholder(variable_name='history'),
                ('ai', 'Now let me see what test data is available'),
                ('human', '{test_date_desc}'),
            ]
        )
        llm = self.think_llm.bind_tools([test_run, google_search], tool_choice='any')  # 这里只允许二选一调用
        chain = prompt | llm

        response = chain.invoke({'history': state.messages, 'test_date_desc': view_testdata()})
        return {'messages': [response]}

    @staticmethod
    def how_to_get_output(state: CreateToolState) -> Literal['test_run_node', 'web_search_node']:
        last_message = state.messages[-1]
        assert isinstance(last_message, AIMessage)
        tool_call = last_message.tool_calls[0]

        match tool_call['name']:
            case 'test_run':
                return 'test_run_node'
            case 'google_search':
                return 'web_search_node'
            case _:
                # 由于上面的工具调用定义，实际上不会到这里
                raise

    def test_run_node(self, state: CreateToolState):
        """如果测试数据可用，则直接测试运行"""
        test_run_input = TestRunState(
            messages=[
                *state.messages[:-1],
                AIMessage(content='Now let me see what test data is available'),
                HumanMessage(
                    content=f'{view_testdata()}\n\nThey are mounted under the `/data/testdata` directory in the container.'
                ),
            ],
            image=state.image,
        )
        report = self.test_run_agent.invoke(test_run_input)

        return {
            'messages': [
                RemoveMessage(id=state.messages[-1].id),
                ('ai', 'Now I need to know the output structure of this tool'),
                ('human', report['messages'][-1].content),
            ]
        }

    def web_search_node(self, state: CreateToolState):
        """没有现成的测试数据，联网搜索"""
        last_message = state.messages[-1]
        assert isinstance(last_message, AIMessage)
        tool_call = last_message.tool_calls[0]

        report = self.web_search_tool.invoke({'query': f'{tool_call["args"]["query"]}\n\n{REPORT_STRUCTURE}'})

        return {
            'messages': [
                RemoveMessage(id=state.messages[-1].id),
                ('ai', 'Now I need to know the output structure of this tool'),
                ('human', report),
            ]
        }

    def choose_group_node(self, state: CreateToolState):
        """最后一步，为工具选择分组"""
        group_choose_input = SelectGroupState(messages=state.messages)
        response = self.select_group_agent.invoke(group_choose_input)

        # 不更新消息
        return {'group_id': response['group_id']}

    def generate_tool_node(self, state: CreateToolState):
        result: ToolConfigSchema = tool_generator().invoke({'history': state.messages})

        with Session(engine) as session:
            group = group_crud.get(session, state.group_id)
            assert group is not None

            if state.checked:
                tag = tag_crud.get_by_name(session, 'AI Checked')
            else:
                tag = tag_crud.get_by_name(session, 'AI Unchecked')

            result_data = result.model_dump(exclude={'dynamic_params', 'file_mounts'})
            tool_data = ToolDefine(
                image_id=state.image_id,
                tool_type=ToolType.DOCKER,
                group_id=group.id,
                tags=[group.tag, tag],
                help_doc_id=state.doc_id,
                dynamic_params=result.dynamic_params,
                file_mounts=result.file_mounts,
                **result_data,
            )

            # 添加描述的向量索引
            embedding_model = qwen3_embedding()
            vector = embedding_model.embed_documents([tool_data.description])[0]
            tool_data.des_vector = vector

            tool_data = tool_crud.insert(session, tool_data)

            tool_node = ToolNodePublic.model_validate(tool_data.model_dump())
            tool_info = (
                f'<tool>\n'
                f'  <name>{tool_node.name}</name>\n'
                f'  <description>{tool_node.description}</description>\n'
                f'  <input_handles>\n'
                f'{handle_formatter(tool_node.input_handles, 4)}\n'
                f'  </input_handles>\n'
                f'  <output_handles>\n'
                f'{handle_formatter(tool_node.output_handles, 4)}\n'
                f'  </output_handles>\n'
                f'  <data>\n'
                f'    <tool_id>{tool_data.id}</tool_id>\n'
                f'    <args>{tool_node.static_params}<args>\n'
                f'  </data>\n'
                f'  <command>{tool_node.complete_command}</command>'
                f'</tool>'
            )

        return {
            'messages': [
                (
                    'human',
                    f'New tool node has been saved to the database for addition to the workflow: \n\n {tool_info}',
                )
            ]
        }

    def build(self, memory: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
        graph = StateGraph(CreateToolState, output_schema=CreateToolOutput)

        graph.add_node('select_image_node', self.select_image_node)
        graph.add_node('get_doc_node', self.get_doc_node)
        graph.add_node('get_output_report_node', self.get_output_report_node)
        graph.add_node('test_run_node', self.test_run_node)
        graph.add_node('web_search_node', self.web_search_node)
        graph.add_node('choose_group_node', self.choose_group_node)
        graph.add_node('generate_tool_node', self.generate_tool_node)

        graph.set_entry_point('select_image_node')
        graph.add_edge('select_image_node', 'get_doc_node')
        graph.add_edge('get_doc_node', 'get_output_report_node')
        graph.add_conditional_edges('get_output_report_node', self.how_to_get_output)
        graph.add_edge('test_run_node', 'choose_group_node')
        graph.add_edge('web_search_node', 'choose_group_node')
        graph.add_edge('choose_group_node', 'generate_tool_node')
        graph.add_edge('generate_tool_node', END)

        if memory is not None:
            return graph.compile(checkpointer=memory)
        return graph.compile()

    def visualize(self):
        graph = StateGraph(CreateToolState)

        graph.add_node('select_image_node', self.search_image_agent)
        graph.add_node('get_doc_node', self.get_doc_agent)
        graph.add_node('get_output_report_node', self.get_output_report_node)
        graph.add_node('test_run_node', self.test_run_agent)
        graph.add_node('web_search_node', self.web_search_node)
        graph.add_node('choose_group_node', self.select_group_agent)
        graph.add_node('generate_tool_node', self.generate_tool_node)

        graph.set_entry_point('select_image_node')
        graph.add_edge('select_image_node', 'get_doc_node')
        graph.add_edge('get_doc_node', 'get_output_report_node')
        graph.add_conditional_edges('get_output_report_node', self.how_to_get_output)
        graph.add_edge('test_run_node', 'choose_group_node')
        graph.add_edge('web_search_node', 'choose_group_node')
        graph.add_edge('choose_group_node', 'generate_tool_node')
        graph.add_edge('generate_tool_node', END)

        compiled_graph = graph.compile()

        mermaid_code = compiled_graph.get_graph(xray=1).draw_mermaid(
            curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
        )
        with open('images/create_tool_agent.md', 'w') as f:
            f.write(f'```mermaid\n{mermaid_code}\n```')


@tool
def select_image(image_id: int, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Select existing image in the database

    Args:
        image_id: selected image id
    """
    with Session(engine) as session:
        tool_image = image_crud.get(session, image_id)

    return Command(
        update={
            'image_id': image_id,
            'image': tool_image.image,
            'messages': [
                ToolMessage(
                    content='Available mirrors have been set up, and the next step can begin.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def select_doc(doc_id: int, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Select existing help document in the database

    Args:
        doc_id: selected doc id
    """

    return Command(
        update={
            'doc_id': doc_id,
            'messages': [
                ToolMessage(
                    content='The corresponding help document has been found, you can proceed to the next step.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def select_group(group_id: int, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Select the corresponding group for the tool

    Args:
        group_id: selected group id
    """

    return Command(
        update={
            'group_id': group_id,
            'messages': [
                ToolMessage(
                    content='The corresponding help document has been found, you can proceed to the next step.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def test_run():
    """Conduct a trial run with appropriate test data"""
    return


@tool
def view_output(
    direction: Literal['begin', 'end'], content: Annotated[str, InjectedState('last_output')], lines: int = 20
):
    """View the output of the last tool run in segments

    Args:
        direction: Reading order, begin means reading the first N lines, end means reading the last N lines
        lines: See how many lines, not too large, default is 20
    """
    if direction == 'begin':
        return '\n'.join(content.splitlines()[:lines])
    else:
        return '\n'.join(content.splitlines()[lines:])
