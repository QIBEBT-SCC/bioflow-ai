from typing import Annotated, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool as tool_crud
from app.core.session import engine
from llm.reproduction.search_info_agent import SearchInfoAgent
from llm.tools.deepwiki import load_deepwiki_tools
from llm.utils.formatters import handle_formatter
from llm.generate_tool_config import tool_generator, ToolConfigSchema
from llm.models import qwen3_embedding, gemini_25_flash, gpt_41_mini
from llm.states import CreateToolState, CreateToolOutput, SearchImageState, GetDocState, SelectGroupState
from llm.tools.container import run_command
from llm.tools.database import (
    search_local_image,
    pull_and_save_image,
    list_docs,
    view_doc,
    save_doc,
    view_groups,
    add_group,
    select_image,
    select_doc,
    select_group,
)
from llm.tools.human import ask_human
from llm.tools.images import search_image
from app.models.tools import ToolType, ToolDefine
from app.schemas.tools import ToolNodePublic, ImageUpdate

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

GENERATE_COMMAND_SYS = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
To achieve this, you first need to generate an actual command to be used as a reference for node construction.

Your command construction needs to take into account the user's original requirements and be built according to the actual help documentation of the tool.
"""

GET_REPORT_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
To achieve this, your task is to **obtain the output structure of the corresponding tool** as the basis for encapsulation.

<ToolCalling>
To accurately understand the output and other relevant information produced by a tool's commands, we provide you with a series of tools. 
They can read GitHub repositories and ask questions, returning the desired information. 
Make full use of these tools to get all the information needed for your report.
</ToolCalling>

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
        self.struct_llm = gpt_41_mini(0.2)
        self.think_llm = gemini_25_flash(0.2)

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
        self.select_group_agent = create_react_agent(
            model=self.think_llm,
            tools=[view_groups, add_group, select_group],
            prompt=SELECT_GROUP_SYSTEM,
            state_schema=SelectGroupState,
        )

    async def select_image_node(self, state: CreateToolState):
        """首先搜索得到此工具的对应镜像"""
        if state.image_uid:
            # 如果提供了镜像，检查是否有GitHub链接
            writer = get_stream_writer()
            writer({'info': 'Searching for github link...'})

            with Session(engine) as session:
                tool_image = image_crud.get(session, str(state.image_uid))
                if not tool_image.homepage:
                    web_search_agent = SearchInfoAgent().build()
                    result = await web_search_agent.ainvoke(
                        {'messages': [('human', f'Search github and paper link for tool "{tool_image.name}"')]}
                    )
                    image_update = ImageUpdate(
                        uid=str(state.image_uid),
                        homepage=result['final_response'].github_url,
                        paper_link=result['final_response'].paper_url,
                    )
                    tool_image = image_crud.update(session, image_update)

                return {'image': tool_image.image, 'github_url': tool_image.homepage}

        writer = get_stream_writer()
        writer({'info': 'Searching for docker image...'})

        search_input = SearchImageState(messages=state.messages)
        response = await self.search_image_agent.ainvoke(search_input)

        return {'image_uid': response['image_uid'], 'image': response['image'], 'github_url': response['github_url']}

    async def get_doc_node(self, state: CreateToolState):
        """然后尝试获取指令对应的帮助文档"""
        writer = get_stream_writer()
        writer({'info': "Searching for tool's help document..."})

        doc_input = GetDocState(messages=state.messages, image_uid=state.image_uid, image=state.image)
        response = await self.get_doc_agent.ainvoke(doc_input)

        with Session(engine) as session:
            doc = doc_crud.get(session, str(response['doc_uid']))

        return {
            'messages': [
                ('ai', 'First I need the help documentation for this tool'),
                ('human', f'The following is the help documentation: \n\n{doc.content}'),
            ],
            'doc_uid': response['doc_uid'],
            'doc': doc.content,
        }

    async def generate_command_node(self, state: CreateToolState):
        """获取到帮助文档之后，就可以生成用于参考的命令"""
        writer = get_stream_writer()
        writer({'info': 'Generating example command...'})

        class CommandStructure(BaseModel):
            command: str = Field(
                description='Reference commands generated based on requirements and help documentation'
            )

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=GENERATE_COMMAND_SYS), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.think_llm.with_structured_output(CommandStructure)
        response = await chain.ainvoke({'history': state.messages})

        return {
            'messages': [
                ('ai', 'Great, it would be even better if you could give me an actual command for reference.'),
                ('human', response.command),
            ],
            'command': response.command,
        }

    @staticmethod
    def how_to_get_output(state: CreateToolState) -> Literal['get_output_report_node', 'web_search_node']:
        if state.github_url:
            return 'get_output_report_node'
        return 'web_search_node'

    async def get_output_report_node(self, state: CreateToolState):
        """尝试获取工具的输出结果（使用deepwiki）"""
        writer = get_stream_writer()
        writer({'info': 'Getting more information through deepwiki...'})

        deepwiki_tools = await load_deepwiki_tools()
        agent = create_react_agent(self.think_llm, deepwiki_tools, prompt=GET_REPORT_SYSTEM)

        response = await agent.ainvoke(
            {
                'messages': [
                    HumanMessage(
                        content=f'The tool command to be analyzed is: `{state.command}`, the github repo of tool is {state.github_url}'
                    )
                ]
            }
        )
        return {
            'messages': [
                (
                    'ai',
                    'Now, I also need to know about the output structure of the tool and some other information not visible in the documentation.',
                ),
                ('human', response['messages'][-1].content),
            ]
        }

    async def web_search_node(self, state: CreateToolState):
        """没有现成的测试数据，联网搜索"""
        writer = get_stream_writer()
        writer({'info': 'Getting more information by web search...'})

        # TODO 先中断，后面再补
        answer = interrupt(state.messages[-1].content)
        return f'Human feedback is: "{answer}"'

    async def choose_group_node(self, state: CreateToolState):
        """最后一步，为工具选择分组"""
        writer = get_stream_writer()
        writer({'info': 'Assigning groups to tools...'})

        group_choose_input = SelectGroupState(messages=state.messages)
        response = await self.select_group_agent.ainvoke(group_choose_input)

        # 不更新消息
        return {'group_id': response['group_id']}

    async def generate_tool_node(self, state: CreateToolState, config: RunnableConfig):
        """生成工具配置"""
        writer = get_stream_writer()
        writer({'info': 'Generating tool config...'})

        result: ToolConfigSchema = await tool_generator().ainvoke({'history': state.messages})

        with Session(engine) as session:
            group = group_crud.get(session, state.group_id)
            assert group is not None

            if state.github_url:
                tag = tag_crud.get_by_name(session, 'AI Checked')
            else:
                tag = tag_crud.get_by_name(session, 'AI Unchecked')

            result_data = result.model_dump(exclude={'dynamic_params', 'file_mounts'})
            tool_data = ToolDefine(
                image_uid=state.image_uid,
                tool_type=ToolType.DOCKER,
                group_id=group.id,
                tags=[group.tag, tag],
                help_doc_uid=state.doc_uid,
                dynamic_params=result.dynamic_params,
                file_mounts=result.file_mounts,
                **result_data,
            )

            if not config['configurable'].get('auto_save'):
                return {'json_response': tool_data.model_dump()}

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

    def build(self, memory: Optional[BaseCheckpointSaver] = None, visualize: bool = False) -> CompiledStateGraph:
        graph = StateGraph(CreateToolState, output_schema=CreateToolOutput)

        graph.add_node('select_image_node', self.select_image_node)
        graph.add_node('get_doc_node', self.get_doc_node)
        graph.add_node('generate_command_node', self.generate_command_node)
        graph.add_node('get_output_report_node', self.get_output_report_node)
        graph.add_node('web_search_node', self.web_search_node)
        graph.add_node('choose_group_node', self.choose_group_node)
        graph.add_node('generate_tool_node', self.generate_tool_node)

        graph.set_entry_point('select_image_node')
        graph.add_edge('select_image_node', 'get_doc_node')
        graph.add_edge('get_doc_node', 'generate_command_node')
        graph.add_conditional_edges('generate_command_node', self.how_to_get_output)
        graph.add_edge('get_output_report_node', 'choose_group_node')
        graph.add_edge('web_search_node', 'choose_group_node')
        graph.add_edge('choose_group_node', 'generate_tool_node')
        graph.add_edge('generate_tool_node', END)

        if memory is not None:
            compiled_graph = graph.compile(checkpointer=memory)
        else:
            compiled_graph = graph.compile()

        if visualize:
            mermaid_code = compiled_graph.get_graph().draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/create_tool_agent.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph


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
