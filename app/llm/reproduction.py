from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.llm_chat import gpt_41_mini
from app.llm.prompt.reproduction_prompt import GET_TOOL_SYSTEM, GET_TOOL_USER


class BaseTool(BaseModel):
    name: str = Field(description='Exact name of the tool')
    paper_text:str=Field(description='The original sentence in the paper referring to this tool')


class ToolsOutput(BaseModel):
    """Bioinformatics tools used in the literature"""

    tools: list[BaseTool] = Field(description='All tools used')


async def get_tool_names(paper_content: str) -> ToolsOutput:
    """从原始文献中提取出所涉及到的工具

    Args:
        paper_content: 论文的具体内容

    Returns:
        ToolsOutput
    """
    llm = gpt_41_mini()
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=GET_TOOL_SYSTEM), ('human', GET_TOOL_USER)])
    chain = prompt | llm.with_structured_output(ToolsOutput)

    output: ToolsOutput = await chain.ainvoke({'paper_content': paper_content})
    return output
