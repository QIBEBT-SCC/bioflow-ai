from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.core.llm_chat import gemini_25_flash
from app.llm.prompt.function_prompt import TOOL_GEN_SYSTEM, TOOL_GEN_USER
from app.schemas.tools import AIGenTool


async def generate_tool_config(tool_name: str, help_doc: str, user_require: str) -> AIGenTool:
    """use llm generate tool config

    Args:
        tool_name: 工具名称
        help_doc: 工具的help文档
        user_require: 用户额外的要求

    Returns:
        AIGenTool
    """
    llm = gemini_25_flash()
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=TOOL_GEN_SYSTEM), ('human', TOOL_GEN_USER)])
    chain = prompt | llm.with_structured_output(AIGenTool, method='json_mode')

    output: AIGenTool = await chain.ainvoke(
        {'tool_name': tool_name, 'help_doc': help_doc, 'user_require': user_require}
    )
    return output
