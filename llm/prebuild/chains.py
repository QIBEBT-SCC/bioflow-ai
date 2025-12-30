from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm.models import get_long_context_model
from llm.utils.token import store_token_usage


async def generate_title(history: list[BaseMessage], user_id: int) -> str:
    """总结对话并生成标题

    Args:
        history:
        user_id:
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一个总结专家，请根据给出的对话历史，为对话生成一个简短的题目。'),
            MessagesPlaceholder(variable_name='history'),
            ('user', '为对话生成一个简短的题目。'),
        ]
    )
    llm = get_long_context_model()
    chain = prompt | llm
    result = await chain.ainvoke({'history': history})
    store_token_usage('generate_title', user_id=user_id, token_usage=result.usage_metadata, **llm.metadata)

    return result.content.strip()
