from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm.models import gemini_25_flash


def conclude_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一个总结专家，请根据给出的对话历史，为对话生成一个简短的题目。'),
            MessagesPlaceholder(variable_name='history'),
            ('user', '为对话生成一个简短的题目。'),
        ]
    )
    llm = gemini_25_flash(temperature=0.5)
    return prompt | llm
