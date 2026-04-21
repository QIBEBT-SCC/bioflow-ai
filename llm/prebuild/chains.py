from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from llm.models import get_long_context_model
from llm.utils.token import store_token_usage


async def generate_title(history: list[BaseMessage], user_id: int) -> str:
    """总结对话并生成标题."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一个总结专家, 请根据给出的对话历史, 为对话生成一个简短的题目。'),
            MessagesPlaceholder(variable_name='history'),
            ('user', '为对话生成一个简短的题目。'),
        ]
    )
    llm = get_long_context_model()
    chain = prompt | llm
    result = await chain.ainvoke({'history': history})
    store_token_usage('generate_title', user_id=user_id, token_usage=result.usage_metadata, **llm.metadata)

    return result.content.strip()


CHECK_SYSTEM = """你是一名生物信息学专家。你将会看到一个生物信息工具的帮助文档, 以及一个用户对于某一段指令的参数修改决策。你的任务是判断用户此次修改是否是一个正确的操作, 并返回结果。

## 输入
用户的修改决策中包括以下三部分信息:
Origin Command: 未修改前原始的生物信息工具命令, 其中用中括号括起来的部分, 是用户可以修改的参数。用圆括号括起来的部分, 是用户不能够修改的参数。
New Args: 用户更新的可变参数, 用于替代现在中括号中的参数。
Modify Reason: 用户是基于什么原因提出了这个修改

## 判断逻辑
用户新给的参数当中如果存在以下几种情况, 则不应该批准用户的此次修改, 并且返回具体的原因:
- 存在帮助文档中不存在的虚假参数
- 某个参数的使用方法与帮助文档中的不一致
- 根据帮助文档显示, 想要满足用户的修改需求, 实际上应该使用别的参数
- 参数修改能够满足用户的要求, 但是其中某一个或几个参数接受的输入是一个文件
- 参数修改能够满足用户的需求, 但是与不可修改参数冲突

背景说明:
实际上在当前系统中, 我们以可拖拽节点的形式对生物信息工具进行了封装。由于节点能够接入的接收输入输出是固定的, 因此一个原始命令对应的节点, 其输入文件和输出文件也是固定的。而可修改参数是那些不会影响到文件输入输出的参数, 因此, 如果用户给出的新参数中涉及到新的输入文件或者会产生新的输出文件, 此时不应该批准用户的此次操作, 并告知用户应当新建一个封装。

## 输出
你的输出应该是一个包括以下两个字段的纯 JSON 字符串, 不产生其他任何字符:
- `approve`: 布尔类型变量, 表明是否批准用户的此次修改
- `feedback`: 字符串变量。当不批准用户的修改时, 需要告知具体的原因。如实际的正确参数, 或需要新建封装等等。
"""


class ArgsCheckResult(BaseModel):
    approve: bool
    feedback: str


async def check_update_args(
    doc: str,
    complete_command: str,
    update_arg: str,
    reason: str,
    user_id: int,
) -> ArgsCheckResult:
    """检查工具参数更新是否合法."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CHECK_SYSTEM),
            (
                'user',
                '<ToolDoc>\n{doc}\n<ToolDoc>\n\nOrigin Command: `{complete_command}`\nNew Args: `{update_arg}`\nModify Reason: {reason}',
            ),
        ]
    )
    llm = get_long_context_model()
    chain = prompt | llm.with_structured_output(ArgsCheckResult, include_raw=True)
    result = await chain.ainvoke(
        {'doc': doc, 'complete_command': complete_command, 'update_arg': update_arg, 'reason': reason}
    )
    store_token_usage('check_update_args', user_id=user_id, token_usage=result['raw'].usage_metadata, **llm.metadata)

    return result['parsed']
