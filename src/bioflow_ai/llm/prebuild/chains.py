from typing import TYPE_CHECKING, cast

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.config import get_config
from pydantic import BaseModel

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.llm.models import get_long_context_model
from bioflow_ai.llm.observability.token_usage import store_token_usage
from bioflow_ai.llm.prebuild.json_output import json_output

if TYPE_CHECKING:
    from bioflow_ai.llm.llm_client import LLMMetadata


async def generate_title(history: list[BaseMessage], user_id: int) -> str:
    """总结对话并生成标题."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一个总结专家, 请根据给出的对话历史, 为对话生成一个简短的题目。'),
            MessagesPlaceholder(variable_name='history'),
            ('user', '为上述对话生成一个简短的题目。'),
        ]
    )
    llm = get_long_context_model()
    chain = prompt | llm
    result = await chain.ainvoke({'history': history})

    metadate = cast('LLMMetadata', llm.metadata)
    store_token_usage(
        'generate_title',
        user_id,
        get_config()['configurable'].get('thread_id', None),
        token_usage=result.usage_metadata,
        **metadate,
    )

    if isinstance(result.content, list):
        return f'{result.content}'
    return result.content.strip()


CHECK_SYSTEM = """You are a bioinformatics expert. You will be given the help documentation of a bioinformatics tool, along with a user's proposed parameter modification for a specific command. Your task is to determine whether the modification is valid and return the result.
If the user provides fabricated parameters or uses parameters incorrectly, the modification should also be rejected, with the correct usage pointed out.

## Input

The user's modification proposal contains the following three parts:

1. **Origin Command**: The original bioinformatics tool command before modification. Parts enclosed in square brackets `[]` are modifiable parameters; parts enclosed in parentheses `()` are non-modifiable parameters.
   - **Modifiable parameters**: Parameters that do not affect the command's core functionality in the default encapsulation. Users may freely modify or replace these.
   - **Non-modifiable parameters**: Parameters strictly tied to the command's functionality (especially its inputs and outputs) in the default encapsulation. Modifying these may alter the output or the overall behavior of the command.
   - If no parameters are specially marked in the Origin Command, it means no such parameters are defined in the original encapsulation.
   - Placeholder tokens such as `{{genome_dir}}`, `{{read1}}`, `{{output_dir}}`, etc. represent fixed file arguments and should be ignored entirely during validation.
2. **New Args**: The updated modifiable parameters provided by the user, which will fully replace the content currently inside the square brackets `[]`.
3. **Modify Reason**: The reason behind the user's proposed modification.

## Validation Logic

Read the tool documentation and analyze the user's new modification parameters to decide whether to approve the modification:
- If the modification includes parameters that do not actually exist (or incorrect parameter names), reject the modification. In the `feedback` field, state the correct parameter based on the user's original modification request, and attach an excerpt from the relevant parameter documentation.
- If the modification includes incorrect usage of parameters, reject the modification. In the `feedback` field, explain the correct usage of the relevant parameter and attach an excerpt from the documentation.
- If the parameters are correct but potentially conflict with **Non-modifiable parameters**, reject the modification and explain the situation in the `feedback` field.
- If the new modification parameters and their usage are both correct and do not conflict with **Non-modifiable parameters**, approve the modification. The `feedback` field may be left blank.

## Output
Your output must be a plain JSON string containing exactly the following two fields, with no additional characters:

- `approve`: Boolean. Indicates whether the modification is approved.
- `feedback`: String. When the modification is not approved, provide the specific reason, such as the correct parameter to use or the need to create a new encapsulation. When approved, this field can be left as an empty string.
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
    llm = get_long_context_model()
    metadata = cast('LLMMetadata', llm.metadata)
    chain = json_output(
        llm,
        CHECK_SYSTEM,
        ArgsCheckResult,
        metadata=metadata,
        name='check_update_args',
    )

    input_msg = (
        f'<ToolDoc>\n{doc}\n<ToolDoc>\n\n'
        f'Origin Command: `{complete_command}`\nNew Args: `{update_arg}`\nModify Reason: {reason}'
    )

    return await chain.ainvoke(
        input_msg,
        context=BaseContext(user_id=user_id),
    )
