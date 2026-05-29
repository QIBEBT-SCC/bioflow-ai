from typing import TYPE_CHECKING, cast

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.config import get_config
from pydantic import BaseModel

from llm.models import get_long_context_model
from llm.utils.token import store_token_usage

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata
    from llm.states import StructuredResponse


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
In the current system, bioinformatics tools are encapsulated as draggable nodes. Since the inputs and outputs of each node are fixed, the input and output files of a node corresponding to an original command are also fixed. Therefore, if the user's new parameters introduce new input files or produce new output files, the modification should NOT be approved, and the user should be informed to create a new encapsulation instead.
If the user provides fabricated parameters or uses parameters incorrectly, the modification should also be rejected, with the correct usage pointed out.

## Input

The user's modification proposal contains the following three parts:

1. **Origin Command**: The original bioinformatics tool command before modification. Parts enclosed in square brackets `[]` are modifiable parameters; parts enclosed in parentheses `()` are non-modifiable parameters.
   - **Modifiable parameters**: Parameters that do not affect the node's core functionality in the default encapsulation. Users may freely modify or replace these.
   - **Non-modifiable parameters**: Parameters strictly tied to the node's functionality (especially its inputs and outputs) in the default encapsulation. Modifying these may alter the output or the overall behavior of the node.
   - If no parameters are specially marked in the Origin Command, it means no such parameters are defined in the original encapsulation.
   - Placeholder tokens such as `{genome_dir}`, `{read1}`, `{output_dir}`, etc. represent fixed file arguments and should be ignored entirely during validation.
2. **New Args**: The updated modifiable parameters provided by the user, which will fully replace the content currently inside the square brackets `[]`.
3. **Modify Reason**: The reason behind the user's proposed modification.

## Validation Logic

The modification should NOT be approved if any of the following conditions are met, and a specific reason must be returned:
- The new parameters include fabricated arguments that do not exist in the help documentation.
- A parameter is used in a way that is inconsistent with the help documentation.
- According to the help documentation, a different parameter should be used to achieve the user's stated goal.
- The modification satisfies the user's requirement but conflicts with one or more **non-modifiable parameters**.
- One or more of the new parameters accept a file as input according to the help documentation. File-based parameters should not be adjusted through modifiable parameters — the user should create a new node encapsulation instead.

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
    result: StructuredResponse[ArgsCheckResult] = await chain.ainvoke(  # type: ignore
        {
            'doc': doc,
            'complete_command': complete_command,
            'update_arg': update_arg,
            'reason': reason,
        }
    )

    metadate = cast('LLMMetadata', llm.metadata)
    store_token_usage(
        'check_update_args',
        user_id,
        get_config()['configurable'].get('thread_id', None),
        token_usage=result['raw'].usage_metadata,
        **metadate,
    )

    return result['parsed']
