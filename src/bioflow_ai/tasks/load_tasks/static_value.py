from typing import Literal

from bioflow_ai.schemas.workflow_config import ToolOutput


def set_value(
    *,
    value: str,
    value_type: Literal['string'],
) -> ToolOutput:
    """返回变量值."""
    match value_type:
        case 'string':
            return ToolOutput(result={'value': value})
        # case 'number':
        #     if '.' in value:
        #         return ToolOutput(result={'value': float(value)})
        #     return ToolOutput(result={'value': int(value)})
        case _:
            raise ValueError
