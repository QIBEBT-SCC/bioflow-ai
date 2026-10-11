"""Convert durable domain messages without dropping their visual content."""

from html import escape
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

import bioflow_ai.crud.agent as agent_crud
from bioflow_ai.models.agent import AgentRun
from bioflow_ai.schemas.agent import ImageMessagePart, WorkflowNodeMessagePart
from bioflow_ai.services.agents.chat_attachment import image_reference_content

MISSING_INPUT_ERROR = 'Agent run input message is missing.'


def load_langchain_messages(session: Session, run: AgentRun) -> tuple[list[BaseMessage], HumanMessage]:
    """Restore durable user images along with the existing text conversation."""
    history: list[BaseMessage] = []
    current: HumanMessage | None = None
    for message in agent_crud.list_messages(session, run.session_uid):
        content = parts_content(message.parts)
        if not content:
            continue
        if message.role == 'assistant':
            history.append(AIMessage(content=content, id=str(message.uid)))
        else:
            node = next((part for part in message.parts if part.get('type') == 'workflow_node'), None)
            converted = HumanMessage(
                content=content,
                id=str(message.uid),
                additional_kwargs={'workflow_node': node} if node is not None else {},
            )
            history.append(converted)
            if message.run_uid == run.uid and current is None:
                current = converted
    if current is None:
        raise ValueError(MISSING_INPUT_ERROR)
    return history, current


def parts_content(parts: list[dict[str, Any]]) -> str | list[str | dict]:
    """Keep text-only compatibility and structured blocks for image messages."""
    content: list[str | dict] = []
    for part in parts:
        if part.get('type') == 'text':
            content.append({'type': 'text', 'text': str(part.get('text', ''))})
        elif part.get('type') == 'image':
            content.extend(image_reference_content(ImageMessagePart.model_validate(part)))
        elif part.get('type') == 'pdf':
            content.append({'type': 'text', 'text': f'<PDF id="{escape(str(part["file_id"]), quote=True)}">'})
        elif part.get('type') == 'workflow_node':
            node = WorkflowNodeMessagePart.model_validate(part)
            content.append(
                {
                    'type': 'text',
                    'text': '\n<workflow_node_attachment>\n'
                    + node.model_dump_json()
                    + '\n</workflow_node_attachment>\n',
                }
            )
    if not any(part.get('type') == 'image' for part in parts):
        return ''.join(str(part['text']) for part in content if isinstance(part, dict))
    return content


def wrap_request(content: str | list[str | dict], opening: str, closing: str) -> str | list[str | dict]:
    """Wrap a request in text delimiters while preserving its image blocks."""
    if isinstance(content, str):
        return f'{opening}{content}{closing}'
    return [{'type': 'text', 'text': opening}, *content, {'type': 'text', 'text': closing}]
