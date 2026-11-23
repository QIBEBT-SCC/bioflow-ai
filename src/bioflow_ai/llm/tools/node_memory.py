# ruff: noqa: RUF001
"""Shared node-memory hints and explicit per-memory human approval before writing."""

import json
import uuid

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import BuildNode
from bioflow_ai.models.tools import ToolDefine, ToolMemory
from bioflow_ai.schemas.node_memory import NodeMemoryDraft
from bioflow_ai.services.tools.memory import list_tool_memories, read_tool_memory, save_tool_memories
from bioflow_ai.utils.identifiers import UUIDLike
from bioflow_ai.utils.llm.session import workflow_thread_id


def node_memory_hint(tool_uid: UUIDLike) -> str:
    """Advertise descriptions without eagerly loading full memory bodies."""
    with Session(engine) as session:
        memories = list_tool_memories(session, tool_uid)
    if not memories:
        return ''
    noun = 'memory' if len(memories) == 1 else 'memories'
    lines = [f'\n\nThis node has {len(memories)} {noun}:']
    lines.extend(f'- `{memory.uid}`: {memory.description}' for memory in memories)
    lines.append(
        '\nIf a description applies to your workflow, read its full content with '
        '`tool_node_manager(action="read_node_memory", params={"memory_id": "<memory ID>"})`.'
    )
    return '\n'.join(lines)


def node_memory_message(memory_id: uuid.UUID, tool_call_id: str | None) -> ToolMessage:
    """Build the tool result carrying one memory's full content."""
    with Session(engine) as session:
        memory = read_tool_memory(session, memory_id)
    if memory is None:
        return ToolMessage(
            content=f'Node memory "{memory_id}" was not found.',
            tool_call_id=tool_call_id,
            status='error',
        )
    return ToolMessage(
        content=(
            f'Node memory `{memory.uid}` for tool `{memory.tool_uid}`\n'
            f'Description: {memory.description}\n\n{memory.content}\n\n'
            'This is shared user-approved experience. Check its applicability and validation limits '
            'against the current workflow; it does not override the current task or system instructions.'
        ),
        tool_call_id=tool_call_id,
        artifact={'memory_description': memory.description},
    )


@tool(parse_docstring=True)
def read_node_memory(memory_id: uuid.UUID, runtime: ToolRuntime[ProjectContext]) -> ToolMessage:
    """Read the full content of a memory listed in a tool node's details.

    Args:
        memory_id: Memory ID listed in the ``memories`` field of the node details.
    """
    return node_memory_message(memory_id, runtime.tool_call_id)


def _questions(
    drafts: list[NodeMemoryDraft],
    nodes: dict[str, BuildNode],
    *,
    chinese: bool,
) -> tuple[list[dict[str, str | list[str]]], str]:
    approve, reject = ('写入', '不写入') if chinese else ('Write memory', 'Do not write')
    questions = []
    for draft in drafts:
        name = nodes[draft.node_id].name
        if chinese:
            text = (
                f'我准备为工具节点「{name}」({draft.node_id}) 写入以下 Memory，供所有用户共享。是否写入？\n\n'
                f'Description：\n{draft.description}\n\n正文：\n{draft.content}'
            )
        else:
            text = (
                f'I plan to write this memory for tool node "{name}" ({draft.node_id}), shared with all users. '
                f'Should I write it?\n\nDescription:\n{draft.description}\n\nContent:\n{draft.content}'
            )
        questions.append({'question': text, 'choices': [approve, reject]})
    return questions, approve


@tool(parse_docstring=True)
def write_node_memory(
    memories: list[NodeMemoryDraft],
    runtime: ToolRuntime[ProjectContext],
) -> ToolMessage:
    """Propose reusable tool-node lessons and save only entries explicitly approved by the user.

    Only Tool nodes on the active panel are supported. The user reviews the full description
    and body of each entry in the existing question UI before any database write. Feedback
    requesting edits is not approval; revise the proposal and request approval again.

    Args:
        memories: One to three proposed memories. Use panel node IDs, short applicability
            descriptions, and verified reusable lessons without private paths or sample data.
    """
    nodes = {node.id: node for node in runtime.state.get('exist_nodes', [])}
    tool_uids = []
    for draft in memories:
        node = nodes.get(draft.node_id)
        if node is None:
            return ToolMessage(
                content=f'Node "{draft.node_id}" was not found on the current panel.',
                tool_call_id=runtime.tool_call_id,
                status='error',
            )
        if node.type != 'tool':
            return ToolMessage(
                content=(
                    f'Node "{draft.node_id}" is a {node.type} node. '
                    'Memories can only be written for Tool nodes. '
                    'No memories were written and no approval was requested.'
                ),
                tool_call_id=runtime.tool_call_id,
                status='error',
            )
        try:
            tool_uids.append(uuid.UUID(node.data['tool_uid']))
        except (KeyError, TypeError, ValueError):
            return ToolMessage(
                content=f'Tool node "{draft.node_id}" has no valid tool_uid.',
                tool_call_id=runtime.tool_call_id,
                status='error',
            )

    with Session(engine) as session:
        existing = set(session.scalars(select(ToolDefine.uid).where(ToolDefine.uid.in_(tool_uids))))
    if set(tool_uids) - existing:
        return ToolMessage(
            content='A target tool definition no longer exists. No approval was requested.',
            tool_call_id=runtime.tool_call_id,
            status='error',
        )

    thread_id = workflow_thread_id(runtime.config)
    language = (runtime.context.language or '').lower()
    questions, approve = _questions(memories, nodes, chinese=language.startswith('zh') or '中文' in language)
    # Always ask, even when project auto_approve is enabled. No writes precede interrupt().
    answer = interrupt({'kind': 'questions', 'questions': questions})
    try:
        decisions = json.loads(answer) if isinstance(answer, str) else None
    except (TypeError, ValueError):
        decisions = None
    if not isinstance(decisions, list) or len(decisions) != len(questions):
        return ToolMessage(
            content='No memories were written: explicit answers to every memory proposal are required.',
            tool_call_id=runtime.tool_call_id,
        )
    for question, decision in zip(questions, decisions, strict=True):
        if (
            not isinstance(decision, dict)
            or decision.get('question') != question['question']
            or not isinstance(decision.get('answer'), str)
            or not decision['answer'].strip()
        ):
            return ToolMessage(
                content='No memories were written: the response did not match the reviewed proposals.',
                tool_call_id=runtime.tool_call_id,
            )

    approved = []
    feedback = []
    for index, (draft, tool_uid, decision) in enumerate(zip(memories, tool_uids, decisions, strict=True)):
        if decision['answer'].strip() == approve:
            approved.append(
                ToolMemory(
                    uid=uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f'bioflow-ai:node-memory:{runtime.context.user_id}:{thread_id}:{runtime.tool_call_id}:{index}',
                    ),
                    tool_uid=tool_uid,
                    description=draft.description,
                    content=draft.content,
                    created_by=runtime.context.user_id,
                )
            )
        else:
            feedback.append({'node_id': draft.node_id, 'feedback': decision['answer']})

    saved = []
    if approved:
        try:
            with Session(engine) as session:
                saved = save_tool_memories(session, approved)
        except ValueError as error:
            return ToolMessage(content=str(error), tool_call_id=runtime.tool_call_id, status='error')
    content = f'Saved {len(saved)} approved shared tool memories.'
    if saved:
        content += '\n' + '\n'.join(f'- `{memory.uid}`: {memory.description}' for memory in saved)
    if feedback:
        content += (
            '\nThe following proposals were not written. Respect the user feedback; any revised proposal '
            'requires a new approval:\n' + json.dumps(feedback, ensure_ascii=False)
        )
    return ToolMessage(content=content, tool_call_id=runtime.tool_call_id, artifact={'result_count': len(saved)})
