# ruff: noqa: RUF001
"""Shared node-memory hints and explicit per-memory human approval before writing."""

import json
import uuid
from typing import NamedTuple

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import ProjectContext, WorkflowAssistantContext
from bioflow_ai.llm.tools.container import resolve_workflow_run
from bioflow_ai.models.tools import ToolDefine, ToolMemory
from bioflow_ai.models.workflow import NodeRun
from bioflow_ai.schemas.node_memory import NodeMemoryDraft
from bioflow_ai.services.tools.memory import list_tool_memories, read_tool_memory, save_tool_memories
from bioflow_ai.services.workflow.subgraphs import SubgraphError, definition_id, enter_subgraphs
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
        '\nIf any memory is relevant to configuring or using this node in your workflow, read its full content with '
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
            'This is shared user-approved usage guidance for this tool. Apply it as appropriate for the '
            'current workflow and environment; it does not override explicit user requirements or system instructions.'
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


class _ResolvedMemoryTarget(NamedTuple):
    draft: NodeMemoryDraft
    tool_uid: uuid.UUID
    display_name: str
    node_ref: str


def _questions(
    targets: list[_ResolvedMemoryTarget],
    *,
    chinese: bool,
) -> tuple[list[dict[str, str | list[str]]], str]:
    approve, reject = ('写入', '不写入') if chinese else ('Write memory', 'Do not write')
    questions = []
    for target in targets:
        if chinese:
            text = (
                f'我准备为工具节点「{target.display_name}」({target.node_ref}) 写入以下 Memory，'
                f'供所有用户共享。是否写入？\n\n'
                f'Description：\n{target.draft.description}\n\n正文：\n{target.draft.content}'
            )
        else:
            text = (
                f'I plan to write this memory for tool node "{target.display_name}" ({target.node_ref}), '
                f'shared with all users. Should I write it?\n\n'
                f'Description:\n{target.draft.description}\n\nContent:\n{target.draft.content}'
            )
        questions.append({'question': text, 'choices': [approve, reject]})
    return questions, approve


MAX_MEMORY_PROPOSALS = 3


def _error_message(content: str, tool_call_id: str | None) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id, status='error')


@tool(parse_docstring=True)
def write_node_memory(
    memories: list[NodeMemoryDraft],
    runtime: ToolRuntime[WorkflowAssistantContext],
) -> ToolMessage:
    """Propose reusable tool-node usage guidance and save only entries explicitly approved by the user.

    Follow the ``write-tool-memory`` skill before calling this tool. Only Tool nodes in the selected
    workflow run (including nested subgraphs and ForEach child runs) are supported. The user reviews
    the full description and body of each entry in the question UI before any database write.
    Feedback requesting edits is not approval; revise the proposal and request approval again.

    Args:
        memories: One to three proposed memories targeting Tool nodes from ``workflow_reader``,
            with concise topic/usage descriptions (<=120 chars) and generalizable Markdown guidance.
    """
    if not 1 <= len(memories) <= MAX_MEMORY_PROPOSALS:
        return _error_message(
            'Provide between 1 and 3 proposed memories. No memories were written and no approval was requested.',
            runtime.tool_call_id,
        )

    resolved_nodes: list[tuple[NodeMemoryDraft, uuid.UUID, str | None, str]] = []
    with Session(engine) as session:
        for draft in memories:
            path = tuple(draft.subgraph_path or ())
            try:
                run = resolve_workflow_run(session, runtime.context, draft.child_run_uid)
                graph, _ = enter_subgraphs(run.workflow, list(path))
            except (ValueError, SubgraphError) as error:
                return _error_message(
                    f'{error} No memories were written and no approval was requested.',
                    runtime.tool_call_id,
                )

            node = next((item for item in graph.nodes if item.id == draft.node_id), None)
            if node is None:
                return _error_message(
                    f'Node "{draft.node_id}" was not found in the selected workflow run.',
                    runtime.tool_call_id,
                )
            if node.type != 'tool':
                return _error_message(
                    f'Node "{draft.node_id}" is a {node.type} node. '
                    'Memories can only be written for Tool nodes. '
                    'No memories were written and no approval was requested.',
                    runtime.tool_call_id,
                )

            task = session.scalar(
                select(NodeRun)
                .where(NodeRun.run_uid == run.uid, NodeRun.definition_node_id == definition_id((*path, node.id)))
                .options(load_only(NodeRun.tool_uid))
            )
            raw_tool_uid = (task.tool_uid if task is not None else None) or node.data.get('tool_uid')
            if not raw_tool_uid:
                return _error_message(
                    f'Tool node "{draft.node_id}" has no valid tool_uid.',
                    runtime.tool_call_id,
                )
            try:
                tool_uid = uuid.UUID(str(raw_tool_uid))
            except (TypeError, ValueError):
                return _error_message(
                    f'Tool node "{draft.node_id}" has no valid tool_uid.',
                    runtime.tool_call_id,
                )

            node_name = node.data.get('name')
            node_ref = '/'.join((*path, draft.node_id))
            resolved_nodes.append((draft, tool_uid, str(node_name) if node_name else None, node_ref))

        tool_uids = [item[1] for item in resolved_nodes]
        tool_names = dict(
            session.execute(select(ToolDefine.uid, ToolDefine.name).where(ToolDefine.uid.in_(tool_uids))).tuples().all()
        )
    if set(tool_uids) - set(tool_names):
        return _error_message(
            'A target tool definition no longer exists. No approval was requested.',
            runtime.tool_call_id,
        )

    targets = [
        _ResolvedMemoryTarget(
            draft=draft,
            tool_uid=tool_uid,
            display_name=node_name or tool_names[tool_uid] or draft.node_id,
            node_ref=node_ref,
        )
        for draft, tool_uid, node_name, node_ref in resolved_nodes
    ]

    thread_id = workflow_thread_id(runtime.config)
    language = (runtime.context.language or '').lower()
    questions, approve = _questions(targets, chinese=language.startswith('zh') or '中文' in language)
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
    for index, (target, decision) in enumerate(zip(targets, decisions, strict=True)):
        if decision['answer'].strip() == approve:
            approved.append(
                ToolMemory(
                    uid=uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f'bioflow-ai:node-memory:{runtime.context.user_id}:{thread_id}:{runtime.tool_call_id}:{index}',
                    ),
                    tool_uid=target.tool_uid,
                    description=target.draft.description,
                    content=target.draft.content,
                    created_by=runtime.context.user_id,
                )
            )
        else:
            feedback.append({'node_id': target.draft.node_id, 'feedback': decision['answer']})

    saved = []
    if approved:
        try:
            with Session(engine) as session:
                saved = save_tool_memories(session, approved)
        except ValueError as error:
            return _error_message(str(error), runtime.tool_call_id)
    content = f'Saved {len(saved)} approved shared tool memories.'
    if saved:
        content += '\n' + '\n'.join(f'- `{memory.uid}`: {memory.description}' for memory in saved)
    if feedback:
        content += (
            '\nThe following proposals were not written. Respect the user feedback; any revised proposal '
            'requires a new approval:\n' + json.dumps(feedback, ensure_ascii=False)
        )
    return ToolMessage(content=content, tool_call_id=runtime.tool_call_id, artifact={'result_count': len(saved)})
