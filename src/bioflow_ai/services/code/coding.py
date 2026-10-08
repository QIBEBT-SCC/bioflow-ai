"""Stream one native CODE editing turn against the latest editor baseline."""

import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_core.messages import BaseMessage, HumanMessage

from bioflow_ai.llm.context import CodingContext
from bioflow_ai.llm.observability.execution_logging import AgentRunLogger
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.code_agent import CodeAgentProposal
from bioflow_ai.services.code.workspace import (
    baseline_hash,
    create_workspace,
    proposal_diff,
    read_candidate,
    write_baseline,
)

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class CodingAgent(Protocol):
    def astream_events(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield native model and tool events, ending with the root graph result."""
        ...


@dataclass(frozen=True)
class CodingTurnResult:
    """Only a complete turn can update conversation history and offer a candidate."""

    messages: list[BaseMessage]
    proposal: CodeAgentProposal | None


def message_text(message: Any) -> str:
    """Extract visible text from message objects or provider content blocks."""
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            item if isinstance(item, str) else str(item.get('text', ''))
            for item in content
            if isinstance(item, (str, dict))
        )
    return ''


async def _keep_workspace_alive(workspace: Path) -> None:
    while True:
        (workspace / '.lease').touch()
        await asyncio.sleep(10)


async def run_coding_turn(
    request: str,
    *,
    node_type: CodeNodeType,
    user_id: int,
    source: str = '',
    dependencies: list[str] | None = None,
    language: str | None = None,
    history: list[BaseMessage] | None = None,
    turn_id: str | None = None,
    session_id: str | None = None,
    on_progress: ProgressCallback | None = None,
    agent: CodingAgent | None = None,
    input_contract: list[dict[str, Any]] | None = None,
    execution_log: AgentRunLogger | None = None,
) -> CodingTurnResult:
    """Rebuild files, stream native events, validate the candidate, and clean up on cancellation."""
    turn_id = turn_id or str(uuid.uuid4())
    run_log = execution_log or AgentRunLogger(
        'simple-coding', turn_id, session_id=session_id, thread_id=session_id or turn_id
    )
    observation = run_log.observe() if execution_log is None else nullcontext()
    with observation:
        if not request.strip():
            msg = 'A coding request is required.'
            raise ValueError(msg)
        baseline_dependencies = list(dependencies or [])
        workspace = create_workspace(node_type)
        lease = asyncio.create_task(_keep_workspace_alive(workspace))
        try:
            write_baseline(workspace, node_type, source, baseline_dependencies)
            context = CodingContext(user_id=user_id, language=language, workspace_path=workspace, node_type=node_type)
            prompt = (
                f'Current target CODE node type: {node_type.value}\n'
                'The workspace was rebuilt from the latest editor source and dependencies for this turn. '
                'Previous output/ fixtures are absent; recreate any needed small fixtures with apply_patch.\n\n'
                f'User request:\n{request}'
            )
            if input_contract is not None:
                prompt += '\n\nAuthoritative input contract:\n' + json.dumps(input_contract, ensure_ascii=False)
            messages = [*(history or []), HumanMessage(prompt)]
            if agent is None:
                from bioflow_ai.llm.agents.coding_agent.simple_coding import build_simple_coding_agent

                agent = cast('CodingAgent', build_simple_coding_agent())
            final: dict[str, Any] | None = None
            streamed_text = False
            async for event in agent.astream_events(
                {'messages': messages},
                config={'configurable': {'thread_id': session_id or turn_id}, 'recursion_limit': 100},
                context=context,
                version='v2',
            ):
                run_log.native_event(event)
                kind = event['event']
                data = event.get('data', {})
                if kind == 'on_chain_end' and not event.get('parent_ids'):
                    final = data['output']
                elif kind == 'on_chat_model_stream':
                    text = message_text(data.get('chunk'))
                    if text and on_progress is not None:
                        streamed_text = True
                        await on_progress('message.delta', {'text': text})
                elif kind in {'on_tool_start', 'on_tool_end', 'on_tool_error'} and on_progress is not None:
                    arguments = data.get('input') or {}
                    detail = ''
                    if isinstance(arguments, dict):
                        detail = str(arguments.get('command') or arguments.get('skill_name') or '')
                    await on_progress(
                        'tool.updated',
                        {
                            'call_id': str(event['run_id']),
                            'name': event['name'],
                            'status': {
                                'on_tool_start': 'running',
                                'on_tool_end': 'completed',
                                'on_tool_error': 'failed',
                            }[kind],
                            'detail': detail[:16_000],
                            'output': message_text(data.get('output', data.get('error', '')))[:16_000],
                        },
                    )
            if final is None or final.get('__interrupt__'):
                msg = 'The coding agent did not return a complete conversation.'
                raise ValueError(msg)
            result_messages = final['messages']
            if not streamed_text and on_progress is not None:
                text = '\n\n'.join(
                    message_text(message)
                    for message in result_messages[len(messages) :]
                    if message.type == 'ai' and message_text(message)
                )
                if text:
                    await on_progress('message.delta', {'text': text})
            candidate_source, candidate_dependencies, warnings = read_candidate(
                workspace,
                node_type,
                allow_empty=not source.strip(),
            )
            diff = proposal_diff(node_type, source, candidate_source, baseline_dependencies, candidate_dependencies)
            proposal = None
            if candidate_source.strip() and diff:
                proposal = CodeAgentProposal(
                    id=str(uuid.uuid4()),
                    turn_id=turn_id,
                    baseline_hash=baseline_hash(node_type, source, baseline_dependencies),
                    source=candidate_source,
                    dependencies=candidate_dependencies,
                    diff=diff,
                    warnings=warnings,
                )
            return CodingTurnResult(messages=result_messages, proposal=proposal)
        finally:
            lease.cancel()
            with suppress(asyncio.CancelledError):
                await lease
            shutil.rmtree(workspace)
