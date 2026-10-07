"""Run a native CODE editing turn and return a proposal without persisting it."""

import json
import shutil
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from bioflow_ai.llm.agents.coding_agent.simple_coding import build_simple_coding_agent
from bioflow_ai.llm.context import CodingContext
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.services.code.workspace import (
    create_workspace,
    proposal_diff,
    read_candidate,
    write_baseline,
)


@dataclass(frozen=True)
class CodeProposal:
    """Validated changes to the editor baseline for explicit review and acceptance."""

    source: str
    dependencies: list[str]
    diff: str
    warnings: list[str]


@dataclass(frozen=True)
class CodingTurnResult:
    """Conversation messages and an optional candidate; clarification may produce none."""

    messages: list[BaseMessage]
    proposal: CodeProposal | None


async def run_coding_turn(
    request: str,
    *,
    node_type: CodeNodeType,
    user_id: int,
    source: str = '',
    dependencies: list[str] | None = None,
    input_contract: list[dict[str, Any]] | None = None,
    language: str | None = None,
    agent: CompiledStateGraph[Any, CodingContext, Any, Any] | None = None,
) -> CodingTurnResult:
    """Initialize the latest baseline before invocation, validate changes, and always clean up.

    This entry point runs one turn without a checkpointer. Stateful callers can use
    build_simple_coding_agent(checkpointer=...) with their own workspace lifecycle.
    """
    if not request.strip():
        msg = 'A coding request is required.'
        raise ValueError(msg)
    baseline_dependencies = list(dependencies or [])
    workspace = create_workspace(node_type)
    try:
        write_baseline(workspace, node_type, source, baseline_dependencies)
        context = CodingContext(user_id=user_id, language=language, workspace_path=workspace, node_type=node_type)
        prompt = f'Current target CODE node type: {node_type.value}\n\nUser request:\n{request}'
        if input_contract is not None:
            prompt += '\n\nAuthoritative input contract:\n' + json.dumps(input_contract, ensure_ascii=False, indent=2)
        coding_agent = agent if agent is not None else build_simple_coding_agent()
        result = await coding_agent.ainvoke({'messages': [HumanMessage(prompt)]}, context=context)
        if result.get('__interrupt__'):
            msg = 'run_coding_turn requires an agent without interrupts or a checkpointer.'
            raise ValueError(msg)
        candidate_source, candidate_dependencies, warnings = read_candidate(
            workspace, node_type, allow_empty=not source.strip()
        )
        if not candidate_source.strip():
            return CodingTurnResult(messages=result['messages'], proposal=None)
        diff = proposal_diff(node_type, source, candidate_source, baseline_dependencies, candidate_dependencies)
        proposal = (
            CodeProposal(
                source=candidate_source,
                dependencies=candidate_dependencies,
                diff=diff,
                warnings=warnings,
            )
            if diff
            else None
        )
        return CodingTurnResult(messages=result['messages'], proposal=proposal)
    finally:
        shutil.rmtree(workspace)
