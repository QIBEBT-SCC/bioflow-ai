from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from llm.agents.fs_agent.sample_manager import build_create_sample_agent
from llm.agents.tool_agent import CreateToolAgent
from llm.agents.workflow_agent.workflow_generation import build_workflow_generation

RouteTask = Literal['workflow-builder', 'sample-manager', 'tool-generator']

AgentFactory = Callable[[Optional[BaseCheckpointSaver]], CompiledStateGraph]


@dataclass(frozen=True)
class AgentSpec:
    """Registry entry for a top-level routable agent.

    Attributes:
        description: Human-readable purpose of the agent.
        factory: Builds the compiled agent with the given checkpointer.
    """

    description: str
    factory: AgentFactory


AGENTS: dict[RouteTask, AgentSpec] = {
    'workflow-builder': AgentSpec(
        description='Prepare project samples and an approved plan, then build the node workflow on the canvas.',
        factory=build_workflow_generation,
    ),
    'sample-manager': AgentSpec(
        description="Explore raw data paths and register samples/files into the project's index.",
        factory=build_create_sample_agent,
    ),
    'tool-generator': AgentSpec(
        description='Create and save a reusable bioinformatics tool node.',
        factory=lambda checkpointer: CreateToolAgent().build(memory=checkpointer),
    ),
}


def build_agents(checkpointer: Optional[BaseCheckpointSaver] = None) -> dict[RouteTask, CompiledStateGraph]:
    """Compile every wired agent once, sharing a single checkpointer.

    Intended to run at application startup so compiled graphs are cached and reused across
    requests. Per-request variation is carried by ``thread_id`` and ``context`` at invoke
    time, never by rebuilding the graph.

    Args:
        checkpointer: The shared, long-lived checkpointer bound to every top-level agent.

    Returns:
        Mapping of route to compiled agent.
    """
    return {route: spec.factory(checkpointer) for route, spec in AGENTS.items()}
