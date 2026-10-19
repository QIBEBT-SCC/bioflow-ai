from collections.abc import Callable
from typing import Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from bioflow_ai.llm.agents.fs_agent.project_planner import build_project_planner
from bioflow_ai.llm.agents.tool_agent import CreateToolAgent
from bioflow_ai.llm.agents.workflow_agent.workflow_assistant import build_workflow_assistant
from bioflow_ai.llm.agents.workflow_agent.workflow_fixer import build_workflow_fixer
from bioflow_ai.llm.agents.workflow_agent.workflow_generation import build_workflow_generation

RouteTask = Literal['workflow-builder', 'workflow-assistant', 'workflow-fixer', 'project-planner', 'tool-generator']

AgentFactory = Callable[[Optional[BaseCheckpointSaver]], CompiledStateGraph]


class AgentSpec(BaseModel):
    """Registry entry for a top-level routable agent.

    Attributes:
        description: Human-readable purpose of the agent.
        factory: Builds the compiled agent with the given checkpointer.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    factory: AgentFactory


AGENTS: dict[RouteTask, AgentSpec] = {
    'workflow-builder': AgentSpec(
        description='Prepare project samples and an approved plan, then build the node workflow on the canvas.',
        factory=build_workflow_generation,
    ),
    'workflow-assistant': AgentSpec(
        description='Answer questions about one workflow run and diagnose failures when requested.',
        factory=build_workflow_assistant,
    ),
    'workflow-fixer': AgentSpec(
        description='Repair the current workflow template from the user request and any available diagnosis.',
        factory=build_workflow_fixer,
    ),
    'project-planner': AgentSpec(
        description='Register project samples from raw data and produce an approved analysis plan.',
        factory=build_project_planner,
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
