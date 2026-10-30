from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any, Literal, Optional, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from bioflow_ai.llm.agents.fs_agent.project_planner import build_project_planner
from bioflow_ai.llm.agents.tool_agent import CreateToolAgent
from bioflow_ai.llm.agents.workflow_agent.workflow_assistant import build_workflow_assistant
from bioflow_ai.llm.agents.workflow_agent.workflow_fixer import build_workflow_fixer
from bioflow_ai.llm.agents.workflow_agent.workflow_generation import build_workflow_generation
from bioflow_ai.utils.observability.execution_logging import AgentRunLogger

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


class AgentGraph(Protocol):
    def astream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream graph events."""
        ...

    async def aget_state(self, config: Any) -> Any:
        """Read the final graph state."""
        ...


ProgressCallback = Callable[[object, tuple[str, ...]], Awaitable[None]]
ChunkCallback = Callable[[Any], None]


class AgentExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: dict[str, Any]
    interrupt: dict[str, Any] | None


async def execute_agent_graph(
    graph: AgentGraph,
    agent_input: object,
    config: object,
    context: object,
    on_progress: ProgressCallback,
    *,
    execution_log: AgentRunLogger,
) -> AgentExecutionResult:
    """Run one compiled graph and normalize progress and interrupt output."""
    state = await consume_agent_graph(
        graph,
        agent_input,
        config,
        context,
        execution_log=execution_log,
        on_progress=on_progress,
        record_lifecycle=False,
    )
    interrupt = state.interrupts[0].value if state.interrupts else None
    return AgentExecutionResult(values=state.values, interrupt=interrupt)


async def consume_agent_graph(
    graph: AgentGraph,
    agent_input: object,
    config: object,
    context: object,
    *,
    execution_log: AgentRunLogger,
    on_chunk: ChunkCallback | None = None,
    on_progress: ProgressCallback | None = None,
    record_lifecycle: bool = True,
    resumed: bool = False,
) -> Any:
    """Consume a graph once, preserving raw callbacks and reading its final state.

    Durable workers own lifecycle records after committing their transitions. CLI
    invocations can record their graph lifecycle here instead. Progress callbacks receive
    each custom value with the namespace of the graph that streamed it.
    """
    observation = execution_log.observe(resumed=resumed) if record_lifecycle else nullcontext()
    with observation:
        async for chunk in graph.astream(
            agent_input,
            config,
            context=context,
            stream_mode=['updates', 'custom'],
            version='v2',
            subgraphs=True,
        ):
            execution_log.stream_chunk(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
            if chunk['type'] == 'custom' and on_progress is not None:
                await on_progress(chunk['data'], tuple(chunk['ns']))

        state = await graph.aget_state(config)
        if record_lifecycle:
            interrupts = state.interrupts or tuple(
                interrupt for task in getattr(state, 'tasks', ()) for interrupt in getattr(task, 'interrupts', ())
            )
            if interrupts:
                execution_log.status('waiting_input', [interrupt.value for interrupt in interrupts])
        return state
