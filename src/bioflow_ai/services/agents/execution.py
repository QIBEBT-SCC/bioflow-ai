from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from bioflow_ai.llm.observability.execution_logging import AgentRunLogger


class AgentGraph(Protocol):
    def astream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream graph events."""
        ...

    async def aget_state(self, config: Any) -> Any:
        """Read the final graph state."""
        ...


ProgressCallback = Callable[[object], Awaitable[None]]
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
    invocations can record their graph lifecycle here instead.
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
                await on_progress(chunk['data'])

        state = await graph.aget_state(config)
        if record_lifecycle:
            interrupts = state.interrupts or tuple(
                interrupt for task in getattr(state, 'tasks', ()) for interrupt in getattr(task, 'interrupts', ())
            )
            if interrupts:
                execution_log.status('waiting_input', [interrupt.value for interrupt in interrupts])
        return state
