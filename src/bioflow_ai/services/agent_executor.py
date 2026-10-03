from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


class AgentGraph(Protocol):
    def astream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream graph events."""
        ...

    async def aget_state(self, config: Any) -> Any:
        """Read the final graph state."""
        ...


ProgressCallback = Callable[[object], Awaitable[None]]


@dataclass(frozen=True)
class AgentExecutionResult:
    values: dict[str, Any]
    interrupt: dict[str, Any] | None


async def execute_agent_graph(
    graph: AgentGraph,
    agent_input: object,
    config: object,
    context: object,
    on_progress: ProgressCallback,
) -> AgentExecutionResult:
    """Run one compiled graph and normalize progress and interrupt output."""
    async for chunk in graph.astream(
        agent_input,
        config,
        context=context,
        stream_mode=['updates', 'custom'],
        version='v2',
        subgraphs=True,
    ):
        if chunk['type'] == 'custom':
            await on_progress(chunk['data'])

    state = await graph.aget_state(config)
    interrupt = state.interrupts[0].value if state.interrupts else None
    return AgentExecutionResult(values=state.values, interrupt=interrupt)
