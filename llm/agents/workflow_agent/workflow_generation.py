from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from llm.context import ProjectContext
from llm.states import BuildState

SAMPLES_FILENAME = 'SAMPLES.md'
PLAN_FILENAME = 'PLAN.md'
BUILD_REQUEST = 'Please complete the task of constructing a workflow based on the above information.'


class WorkflowGenerationState(BuildState, total=False):
    """Shared state for the durable Sample -> Plan -> Build orchestration."""

    summary: str
    plan: str
    samples_ready: bool
    samples_changed: bool
    plan_ready: bool


def _phase(phase: str, status: str) -> None:
    get_stream_writer()({'workflow_phase': {'phase': phase, 'status': status}})


async def inspect_samples(
    state: WorkflowGenerationState,  # noqa: ARG001
    runtime: Runtime[ProjectContext],
) -> dict[str, bool]:
    """Decide whether sample initialization can be reused."""
    ready = (Path(runtime.context.workdir) / SAMPLES_FILENAME).is_file()
    _phase('sample_manager', 'skipped' if ready else 'running')
    return {'samples_ready': ready}


async def route_samples(state: WorkflowGenerationState) -> Literal['load_samples', 'sample_manager']:
    """Route to sample reuse or generation."""
    return 'load_samples' if state.get('samples_ready', False) else 'sample_manager'


async def load_samples(
    state: WorkflowGenerationState,
    runtime: Runtime[ProjectContext],
) -> dict[str, object]:
    """Load the authoritative sample summary into downstream agent context."""
    content = (Path(runtime.context.workdir) / SAMPLES_FILENAME).read_text(encoding='utf-8')
    if not state.get('samples_ready', False):
        _phase('sample_manager', 'completed')
    return {
        'messages': [HumanMessage(content=content)],
        'samples_changed': not state.get('samples_ready', False),
    }


async def inspect_plan(
    state: WorkflowGenerationState,  # noqa: ARG001
    runtime: Runtime[ProjectContext],
) -> dict[str, bool]:
    """Decide whether an accepted project plan can be reused."""
    workdir = Path(runtime.context.workdir)
    plan_file = workdir / PLAN_FILENAME
    ready = plan_file.is_file()
    _phase('workflow_planner', 'skipped' if ready else 'running')
    return {'plan_ready': ready}


async def route_plan(state: WorkflowGenerationState) -> Literal['load_plan', 'workflow_planner']:
    """Route to plan reuse or interactive planning."""
    return 'load_plan' if state.get('plan_ready', False) else 'workflow_planner'


async def load_plan(
    state: WorkflowGenerationState,
    runtime: Runtime[ProjectContext],
) -> dict[str, list[HumanMessage]]:
    """Load the authoritative plan and hand the prepared context to the builder."""
    workdir = Path(runtime.context.workdir)
    content = (workdir / PLAN_FILENAME).read_text(encoding='utf-8')
    if not state.get('plan_ready', False):
        _phase('workflow_planner', 'completed')
    _phase('workflow_builder', 'running')
    return {'messages': [HumanMessage(content=content), HumanMessage(content=BUILD_REQUEST)]}


async def finish_workflow(
    state: WorkflowGenerationState,  # noqa: ARG001
    runtime: Runtime[ProjectContext],  # noqa: ARG001
) -> None:
    """Publish completion of the final orchestration phase."""
    _phase('workflow_builder', 'completed')


def build_workflow_generation(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Build the durable interactive Sample -> Plan -> Build parent graph.

    Existing ``SAMPLES.md`` and ``PLAN.md`` files are authoritative reusable stage outputs.
    Missing files cause the corresponding agent to run. All nested agents inherit the parent
    checkpoint, so any human interrupt pauses and resumes the same web run at the exact stage.

    Args:
        checkpointer: Checkpointer used by the top-level web AgentRun.

    Returns:
        Compiled workflow-generation graph.
    """
    from llm.agents.fs_agent.sample_manager import build_create_sample_agent
    from llm.agents.workflow_agent.workflow_builder import build_workflow_builder
    from llm.agents.workflow_agent.workflow_designer_v2 import build_workflow_planner

    return _compile_workflow_generation(
        build_create_sample_agent(),
        build_workflow_planner(),
        build_workflow_builder(),
        checkpointer,
    )


def _compile_workflow_generation(
    sample_manager: CompiledStateGraph,
    workflow_planner: CompiledStateGraph,
    workflow_builder: CompiledStateGraph,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Compile the parent graph around already-built child graphs."""
    # pyrefly: ignore [bad-specialization]
    graph = StateGraph(WorkflowGenerationState, context_schema=ProjectContext)
    graph.add_node('inspect_samples', inspect_samples)
    graph.add_node('sample_manager', sample_manager)
    graph.add_node('load_samples', load_samples)
    graph.add_node('inspect_plan', inspect_plan)
    graph.add_node('workflow_planner', workflow_planner)
    graph.add_node('load_plan', load_plan)
    graph.add_node('workflow_builder', workflow_builder)
    graph.add_node('finish_workflow', finish_workflow)

    graph.add_edge(START, 'inspect_samples')
    graph.add_conditional_edges('inspect_samples', route_samples)
    graph.add_edge('sample_manager', 'load_samples')
    graph.add_edge('load_samples', 'inspect_plan')
    graph.add_conditional_edges('inspect_plan', route_plan)
    graph.add_edge('workflow_planner', 'load_plan')
    graph.add_edge('load_plan', 'workflow_builder')
    graph.add_edge('workflow_builder', 'finish_workflow')
    graph.add_edge('finish_workflow', END)

    # pyrefly: ignore [bad-return]
    return graph.compile(checkpointer=checkpointer)
