from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import BuildState

SAMPLES_FILENAME = 'SAMPLES.md'
PLAN_FILENAME = 'PLAN.md'
BUILD_REQUEST = 'Please complete the task of constructing a workflow based on the above information.'


class WorkflowGenerationState(BuildState, total=False):
    """Shared state for the durable Plan -> Build orchestration."""

    samples_ready: bool
    samples_changed: bool
    plan_ready: bool


def _phase(phase: str, status: str) -> None:
    get_stream_writer()({'workflow_phase': {'phase': phase, 'status': status}})


async def inspect_plan(
    state: WorkflowGenerationState,  # noqa: ARG001
    runtime: Runtime[ProjectContext],
) -> dict[str, bool]:
    """Decide whether the prepared samples and accepted plan can be reused."""
    workdir = Path(runtime.context.workdir)
    samples_ready = (workdir / SAMPLES_FILENAME).is_file()
    plan_ready = samples_ready and (workdir / PLAN_FILENAME).is_file()
    _phase('project_planner', 'skipped' if plan_ready else 'running')
    return {'samples_ready': samples_ready, 'plan_ready': plan_ready}


async def route_plan(state: WorkflowGenerationState) -> Literal['load_plan', 'project_planner']:
    """Route to reuse or run the project-planner agent."""
    return 'load_plan' if state.get('plan_ready', False) else 'project_planner'


async def load_plan(
    state: WorkflowGenerationState,
    runtime: Runtime[ProjectContext],
) -> dict[str, object]:
    """Load the authoritative sample summary and plan, then hand them to the builder."""
    workdir = Path(runtime.context.workdir)
    samples = (workdir / SAMPLES_FILENAME).read_text(encoding='utf-8')
    plan = (workdir / PLAN_FILENAME).read_text(encoding='utf-8')
    if not state.get('plan_ready', False):
        _phase('project_planner', 'completed')
    _phase('workflow_builder', 'running')
    return {
        'messages': [HumanMessage(content=samples), HumanMessage(content=plan), HumanMessage(content=BUILD_REQUEST)],
        'samples_changed': not state.get('samples_ready', False),
    }


async def finish_workflow(
    state: WorkflowGenerationState,  # noqa: ARG001
    runtime: Runtime[ProjectContext],  # noqa: ARG001
) -> None:
    """Publish completion of the final orchestration phase."""
    _phase('workflow_builder', 'completed')


def build_workflow_generation(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Build the durable interactive Plan -> Build parent graph.

    Existing ``SAMPLES.md`` and ``PLAN.md`` files are authoritative reusable outputs: when both exist the
    project-planner agent is skipped; otherwise it runs and itself skips sample registration if
    ``SAMPLES.md`` already exists. All nested agents inherit the parent checkpoint, so any human
    interrupt pauses and resumes the same web run at the exact stage.

    Args:
        checkpointer: Checkpointer used by the top-level web AgentRun.

    Returns:
        Compiled workflow-generation graph.
    """
    from bioflow_ai.llm.agents.fs_agent.project_planner import build_project_planner
    from bioflow_ai.llm.agents.workflow_agent.workflow_builder import build_workflow_builder

    return _compile_workflow_generation(build_project_planner(), build_workflow_builder(), checkpointer)


def _compile_workflow_generation(
    project_planner: CompiledStateGraph,
    workflow_builder: CompiledStateGraph,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Compile the parent graph around already-built child graphs."""
    # ty: ignore[invalid-argument-type]
    graph = StateGraph(WorkflowGenerationState, context_schema=ProjectContext)
    graph.add_node('inspect_plan', inspect_plan)
    graph.add_node('project_planner', project_planner)
    graph.add_node('load_plan', load_plan)
    graph.add_node('workflow_builder', workflow_builder)
    graph.add_node('finish_workflow', finish_workflow)

    graph.add_edge(START, 'inspect_plan')
    graph.add_conditional_edges('inspect_plan', route_plan)
    graph.add_edge('project_planner', 'load_plan')
    graph.add_edge('load_plan', 'workflow_builder')
    graph.add_edge('workflow_builder', 'finish_workflow')
    graph.add_edge('finish_workflow', END)

    # ty: ignore[invalid-return-type]
    return graph.compile(checkpointer=checkpointer)
