"""Task Scheduler.

Responsible for workflow node sorting, DAG construction, and task distribution.
"""

import uuid

import rich
import rustworkx as rx
from celery import chain
from loguru import logger
from sqlalchemy.orm import Session
from uuid_extensions import uuid7str

from app.schemas.react_flow import TaskInput, TaskNode, WorkflowDefinition
from app.schemas.user import CurrentUser
from app.utils.workflow import TaskContext, get_handler
from app.utils.workflow.workflow_definition import update_node_id

__all__ = ['build_task_list', 'send_task', 'update_node_id']


def build_task_list(workflow: WorkflowDefinition) -> list[list[TaskNode]]:
    """Reorganize and sort the task queue.

    Analyze raw react-flow data, convert to DAG.
    Remove isolated nodes (in-degree and out-degree are 0), and sort by depth layers.

    Args:
        workflow: Raw react-flow data.

    Returns:
        Sorted task node queue.
    """
    # rustworkx DAG object
    graph = rx.PyDiGraph()
    # Mapping from node uuid to rustworkx node_index
    node_index_map: dict[str, int] = {}

    # Build TaskNode
    node_map: dict[str, TaskNode] = {}
    for node in workflow.nodes:
        node_map[node.id] = TaskNode(id=node.id, type=node.type, data=node.data)
    for edge in workflow.edges:
        target_param = edge.targetHandle.split('-in-')[-1]
        task_input = TaskInput(source_uid=edge.source, source_k=edge.sourceHandle.split('-out-')[-1])

        # Check if this parameter already has an input
        if target_param in node_map[edge.target].kwarg:
            existing = node_map[edge.target].kwarg[target_param]
            # If already a list, append
            if isinstance(existing, list):
                existing.append(task_input)
            # If single TaskInput, convert to list
            else:
                node_map[edge.target].kwarg[target_param] = [existing, task_input]
        else:
            # First time adding, keep as single TaskInput
            node_map[edge.target].kwarg[target_param] = task_input

    # DAG sorting
    edge_set = [(edge.source, edge.target) for edge in workflow.edges]
    for source_uid, target_uid in edge_set:
        if source_uid not in node_index_map:
            node_index_map[source_uid] = graph.add_node(source_uid)

        if target_uid not in node_index_map:
            node_index_map[target_uid] = graph.add_node(target_uid)

        graph.add_edge(node_index_map[source_uid], node_index_map[target_uid], None)

    first_layers = [node for node in graph.node_indices() if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
    task_groups = rx.layers(graph, first_layers)
    # pyrefly: ignore [bad-index]
    return [[node_map[index] for index in sublist] for sublist in task_groups]


def send_task(
    task_nodes: list[list[TaskNode]],
    session: Session,
    project_id: int | None,
    sample_uid: uuid.UUID | None,
    sample_name: str | None,
    owner: CurrentUser,
    output_base: str,
    run_uid: uuid.UUID,
    *,
    auto_summary: bool = False,
    summary_prompt: str | None = None,
) -> None:
    """Build the actual task queue and send it to celery worker.

    Args:
        task_nodes: Sorted task node queue.
        session: SQLAlchemy session.
        project_id: Project ID.
        sample_uid: Sample UID for this analysis.
        sample_name: Sample name for this analysis.
        owner: Task creator.
        output_base: Root output directory.
        run_uid: Run instance UID.
        auto_summary: Whether to append an upstream summary task.
        summary_prompt: Custom summary instruction; None uses default logic.
    """
    if auto_summary:
        task_nodes.append([TaskNode(id=uuid7str(), type='upstream_summary', data={'summary_prompt': summary_prompt})])

    rich.print(task_nodes)

    context = TaskContext(
        session=session,
        project_id=project_id,
        sample_uid=sample_uid,
        sample_name=sample_name,
        owner=owner,
        output_base=output_base,
        run_uid=run_uid,
        task_index=1,
    )

    tasks = []
    for sub_group in task_nodes:
        sub_tasks = []
        for task_node in sub_group:
            handler = get_handler(task_node.type)

            if handler is None:
                logger.warning(f'Handler not found for node type {task_node.type}')
                continue

            celery_task, new_task_index = handler.handle(task_node, context)
            context.task_index = new_task_index

            if celery_task is not None:
                sub_tasks.append(celery_task)

        if sub_tasks:
            # tasks.append(group(*sub_tasks))
            tasks.extend(sub_tasks)  # celery currently does not support multiple groups

    logger.debug(f'{len(tasks)} tasks have been submitted to the queue.')
    if tasks:
        workflow = chain(*tasks)
        workflow.delay()
