import rustworkx as rx

from pathlib import Path

from celery import chain

from app.tasks.run_tool import docker_task
from app.tasks.load_file import load_file_task
from app.models.react_flow import WorkflowDefinition, TaskNode, TaskInput


def build_task_list(workflow: WorkflowDefinition) -> list[TaskNode]:
    # rustworkx DAG对象
    graph = rx.PyDiGraph()
    # 节点uuid到rustworkx node_index映射
    node_index_map: dict[str, int] = {}

    # 构建TaskNode
    node_map: dict[str, TaskNode] = {}
    for node in workflow.nodes:
        node_map[node.id] = TaskNode(
            id=node.id, type=node.type, tool_uid=node.data.tool_uid, args=node.data.args
        )
    for edge in workflow.edges:
        node_map[edge.target].kwarg[edge.targetHandle.split('-in-')[-1]] = TaskInput(
            source_uid=edge.source, source_k=edge.sourceHandle.split('-out-')[-1]
        )

    # DAG排序
    edge_set = [(edge.source, edge.target) for edge in workflow.edges]
    for source_uid, target_uid in edge_set:
        if source_uid not in node_index_map:
            node_index_map[source_uid] = graph.add_node(source_uid)

        if target_uid not in node_index_map:
            node_index_map[target_uid] = graph.add_node(target_uid)

        graph.add_edge(node_index_map[source_uid], node_index_map[target_uid], None)

    topo_sort = rx.topological_sort(graph)
    return [node_map[graph.nodes()[index]] for index in topo_sort]


def send_task(task_nodes: list[TaskNode]):
    tasks = []
    task_index = 1
    for task_node in task_nodes:
        if task_node.type == 'tool':
            tasks.append(
                docker_task.s(
                    tool_uid=task_node.tool_uid,
                    inputs=task_node.kwarg,
                    output_dir=f'testoutput/HC23/{task_index}_{task_node.tool_uid}',
                    params=task_node.args,
                    sample_name='HC23',
                ).set(task_id=task_node.id)
            )
            task_index += 1
        elif task_node.type == 'fileInput':
            tasks.append(load_file_task.s(file_path=task_node.args).set(task_id=task_node.id))

    workflow = chain(*tasks)
    workflow.delay()


def run_from_file(json_file: str | Path):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_str = f.read()
        workflow = WorkflowDefinition.model_validate_json(json_str)

    return build_task_list(workflow)
