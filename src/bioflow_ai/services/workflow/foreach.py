"""Durable, bounded child-run coordination for dynamic file collections."""

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.resource import Sample
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import Edge, Node, WorkflowDefinition
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.workflow.collection import read_collection
from bioflow_ai.services.workflow.compiler import CompileContext, compile_run
from bioflow_ai.services.workflow.subgraphs import definition_id
from bioflow_ai.utils.system import random_uid, utc_now


def _source_value(session: Session, ref: dict[str, str]) -> tuple[str, str | None]:
    source = session.get(NodeRun, uuid.UUID(ref['source_uid']))
    if source is None or source.status != NodeRunStatus.SUCCEEDED or source.tool_output is None:
        message = 'ForEach input source has not succeeded'
        raise ValueError(message)
    if ref['source_k'] == 'workdir' and source.base_dir:
        return source.base_dir, source.tool_output.sample_name
    value = source.tool_output.all_files().get(ref['source_k'])
    if value is None:
        message = f'ForEach source lacks output {ref["source_k"]!r}'
        raise ValueError(message)
    return str(value), source.tool_output.sample_name


def _child_sample_name(base_name: str | None, item_name: str, index: int) -> str:
    """Give one file a stable name based on its upstream sample and filename."""
    stem = re.sub(r'[^A-Za-z0-9._-]', '_', Path(item_name).stem).strip('._-') or f'item_{index + 1}'
    if not base_name:
        return stem
    if stem == base_name or stem.startswith(tuple(f'{base_name}{separator}' for separator in '._-')):
        return stem
    return f'{base_name}.{stem}'


def _child_graph(graph: WorkflowDefinition, paths: dict[str, str], sample_name: str) -> WorkflowDefinition:
    """Bind each exposed child input to a persisted, ordinary DAG source."""
    child = graph.model_copy(deep=True)
    if child.interface is None:
        message = 'ForEach child has no interface'
        raise ValueError(message)
    for index, port in enumerate(child.interface.inputs):
        source_id = f'@foreach/input/{index}'
        child.nodes.append(
            Node(
                id=source_id,
                type='resource_frozen_path',
                data={'path': paths[port.id], 'sample_name': sample_name},
                dragHandle='',
                zIndex=0,
            )
        )
        for target_index, target in enumerate(port.targets):
            child.edges.append(
                Edge(
                    id=f'@foreach/edge/{index}/{target_index}',
                    source=source_id,
                    sourceHandle=f'{source_id}-out-file_path',
                    target=target.node_id,
                    targetHandle=f'{target.node_id}-in-{target.handle}',
                )
            )
    return child


def _initialize(session: Session, node: NodeRun, parent: WorkflowRun) -> None:
    plan = node.prepared_data.copy()
    kwargs = plan['kwargs']
    refs = kwargs['inputs']
    sources = {key: _source_value(session, ref) for key, ref in refs.items()}
    collection_path, base_sample_name = sources.pop('collection')
    items = read_collection(collection_path)
    paths = {key: path for key, (path, _) in sources.items()}
    if not base_sample_name:
        base_sample_name = next((name for _, name in sources.values() if name), None)
    if not base_sample_name and parent.sample_uid is not None:
        sample = session.get(Sample, parent.sample_uid)
        base_sample_name = sample.sample_name if sample else None
    graph = WorkflowDefinition.model_validate(kwargs['workflow'])
    if graph.interface is None:
        message = 'ForEach child has no interface'
        raise ValueError(message)
    if len({port.id for port in graph.interface.inputs}) != len(graph.interface.inputs):
        message = 'Duplicate ForEach input port'
        raise ValueError(message)
    node.prepared_data = {**plan, 'items': items, 'shared_paths': paths, 'base_sample_name': base_sample_name}
    node.status = NodeRunStatus.RUNNING
    node.start_time = utc_now()
    parent.status = RunStatus.RUNNING
    parent.settled = False
    if parent.start_time is None:
        parent.start_time = utc_now()
    for index, item in enumerate(items):
        subfolder = f'{kwargs["subfolder_prefix"]}_{index + 1}'
        child = WorkflowRun(
            uid=random_uid(),
            generation=1,
            name=f'{parent.name} / {item["name"]}',
            owner_id=parent.owner_id,
            project_id=parent.project_id,
            sample_uid=parent.sample_uid,
            workflow_uid=None,
            parent_node_uid=node.uid,
            parent_generation=parent.generation,
            item_key=item['key'],
            item_index=index,
            item_name=item['name'],
            base_dir=str(Path(kwargs['output_dir']) / 'children' / subfolder),
            workflow=graph.model_copy(deep=True),
            status=RunStatus.PENDING,
            settled=False,
        )
        session.add(child)
    session.flush()


def _output_source(graph: WorkflowDefinition, node_id: str, handle: str, path: tuple[str, ...] = ()) -> tuple[str, str]:
    node = next((candidate for candidate in graph.nodes if candidate.id == node_id), None)
    if node is None:
        message = f'Invalid ForEach output node {node_id!r}'
        raise ValueError(message)
    if node.type != 'subgraph':
        return definition_id((*path, node_id)), handle
    child = WorkflowDefinition.model_validate(node.data['workflow'])
    if child.interface is None:
        message = 'Subgraph output has no interface'
        raise ValueError(message)
    port = next((candidate for candidate in child.interface.outputs if candidate.id == handle), None)
    if port is None:
        message = f'Invalid subgraph output port {handle!r}'
        raise ValueError(message)
    return _output_source(child, port.source.node_id, port.source.handle, (*path, node_id))


def _publish_results(session: Session, node: NodeRun, children: list[WorkflowRun]) -> ToolOutput:
    graph = WorkflowDefinition.model_validate(node.prepared_data['kwargs']['workflow'])
    if graph.interface is None:
        message = 'ForEach child has no interface'
        raise ValueError(message)
    results = Path(node.prepared_data['kwargs']['output_dir']) / 'results'
    results.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for child in children:
        subfolder = f'{node.prepared_data["kwargs"]["subfolder_prefix"]}_{child.item_index + 1}'
        item = node.prepared_data['items'][child.item_index]
        outputs: dict[str, str] = {}
        node_runs = {
            item.definition_node_id: item
            for item in session.scalars(select(NodeRun).where(NodeRun.run_uid == child.uid))
        }
        for port in graph.interface.outputs if child.status == RunStatus.SUCCEEDED else []:
            source_id, handle = _output_source(graph, port.source.node_id, port.source.handle)
            source = node_runs.get(source_id)
            if source is None or source.tool_output is None:
                message = f'ForEach item {child.item_key} lacks output {port.name}'
                raise ValueError(message)
            value = source.tool_output.all_files().get(handle)
            if value is None and handle == 'workdir':
                value = source.base_dir
            if value is None:
                message = f'ForEach item {child.item_key} lacks output {port.name}'
                raise ValueError(message)
            origin = Path(value).resolve(strict=True)
            target_root = results / subfolder / hashlib.sha256(port.id.encode()).hexdigest()[:24]
            target_root.parent.mkdir(parents=True, exist_ok=True)
            if origin.is_dir():
                if target_root.is_relative_to(origin):
                    message = 'ForEach output directory contains its own result destination'
                    raise ValueError(message)
                shutil.copytree(origin, target_root, dirs_exist_ok=True)
                target = target_root
            elif origin.is_file():
                target_root.mkdir(parents=True, exist_ok=True)
                target = target_root / origin.name
                shutil.copy2(origin, target)
            else:
                message = f'ForEach output is not a file or directory: {origin}'
                raise ValueError(message)
            outputs[port.id] = target.relative_to(results).as_posix()
        manifest_items.append(
            {
                'key': child.item_key,
                'name': child.item_name,
                'relative_path': item['relative_path'],
                'subfolder': subfolder,
                'status': child.status.value,
                'run_uid': str(child.uid),
                'outputs': outputs,
            }
        )
    manifest = results / 'manifest.json'
    temporary = results / 'manifest.json.tmp'
    temporary.write_text(json.dumps({'version': 1, 'items': manifest_items}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(manifest)
    return ToolOutput(result={'results_folder': results})


def _process_one(node_uid: uuid.UUID) -> tuple[ToolOutput | None, str | None]:
    """One locked step. Filesystem copies are repeatable after a crash."""
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_uid, with_for_update=True)
        if (
            node is None
            or node.node_type != 'foreach'
            or node.status not in {NodeRunStatus.READY, NodeRunStatus.RUNNING}
        ):
            return None, None
        parent = session.get(WorkflowRun, node.run_uid, with_for_update=True)
        if parent is None or parent.generation != node.generation:
            return None, None
        if node.status == NodeRunStatus.READY:
            _initialize(session, node, parent)
        children = list(
            session.scalars(
                select(WorkflowRun)
                .where(WorkflowRun.parent_node_uid == node.uid, WorkflowRun.parent_generation == node.generation)
                .order_by(WorkflowRun.item_index)
                .with_for_update()
            )
        )
        active = sum(child.start_time is not None and not child.settled for child in children)
        capacity = get_settings().workflow.MAX_PARALLEL_ITEMS - active
        for child in children:
            if capacity <= 0:
                break
            if child.start_time is not None:
                continue
            item = node.prepared_data['items'][child.item_index]
            base_sample_name = node.prepared_data.get('base_sample_name')
            if not base_sample_name and child.sample_uid is not None:
                sample = session.get(Sample, child.sample_uid)
                base_sample_name = sample.sample_name if sample else None
            sample_name = _child_sample_name(base_sample_name, item['name'], child.item_index)
            paths = {**node.prepared_data['shared_paths'], node.prepared_data['kwargs']['item_port_id']: item['path']}
            graph = _child_graph(child.workflow, paths, sample_name)
            child.workflow = graph
            child.start_time = utc_now()
            compile_run(
                session,
                child,
                graph,
                CompileContext(
                    project_id=child.project_id,
                    sample_uid=child.sample_uid,
                    sample_name=sample_name,
                    output_base=child.base_dir,
                ),
                allow_internal_frozen_paths=True,
            )
            capacity -= 1
        if any(not child.settled for child in children):
            return None, None
        output = _publish_results(session, node, children)
        if any(child.status == RunStatus.FAILED for child in children):
            node.tool_output = output
            return None, 'One or more ForEach items failed'
        return output, None


def advance_foreach() -> int:
    """Advance ready and running control nodes without occupying Celery workers."""
    from bioflow_ai.services.workflow.runtime import mark_node_failed, mark_node_succeeded

    with Session(engine) as session:
        candidates = list(
            session.scalars(
                select(NodeRun.uid)
                .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
                .where(
                    NodeRun.node_type == 'foreach',
                    NodeRun.status.in_((NodeRunStatus.READY, NodeRunStatus.RUNNING)),
                    NodeRun.generation == WorkflowRun.generation,
                )
                .order_by(NodeRun.create_time)
            )
        )
    for uid in candidates:
        try:
            output, error = _process_one(uid)
        except OperationalError:
            logger.exception(f'Transient ForEach database failure for {uid}; recovery will retry')
            continue
        except Exception as exc:
            logger.exception(f'ForEach coordination failed for {uid}')
            mark_node_failed(uid, _generation(uid), exc)
        else:
            if error:
                mark_node_failed(uid, _generation(uid), error)
            elif output is not None:
                mark_node_succeeded(uid, _generation(uid), output)
    return len(candidates)


def _generation(uid: uuid.UUID) -> int:
    with Session(engine) as session:
        node = session.get(NodeRun, uid)
        return node.generation if node is not None else -1
