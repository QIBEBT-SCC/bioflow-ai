"""Merkle DAG fingerprinting, artifact validation, and pass-through taint pruning for incremental workflow reruns."""

import hashlib
import json
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.crud import project_file_mapping as project_file_mapping_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.models.resource import BioDb
from bioflow_ai.models.tools import ToolDefine, ToolImage
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import Edge, Node, WorkflowDefinition
from bioflow_ai.schemas.tool_config import ToolType
from bioflow_ai.schemas.tools import WorkflowToolDefinition
from bioflow_ai.schemas.workflow_run import NodeRunStatus
from bioflow_ai.services.workflow.batch_graph import lower_batch_subgraphs
from bioflow_ai.services.workflow.collection import read_collection
from bioflow_ai.services.workflow.execution_context import resolve_path
from bioflow_ai.services.workflow.output_validation import all_output_paths_exist, output_path_exists
from bioflow_ai.services.workflow.subgraphs import definition_id, flatten_workflow
from bioflow_ai.services.workflow.validation._topology import order_nodes
from bioflow_ai.services.workflow.validation.passthrough import PATH_INPUTS
from bioflow_ai.services.workflow.validation.registry import PRESENTATION_NODE_TYPES
from bioflow_ai.tasks.code import normalize_python_dependencies

FINGERPRINT_VERSION = 1
_DIR_INDEX_PATTERN = re.compile(r'^(\d+)_')


def parse_dir_index(dir_name: str) -> int | None:
    """Extract the leading integer prefix from a numbered node directory name."""
    match = _DIR_INDEX_PATTERN.match(dir_name)
    return int(match.group(1)) if match else None


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _to_utc_timestamp(value: datetime) -> float:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).timestamp()


def load_tool_images(session: Session, tool_uids: Iterable[str]) -> dict[str, dict[str, Any] | None]:
    """Batch-load image ref and config snapshots for tool definitions."""
    parsed_uids: list[uuid.UUID] = []
    for raw in tool_uids:
        try:
            parsed_uids.append(uuid.UUID(str(raw)))
        except ValueError:
            continue
    if not parsed_uids:
        return {}

    rows = session.execute(
        select(ToolDefine.uid, ToolImage.ref, ToolImage.image)
        .outerjoin(ToolImage, ToolDefine.image_uid == ToolImage.uid)
        .where(ToolDefine.uid.in_(parsed_uids))
    ).all()
    return {
        str(uid): {
            'ref': ref,
            'image': image.model_dump(mode='json') if image is not None else None,
        }
        for uid, ref, image in rows
    }


def _path_stat_signature(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        st = resolved.stat()
    except OSError:
        return {'exists': False, 'path': str(path)}

    if resolved.is_dir():
        return {'exists': True, 'kind': 'dir', 'path': str(resolved), 'mtime_ns': st.st_mtime_ns}
    return {
        'exists': True,
        'kind': 'file',
        'path': str(resolved),
        'size': st.st_size,
        'mtime_ns': st.st_mtime_ns,
    }


def _resolve_resource_signature(
    session: Session,
    ref: str,
    project_id: int | None,
    sample_uid: str | None,
) -> dict[str, Any]:
    try:
        resolved_path, _ = resolve_path(ref, session, project_id, sample_uid)
    except (FileNotFoundError, KeyError, ValueError):
        return {'ref': ref, 'resolved': False}

    signature: dict[str, Any] = {
        'ref': ref,
        'resolved': True,
        'file': _path_stat_signature(resolved_path),
    }
    if ref.startswith('proj:') and project_id is not None:
        mapping = project_file_mapping_crud.get_by_keyword(session, project_id, ref.removeprefix('proj:'))
        if mapping is not None and mapping.is_dynamic:
            signature['dynamic_producer_run_uid'] = str(mapping.producer_run_uid)
    elif ref.startswith('sample:') and sample_uid is not None:
        sample = sample_crud.get_with_files(session, sample_uid)
        if sample is not None:
            tag = ref.removeprefix('sample:')
            matching = next((item for item in sample.sample_files if item.tag == tag), None)
            if matching is not None and matching.is_dynamic:
                signature['dynamic_producer_run_uid'] = str(matching.producer_run_uid)
    return signature


def _tool_definition_snapshot(
    tool_uid: str,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
    tool_images: Mapping[str, dict[str, Any] | None],
) -> dict[str, Any]:
    tool = tool_definitions.get(tool_uid)
    if tool is None:
        return {'tool_uid': tool_uid, 'missing': True}
    return {
        'tool_uid': str(tool.uid),
        'tool_type': int(tool.tool_type),
        'command_template': tool.command_template,
        'dynamic_params': [param.model_dump(mode='json') for param in tool.dynamic_params],
        'immutable_static_params': tool.immutable_static_params,
        'modifiable_static_params': tool.modifiable_static_params,
        'file_mounts': [mount.model_dump(mode='json') for mount in tool.file_mounts],
        'image': tool_images.get(tool_uid),
    }


def _output_source(
    graph: WorkflowDefinition,
    node_id: str,
    handle: str,
    path: tuple[str, ...] = (),
) -> tuple[str, str]:
    node = next(candidate for candidate in graph.nodes if candidate.id == node_id)
    if node.type != 'subgraph':
        return definition_id((*path, node_id)), handle
    child = WorkflowDefinition.model_validate(node.data['workflow'])
    if child.interface is None:
        message = f'Subgraph {node_id} has no interface'
        raise ValueError(message)
    port = next(candidate for candidate in child.interface.outputs if candidate.id == handle)
    return _output_source(child, port.source.node_id, port.source.handle, (*path, node_id))


def _compute_foreach_subgraph_fingerprint(
    session: Session,
    child: WorkflowDefinition,
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
    tool_images: Mapping[str, dict[str, Any] | None],
    project_id: int | None,
    sample_uid: str | None,
    sample_name: str | None,
) -> dict[str, Any]:
    if child.interface is None:
        return {'has_interface': False}

    synthetic = child.model_copy(deep=True)
    if synthetic.interface is None:
        return {'has_interface': False}
    for index, port in enumerate(synthetic.interface.inputs):
        source_id = f'@foreach/input/{index}'
        synthetic.nodes.append(
            Node(
                id=source_id,
                type='__foreach_port__',
                data={'port_id': port.id},
                dragHandle='',
                zIndex=0,
            )
        )
        for target_index, target in enumerate(port.targets):
            synthetic.edges.append(
                Edge(
                    id=f'@foreach/edge/{index}/{target_index}',
                    source=source_id,
                    sourceHandle=f'{source_id}-out-file_path',
                    target=target.node_id,
                    targetHandle=f'{target.node_id}-in-{target.handle}',
                )
            )
    synthetic.interface.inputs = []

    flat = flatten_workflow(lower_batch_subgraphs(synthetic))
    node_map = {node.id: node for node in flat.nodes if node.type not in PRESENTATION_NODE_TYPES}
    edges = [edge for edge in flat.edges if edge.target in node_map and edge.source in node_map]
    order = order_nodes(list(node_map), [(edge.source, edge.target) for edge in edges])
    inner_fps = compute_graph_fingerprints(
        session,
        flat,
        order,
        tool_definitions=tool_definitions,
        tool_images=tool_images,
        project_id=project_id,
        sample_uid=sample_uid,
        sample_name=sample_name,
    )
    outputs: list[tuple[str, str, str]] = []
    for port in child.interface.outputs:
        source_id, handle = _output_source(child, port.source.node_id, port.source.handle)
        outputs.append((port.id, inner_fps.get(source_id, ''), handle))
    return {'has_interface': True, 'outputs': sorted(outputs)}


def _node_self_config(
    session: Session,
    node: Node,
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
    tool_images: Mapping[str, dict[str, Any] | None],
    project_id: int | None,
    sample_uid: str | None,
    sample_name: str | None,
) -> dict[str, Any]:
    data = node.data
    node_type = node.type

    if node_type == '__foreach_port__':
        return {'port_id': str(data.get('port_id', ''))}

    if node_type == 'tool':
        tool_uid = str(data.get('tool_uid', ''))
        return {
            'tool': _tool_definition_snapshot(tool_uid, tool_definitions, tool_images),
            'params': data.get('modifiable_params') or '',
            'sample_name': sample_name or '',
        }

    if node_type == 'resource_file':
        ref = str(data.get('file_path', ''))
        return {
            'file_path': ref,
            'resolved': _resolve_resource_signature(session, ref, project_id, sample_uid),
            'sample_name': sample_name,
        }

    if node_type == 'resource_sequence':
        r1 = str(data.get('r1', 'sample:r1'))
        r2 = str(data.get('r2', 'sample:r2'))
        return {
            'r1': r1,
            'r2': r2,
            'resolved_r1': _resolve_resource_signature(session, r1, project_id, sample_uid),
            'resolved_r2': _resolve_resource_signature(session, r2, project_id, sample_uid),
            'sample_name': sample_name,
        }

    if node_type == 'resource_frozen_path':
        path = str(data.get('path', ''))
        return {
            'path': path,
            'sample_name': data.get('sample_name'),
            'file': _path_stat_signature(Path(path)),
        }

    if node_type == 'resource_text_file':
        return {
            'file_name': str(data.get('file_name', '')),
            'content': str(data.get('content', '')),
        }

    if node_type == 'resource_db':
        db_id = int(data['db_id'])
        bio_db = session.get(BioDb, db_id)
        return {
            'db_id': db_id,
            'path': bio_db.path if bio_db is not None else None,
            'last_update': bio_db.last_update if bio_db is not None else None,
        }

    if node_type in {'resource_ncbi_genome', 'resource_GRCh38', 'resource_GRCm39'}:
        return {'required_index': sorted({str(item) for item in data.get('required_index', [])})}

    if node_type == 'value_string':
        value = str(data.get('value', ''))
        resolved_meta: str | None = None
        if value.startswith('sample:') and sample_uid is not None:
            sample = sample_crud.get(session, sample_uid)
            if sample is not None and sample.meta_data is not None:
                key = value.removeprefix('sample:')
                if key in sample.meta_data and sample.meta_data[key] is not None:
                    resolved_meta = str(sample.meta_data[key])
        return {'value': value, 'resolved_meta': resolved_meta}

    if node_type == 'bind_param':
        return {'parameter': str(data.get('parameter', ''))}

    if node_type == 'copy2folder':
        return {'folder_name': str(data.get('folder_name', ''))}

    if node_type in {'project_mark', 'sample_mark'}:
        return {
            'mark_name': str(data.get('mark_name', '')).strip(),
            'description': str(data.get('description', '')),
        }

    if node_type == 'resource_sample_mark_collection':
        mark_names = list(dict.fromkeys(str(name).strip() for name in data.get('mark_names', [])))
        entries: list[dict[str, Any]] = []
        if project_id is not None and mark_names:
            for sample_file, item_sample_name in sample_crud.get_dynamic_files_by_tags(session, project_id, mark_names):
                entries.append(
                    {
                        'sample_uid': str(sample_file.sample_uid),
                        'sample_name': item_sample_name,
                        'tag': sample_file.tag,
                        'file_path': sample_file.file_path,
                        'producer_run_uid': (
                            str(sample_file.producer_run_uid) if sample_file.producer_run_uid is not None else None
                        ),
                        'stat': _path_stat_signature(Path(sample_file.file_path)),
                    }
                )
        return {'project_id': project_id, 'mark_names': mark_names, 'entries': entries}

    if node_type in {'select_file_in_folder', 'read_file_content'}:
        return {'file_name': str(data.get('file_name', ''))}

    if node_type == 'join_path':
        return {'relative_path': str(data.get('relative_path', ''))}

    if node_type == 'collect_file_collection':
        return {'pattern': str(data.get('pattern', '')).strip()}

    if node_type == 'foreach':
        child = WorkflowDefinition.model_validate(data['workflow'])
        return {
            'item_port_id': str(data.get('item_port_id', '')),
            'subfolder_prefix': str(data.get('subfolder_prefix', '')),
            'subgraph': _compute_foreach_subgraph_fingerprint(
                session,
                child,
                tool_definitions=tool_definitions,
                tool_images=tool_images,
                project_id=project_id,
                sample_uid=sample_uid,
                sample_name=sample_name,
            ),
        }

    if node_type == 'rename_file':
        return {'new_file_name': str(data.get('new_file_name', ''))}

    if node_type == 'compress':
        return {
            'compressed_file_name': str(data.get('compressed_file_name', '')),
            'compression_format': data.get('compression_format'),
        }

    if node_type in {'code_bash', 'code_R'}:
        return {'code': str(data.get('code', ''))}

    if node_type == 'code_python':
        return {
            'code': str(data.get('code', '')),
            'dependencies': normalize_python_dependencies(data.get('dependencies')),
        }

    if node_type == 'downstream_summary':
        return {'prompt': str(data.get('prompt', ''))}

    if node_type in {'collect_mount_dirs', 'decompress'}:
        return {}

    return {'data': {key: value for key, value in data.items() if key not in {'run_data', 'preview_data'}}}


def compute_graph_fingerprints(
    session: Session,
    flat_graph: WorkflowDefinition,
    order: Sequence[str],
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
    tool_images: Mapping[str, dict[str, Any] | None],
    project_id: int | None,
    sample_uid: str | None,
    sample_name: str | None,
) -> dict[str, str]:
    """Compute topology-ordered Merkle DAG fingerprints independent of node IDs and indices."""
    node_map = {node.id: node for node in flat_graph.nodes if node.type not in PRESENTATION_NODE_TYPES}
    incoming_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for edge in flat_graph.edges:
        if edge.target in node_map and edge.source in node_map:
            source_key = edge.sourceHandle.removeprefix(f'{edge.source}-out-')
            target_key = edge.targetHandle.removeprefix(f'{edge.target}-in-')
            incoming_edges[edge.target].append((target_key, edge.source, source_key))

    fingerprints: dict[str, str] = {}
    for node_id in order:
        node = node_map[node_id]
        upstream = sorted(
            (target_key, fingerprints[source_id], source_key)
            for target_key, source_id, source_key in incoming_edges[node_id]
        )
        config = _node_self_config(
            session,
            node,
            tool_definitions=tool_definitions,
            tool_images=tool_images,
            project_id=project_id,
            sample_uid=sample_uid,
            sample_name=sample_name,
        )
        fingerprints[node_id] = _canonical_hash(
            {
                'v': FINGERPRINT_VERSION,
                'type': node.type,
                'config': config,
                'inputs': upstream,
            }
        )
    return fingerprints


def _legacy_resource_matches(old_node: NodeRun, resolved_paths: Sequence[Path]) -> bool:
    if old_node.tool_output is None:
        return False
    result = old_node.tool_output.result
    if old_node.node_type == 'resource_file':
        recorded = result.get('file_path')
        expected = [Path(str(recorded)).resolve()] if recorded is not None else []
    else:
        r1 = result.get('r1')
        r2 = result.get('r2')
        expected = [Path(str(r1)).resolve(), Path(str(r2)).resolve()] if r1 is not None and r2 is not None else []
    if len(expected) != len(resolved_paths):
        return False

    end_ts = _to_utc_timestamp(old_node.end_time) if old_node.end_time is not None else None
    for resolved, recorded_path in zip(resolved_paths, expected, strict=True):
        try:
            actual = resolved.resolve(strict=True)
            if actual != recorded_path:
                return False
            if end_ts is not None and actual.stat().st_mtime > end_ts + 1.0:
                return False
        except OSError:
            return False
    return True


def resolve_old_fingerprints(
    session: Session,
    previous_workflow: WorkflowDefinition | None,
    previous_nodes: Sequence[NodeRun],
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
    tool_images: Mapping[str, dict[str, Any] | None],
    project_id: int | None,
    sample_uid: str | None,
    sample_name: str | None,
) -> dict[uuid.UUID, str]:
    """Read persisted node fingerprints and lazily backfill legacy runs from their workflow snapshot."""
    resolved: dict[uuid.UUID, str] = {}
    missing: list[NodeRun] = []
    for old_node in previous_nodes:
        stored = (old_node.prepared_data or {}).get('fingerprint')
        if isinstance(stored, str) and stored:
            resolved[old_node.uid] = stored
        else:
            missing.append(old_node)

    if not missing or previous_workflow is None:
        return resolved

    try:
        prev_flat = flatten_workflow(lower_batch_subgraphs(previous_workflow))
        prev_node_map = {node.id: node for node in prev_flat.nodes if node.type not in PRESENTATION_NODE_TYPES}
        prev_edges = [edge for edge in prev_flat.edges if edge.target in prev_node_map and edge.source in prev_node_map]
        prev_order = order_nodes(list(prev_node_map), [(edge.source, edge.target) for edge in prev_edges])
        backfilled = compute_graph_fingerprints(
            session,
            prev_flat,
            prev_order,
            tool_definitions=tool_definitions,
            tool_images=tool_images,
            project_id=project_id,
            sample_uid=sample_uid,
            sample_name=sample_name,
        )
    except Exception:
        logger.exception('Failed to backfill legacy workflow fingerprints; falling back to rerun')
        return resolved

    for old_node in missing:
        fp = backfilled.get(old_node.definition_node_id)
        if fp is None:
            continue
        if old_node.node_type in {'resource_file', 'resource_sequence'}:
            kwargs = (old_node.prepared_data or {}).get('kwargs', {})
            refs = (
                [str(kwargs.get('file', ''))]
                if old_node.node_type == 'resource_file'
                else [str(kwargs.get('r1', 'sample:r1')), str(kwargs.get('r2', 'sample:r2'))]
            )
            try:
                resolved_paths = [resolve_path(ref, session, project_id, sample_uid)[0] for ref in refs]
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if not _legacy_resource_matches(old_node, resolved_paths):
                continue
        resolved[old_node.uid] = fp
    return resolved


def _verify_manifest_sources(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    entries = payload.get('entries')
    if not isinstance(entries, list) or not entries:
        return False
    return all(
        isinstance(entry, dict) and entry.get('source_path') and Path(str(entry['source_path'])).exists()
        for entry in entries
    )


def node_artifacts_intact(
    session: Session,
    run: WorkflowRun,
    old_node: NodeRun,
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
) -> bool:
    """Verify that a previously succeeded node still has valid DB and filesystem artifacts."""
    if old_node.status != NodeRunStatus.SUCCEEDED or old_node.tool_output is None:
        return False

    run_root = Path(run.base_dir).resolve()
    if old_node.base_dir:
        try:
            node_dir = Path(old_node.base_dir).resolve(strict=True)
        except OSError:
            return False
        if not node_dir.is_dir() or not node_dir.is_relative_to(run_root):
            return False

    output = old_node.tool_output
    result = output.result
    node_type = old_node.node_type
    kwargs = (old_node.prepared_data or {}).get('kwargs', {})

    if node_type == 'tool':
        tool = tool_definitions.get(str(old_node.tool_uid)) if old_node.tool_uid is not None else None
        if tool is None:
            return False
        required_output_names = frozenset(
            'log' if file.is_log else file.name
            for file in tool.file_mounts
            if (file.file_type == 'OUTPUT' and file.is_required) or file.is_passthrough
        )
        if not all_output_paths_exist(output, required_output_names=required_output_names):
            return False
        if (ws := output.workspace) is not None:
            if ws.run_uid != str(run.uid) or not Path(ws.path).is_dir():
                return False
            if not all(Path(src).exists() for src in ws.mounts.values()):
                return False
        return True

    if node_type in {'value_string', 'read_file_content'}:
        return 'value' in result

    if node_type == 'bind_param':
        return 'file_with_param' in result

    if node_type == 'join_path':
        return 'path' in result

    if node_type == 'collect_mount_dirs':
        raw = str(result.get('mount_dir_list', ''))
        if not raw.startswith('mountlist:'):
            return False
        return _verify_manifest_sources(Path(raw.removeprefix('mountlist:')))

    if node_type == 'resource_sample_mark_collection':
        raw = result.get('collection_manifest')
        return raw is not None and _verify_manifest_sources(Path(str(raw)))

    if node_type == 'collect_file_collection':
        raw = result.get('collection')
        if raw is None or not Path(str(raw)).is_file():
            return False
        try:
            items = read_collection(str(raw))
        except (OSError, ValueError, KeyError):
            return False
        return bool(items) and all(Path(item['path']).exists() for item in items)

    if node_type == 'foreach':
        results_folder = result.get('results_folder')
        if results_folder is None:
            return False
        folder = Path(str(results_folder))
        return folder.is_dir() and (folder / 'manifest.json').is_file()

    if node_type == 'downstream_summary':
        if not old_node.base_dir:
            return False
        summary_dir = Path(old_node.base_dir)
        if not summary_dir.is_dir() or not (summary_dir.parent / 'run_downstream.sh').is_file():
            return False
        return all(entry.exists() for entry in summary_dir.iterdir())

    if node_type == 'project_mark':
        marked = result.get('file_path')
        if marked is None or not output_path_exists(marked) or run.project_id is None:
            return False
        keyword = str(kwargs.get('mark_name', '')).strip()
        mapping = project_file_mapping_crud.get_by_keyword(session, run.project_id, keyword)
        return (
            mapping is not None
            and mapping.is_dynamic
            and mapping.producer_workflow_uid == run.workflow_uid
            and mapping.producer_run_uid == run.uid
            and mapping.file_path == str(Path(str(marked)).absolute())
            and mapping.description == str(kwargs.get('description', ''))
        )

    if node_type == 'sample_mark':
        marked = result.get('file_path')
        if marked is None or not output_path_exists(marked) or run.sample_uid is None:
            return False
        tag = str(kwargs.get('mark_name', '')).strip()
        sample_file = sample_crud.get_file_by_tag(session, run.sample_uid, tag)
        return (
            sample_file is not None
            and sample_file.is_dynamic
            and sample_file.producer_workflow_uid == run.workflow_uid
            and sample_file.producer_run_uid == run.uid
            and sample_file.file_path == str(Path(str(marked)).absolute())
            and sample_file.description == str(kwargs.get('description', ''))
        )

    return bool(result) and all(output_path_exists(value) for value in result.values())


def _resolve_old_passthrough_origin_dir(old_by_uid: Mapping[str, NodeRun], source_uid: str) -> Path | None:
    seen: set[str] = set()
    current_uid = source_uid
    while current_uid and current_uid not in seen:
        seen.add(current_uid)
        node = old_by_uid.get(current_uid)
        if node is None:
            return None
        base_input = PATH_INPUTS.get(node.node_type)
        if base_input is None:
            return Path(node.base_dir).resolve() if node.base_dir else None
        upstream = (node.prepared_data or {}).get('kwargs', {}).get('inputs', {}).get(base_input)
        if not isinstance(upstream, dict):
            return None
        current_uid = str(upstream.get('source_uid', ''))
    return None


def _paths_overlap(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _drop_with_descendants(
    reused: dict[str, NodeRun],
    root_id: str,
    new_children: Mapping[str, Iterable[str]],
) -> None:
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current in reused:
            reused.pop(current)
            stack.extend(new_children.get(current, ()))


def prune_passthrough_tainted_reuses(
    reused: dict[str, NodeRun],
    previous_nodes: Sequence[NodeRun],
    new_children: Mapping[str, Iterable[str]],
    *,
    tool_definitions: Mapping[str, WorkflowToolDefinition],
) -> None:
    """Invalidate reused nodes whose workspace or sidecar directory was touched by a non-reused started node."""
    old_by_uid = {str(node.uid): node for node in previous_nodes}
    while True:
        reused_uids = {node.uid for node in reused.values()}
        dirty_dirs: set[Path] = set()
        for old_node in previous_nodes:
            if old_node.uid in reused_uids or old_node.start_time is None:
                continue
            if old_node.base_dir:
                dirty_dirs.add(Path(old_node.base_dir).resolve())
            if old_node.tool_output is not None and old_node.tool_output.workspace is not None:
                ws = old_node.tool_output.workspace
                dirty_dirs.add(Path(ws.path).resolve())
                for src in ws.mounts.values():
                    dirty_dirs.add(Path(src).resolve())
            if old_node.node_type == 'tool' and old_node.tool_uid is not None:
                tool = tool_definitions.get(str(old_node.tool_uid))
                if tool is not None and tool.tool_type == ToolType.DOCKER_PASSTHROUGH:
                    raw_inputs = (old_node.prepared_data or {}).get('kwargs', {}).get('inputs', {})
                    for file in tool.file_mounts:
                        if not file.is_passthrough:
                            continue
                        ref = raw_inputs.get(file.name)
                        if isinstance(ref, dict) and (source_uid := ref.get('source_uid')):
                            origin_dir = _resolve_old_passthrough_origin_dir(old_by_uid, str(source_uid))
                            if origin_dir is not None:
                                dirty_dirs.add(origin_dir)

        if not dirty_dirs:
            return

        tainted_ids: list[str] = []
        for new_id, old_node in list(reused.items()):
            node_dirs: list[Path] = []
            if old_node.base_dir:
                node_dirs.append(Path(old_node.base_dir).resolve())
            if old_node.tool_output is not None and old_node.tool_output.workspace is not None:
                node_dirs.append(Path(old_node.tool_output.workspace.path).resolve())
            if any(_paths_overlap(node_dir, dirty_dir) for node_dir in node_dirs for dirty_dir in dirty_dirs):
                tainted_ids.append(new_id)

        if not tainted_ids:
            return
        for new_id in tainted_ids:
            _drop_with_descendants(reused, new_id, new_children)
