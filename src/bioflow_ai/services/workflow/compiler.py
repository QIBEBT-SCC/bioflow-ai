"""Validation and durable compilation of workflow definitions."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from bioflow_ai.crud import node_dependency as dependency_crud
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.models.workflow import NodeDependency, NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.tools.cache import get_tool_definitions_sync
from bioflow_ai.services.workflow.batch_graph import lower_batch_subgraphs
from bioflow_ai.services.workflow.executors import (
    ExecutorContext,
    node_numbered_dir_suffix,
    prepare_node,
)
from bioflow_ai.services.workflow.fingerprint import (
    compute_graph_fingerprints,
    load_tool_images,
    node_artifacts_intact,
    parse_dir_index,
    prune_passthrough_tainted_reuses,
    resolve_old_fingerprints,
)
from bioflow_ai.services.workflow.subgraphs import flatten_workflow
from bioflow_ai.services.workflow.validation import validate_workflow
from bioflow_ai.services.workflow.validation._topology import order_nodes
from bioflow_ai.services.workflow.validation.registry import PRESENTATION_NODE_TYPES
from bioflow_ai.services.workflow.validation.resolution import collect_tool_uids, resolve_workflow
from bioflow_ai.utils.system import utc_now


class WorkflowCompileError(ValueError):
    """The submitted workflow failed the shared validation operation."""


class CompileResult(BaseModel):
    """Compilation outcome used by reruns to prune stale files and dynamic marks."""

    model_config = ConfigDict(frozen=True)

    stale_dirs: list[Path] = Field(default_factory=list)
    stale_files: list[Path] = Field(default_factory=list)
    reused_project_marks: set[str] = Field(default_factory=set)
    reused_sample_marks: set[str] = Field(default_factory=set)
    reused_count: int = 0
    executed_count: int = 0


def _paths_overlap(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _match_reusable_nodes(
    session: Session,
    run: WorkflowRun,
    *,
    order: Sequence[str],
    source_nodes_by_target: dict[str, set[str]],
    children_by_source: dict[str, set[str]],
    new_fingerprints: dict[str, str],
    previous_workflow: WorkflowDefinition | None,
    previous_nodes: Sequence[NodeRun],
    tool_definitions: dict,
    tool_images: dict,
    sample_name: str | None,
) -> dict[str, NodeRun]:
    sample_uid_str = str(run.sample_uid) if run.sample_uid else None
    old_fingerprints = resolve_old_fingerprints(
        session,
        previous_workflow,
        previous_nodes,
        tool_definitions=tool_definitions,
        tool_images=tool_images,
        project_id=run.project_id,
        sample_uid=sample_uid_str,
        sample_name=sample_name,
    )
    candidates_by_fp: dict[str, list[NodeRun]] = defaultdict(list)
    for old_node in previous_nodes:
        fingerprint = old_fingerprints.get(old_node.uid)
        if fingerprint and node_artifacts_intact(session, run, old_node, tool_definitions=tool_definitions):
            candidates_by_fp[fingerprint].append(old_node)

    reused: dict[str, NodeRun] = {}
    for node_id in order:
        if not all(parent in reused for parent in source_nodes_by_target[node_id]):
            continue
        candidates = candidates_by_fp.get(new_fingerprints[node_id])
        if not candidates:
            continue
        preferred = next(
            (index for index, candidate in enumerate(candidates) if candidate.definition_node_id == node_id),
            0,
        )
        reused[node_id] = candidates.pop(preferred)

    prune_passthrough_tainted_reuses(
        reused,
        previous_nodes,
        children_by_source,
        tool_definitions=tool_definitions,
    )
    return reused


def compile_run(
    session: Session,
    run: WorkflowRun,
    *,
    sample_name: str | None,
    previous_workflow: WorkflowDefinition | None = None,
    previous_nodes: Sequence[NodeRun] | None = None,
) -> CompileResult:
    """Compile ``run.workflow`` into NodeRun and dependency rows, reusing intact nodes when rerunning.

    The caller owns the surrounding transaction. Nothing is published here.
    """
    document = resolve_workflow(
        run.workflow,
        session,
        fresh=True,
        project_id=run.project_id,
        execution_scope='sample_level' if run.sample_uid is not None else 'project_level',
        workflow_uid=run.workflow_uid,
    )
    result = validate_workflow(document)
    if not result.valid:
        raise WorkflowCompileError(result.message)

    compiled_workflow = flatten_workflow(lower_batch_subgraphs(run.workflow))
    node_map = {node.id: node for node in compiled_workflow.nodes if node.type not in PRESENTATION_NODE_TYPES}
    edges = [edge for edge in compiled_workflow.edges if edge.target in node_map]
    order = order_nodes(list(node_map), [(edge.source, edge.target) for edge in edges])

    source_nodes_by_target: dict[str, set[str]] = defaultdict(set)
    children_by_source: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source_nodes_by_target[edge.target].add(edge.source)
        children_by_source[edge.source].add(edge.target)

    all_tool_definitions = dict(document.catalog or {})
    if previous_workflow is not None or previous_nodes:
        extra_tool_uids: set[str] = set()
        if previous_workflow is not None:
            extra_tool_uids.update(collect_tool_uids(previous_workflow.model_dump(mode='json')))
        if previous_nodes:
            extra_tool_uids.update(str(old.tool_uid) for old in previous_nodes if old.tool_uid is not None)
        missing_uids = [uid for uid in extra_tool_uids if uid not in all_tool_definitions]
        if missing_uids:
            all_tool_definitions.update(get_tool_definitions_sync(session, missing_uids, fresh=True))

    tool_images = load_tool_images(session, all_tool_definitions.keys())
    sample_uid_str = str(run.sample_uid) if run.sample_uid else None
    new_fingerprints = compute_graph_fingerprints(
        session,
        compiled_workflow,
        order,
        tool_definitions=all_tool_definitions,
        tool_images=tool_images,
        project_id=run.project_id,
        sample_uid=sample_uid_str,
        sample_name=sample_name,
    )

    reused: dict[str, NodeRun] = {}
    stale_dirs: list[Path] = []
    stale_files: list[Path] = []
    used_indices: set[int] = set()
    used_dir_names: set[str] = set()

    if previous_nodes:
        reused = _match_reusable_nodes(
            session,
            run,
            order=order,
            source_nodes_by_target=source_nodes_by_target,
            children_by_source=children_by_source,
            new_fingerprints=new_fingerprints,
            previous_workflow=previous_workflow,
            previous_nodes=previous_nodes,
            tool_definitions=all_tool_definitions,
            tool_images=tool_images,
            sample_name=sample_name,
        )
        run_root = Path(run.base_dir).resolve()
        reused_uids = {nr.uid for nr in reused.values()}
        kept_dirs: set[Path] = set()
        for nr in reused.values():
            if nr.base_dir:
                kept_dir = Path(nr.base_dir).resolve()
                kept_dirs.add(kept_dir)
                if kept_dir.parent == run_root:
                    used_dir_names.add(kept_dir.name)
                    if (parsed := parse_dir_index(kept_dir.name)) is not None:
                        used_indices.add(parsed)

        seen_stale_dirs: set[Path] = set()
        for old_node in previous_nodes:
            if old_node.uid in reused_uids:
                continue
            for raw_dir in (
                old_node.base_dir,
                (old_node.prepared_data or {}).get('kwargs', {}).get('output_dir'),
            ):
                if not raw_dir:
                    continue
                candidate = Path(str(raw_dir)).resolve()
                if (
                    candidate != run_root
                    and candidate.is_relative_to(run_root)
                    and not any(_paths_overlap(candidate, kept) for kept in kept_dirs)
                    and candidate not in seen_stale_dirs
                ):
                    seen_stale_dirs.add(candidate)
                    stale_dirs.append(candidate)
            if old_node.node_type == 'downstream_summary' and not any(
                nr.node_type == 'downstream_summary' for nr in reused.values()
            ):
                script = run_root / 'run_downstream.sh'
                if script not in stale_files:
                    stale_files.append(script)

        dependency_crud.delete_for_run(session, run.uid)
        node_run_crud.delete_by_uids(session, [n.uid for n in previous_nodes if n.uid not in reused_uids])
        session.flush()

        renamed = False
        for new_id, old_node in reused.items():
            if old_node.definition_node_id != new_id:
                old_node.definition_node_id = f'__reuse_tmp__/{old_node.uid}'
                renamed = True
        if renamed:
            session.flush()

    node_runs: dict[str, NodeRun] = {}
    for position_index, node_id in enumerate(order, start=1):
        node = node_map[node_id]
        if node_id in reused:
            node_run = reused[node_id]
            node_run.definition_node_id = node_id
            node_run.position_index = position_index
        else:
            node_run = NodeRun(
                run_uid=run.uid,
                definition_node_id=node_id,
                node_type=node.type,
                name=node.type,
                position_index=position_index,
                status=NodeRunStatus.PENDING,
            )
            node_run_crud.insert(session, node_run)
        node_runs[node_id] = node_run
    session.flush()

    incoming: dict[str, dict[str, dict[str, str] | list[dict[str, str]]]] = defaultdict(dict)
    for edge in edges:
        source_key = edge.sourceHandle.removeprefix(f'{edge.source}-out-')
        target_key = edge.targetHandle.removeprefix(f'{edge.target}-in-')
        source_run = node_runs[edge.source]
        target_run = node_runs[edge.target]
        reference = {'source_uid': str(source_run.uid), 'source_k': source_key}
        existing = incoming[edge.target].get(target_key)
        if existing is None:
            incoming[edge.target][target_key] = reference
        elif isinstance(existing, list):
            existing.append(reference)
        else:
            incoming[edge.target][target_key] = [existing, reference]
        dependency_crud.insert(
            session,
            NodeDependency(
                run_uid=run.uid,
                source_node_uid=source_run.uid,
                target_node_uid=target_run.uid,
                source_handle=source_key,
                target_handle=target_key,
            ),
        )

    context = ExecutorContext(run=run, sample_name=sample_name, tool_definitions=dict(document.catalog or {}))
    effective_dir_index: dict[str, int] = {}
    for node_id in order:
        node_run = node_runs[node_id]
        parent_max = max((effective_dir_index[parent] for parent in source_nodes_by_target[node_id]), default=0)
        if node_id in reused:
            parsed = parse_dir_index(Path(node_run.base_dir).name) if node_run.base_dir else None
            effective_dir_index[node_id] = parsed if parsed is not None else parent_max
            prepared_data = dict(node_run.prepared_data or {})
            kwargs = dict(prepared_data.get('kwargs') or {})
            if 'inputs' in kwargs:
                kwargs['inputs'] = incoming[node_id]
            prepared_data['kwargs'] = kwargs
            prepared_data['fingerprint'] = new_fingerprints[node_id]
            prepared_data['reused'] = True
            node_run.prepared_data = prepared_data
        else:
            suffix = node_numbered_dir_suffix(node_map[node_id], context, used_dir_names=used_dir_names)
            if suffix is not None:
                dir_index = parent_max + 1
                while dir_index in used_indices or f'{dir_index:02d}_{suffix}' in used_dir_names:
                    dir_index += 1
                used_indices.add(dir_index)
                used_dir_names.add(f'{dir_index:02d}_{suffix}')
                effective_dir_index[node_id] = dir_index
            else:
                dir_index = 0
                effective_dir_index[node_id] = parent_max

            prepared = prepare_node(
                node_map[node_id],
                incoming[node_id],
                dir_index,
                context,
                used_dir_names=used_dir_names,
            )
            if prepared.base_dir:
                used_dir_names.add(Path(prepared.base_dir).name)
            node_run.name = prepared.name
            node_run.track_progress = prepared.track_progress
            node_run.tool_uid = prepared.tool_uid
            node_run.base_dir = prepared.base_dir
            node_run.prepared_data = {
                'executor_name': node_run.node_type,
                'kwargs': prepared.kwargs,
                'fingerprint': new_fingerprints[node_id],
                'reused': False,
            }
            if all(node_runs[parent].status == NodeRunStatus.SUCCEEDED for parent in source_nodes_by_target[node_id]):
                node_run.status = NodeRunStatus.READY

    if not node_runs or all(nr.status == NodeRunStatus.SUCCEEDED for nr in node_runs.values()):
        now = utc_now()
        run.status = RunStatus.SUCCEEDED
        run.settled = True
        run.start_time = run.start_time or now
        run.end_time = now

    reused_project_marks: set[str] = set()
    reused_sample_marks: set[str] = set()
    for node_run in reused.values():
        mark_name = str((node_run.prepared_data or {}).get('kwargs', {}).get('mark_name', '')).strip()
        if not mark_name:
            continue
        if node_run.node_type == 'project_mark':
            reused_project_marks.add(mark_name)
        elif node_run.node_type == 'sample_mark':
            reused_sample_marks.add(mark_name)

    session.flush()
    return CompileResult(
        stale_dirs=stale_dirs,
        stale_files=stale_files,
        reused_project_marks=reused_project_marks,
        reused_sample_marks=reused_sample_marks,
        reused_count=len(reused),
        executed_count=len(node_runs) - len(reused),
    )


__all__ = ['CompileResult', 'WorkflowCompileError', 'compile_run']
