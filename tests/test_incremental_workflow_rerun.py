# ruff: noqa: INP001, PLR2004, S106
"""Offline tests for Merkle-fingerprint incremental workflow reruns and pinned directory numbering."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.models.project import Project, ProjectFileMapping
from bioflow_ai.models.resource import BioDb, Sample, SampleFiles
from bioflow_ai.models.tools import ToolDefine, ToolGroup, ToolHelpDoc, ToolImage, ToolTag, ToolTagLink
from bioflow_ai.models.user import User
from bioflow_ai.models.workflow import NodeDependency, NodeRun, ProjectWorkflow, Workflow, WorkflowRun
from bioflow_ai.schemas.passthrough import ToolWorkspace
from bioflow_ai.schemas.react_flow import (
    InterfaceInput,
    InterfaceOutput,
    PortTarget,
    WorkflowDefinition,
    WorkflowInterface,
)
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ToolType
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.schemas.workflow_config import ExecutionScope, ToolOutput, WorkflowType
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.workflow import runs as workflow_runs_service
from bioflow_ai.utils.system import random_uid, utc_now

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator


@compiles(JSONB, 'sqlite')
def _compile_jsonb_sqlite(_element: JSONB, _compiler: object, **_kw: object) -> str:
    return 'JSON'


@compiles(ARRAY, 'sqlite')
def _compile_array_sqlite(_element: ARRAY, _compiler: object, **_kw: object) -> str:
    return 'JSON'


@pytest.fixture
def db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    """Create an isolated SQLite in-memory database and temporary workflow storage."""
    storage_root = tmp_path / 'storage'
    storage_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workflow_runs_service, 'STORAGE_PATH', str(storage_root))

    engine = create_engine('sqlite:///:memory:')

    @event.listens_for(engine, 'connect')
    def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.close()

    tables = [
        User.__table__,
        Project.__table__,
        ProjectFileMapping.__table__,
        Sample.__table__,
        SampleFiles.__table__,
        BioDb.__table__,
        ToolTag.__table__,
        ToolGroup.__table__,
        ToolImage.__table__,
        ToolHelpDoc.__table__,
        ToolDefine.__table__,
        ToolTagLink.__table__,
        Workflow.__table__,
        ProjectWorkflow.__table__,
        WorkflowRun.__table__,
        NodeRun.__table__,
        NodeDependency.__table__,
    ]
    for table in tables:
        table.create(bind=engine)

    with Session(engine) as session:
        yield session


def _node(node_id: str, node_type: str, data: dict | None = None) -> dict:
    return {
        'id': node_id,
        'type': node_type,
        'position': {'x': 0, 'y': 0},
        'dragHandle': '.nodeDragable',
        'zIndex': 20,
        'data': data or {},
    }


def _edge(edge_id: str, source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {
        'id': edge_id,
        'source': source,
        'sourceHandle': f'{source}-out-{source_handle}',
        'target': target,
        'targetHandle': f'{target}-in-{target_handle}',
    }


def _seed_user_project(
    session: Session,
    tmp_path: Path,
    *,
    create_sample: bool = True,
) -> tuple[CurrentUser, Project, Sample | None, Path, Path]:
    user = User(
        username='tester',
        email='tester@example.com',
        role=UserRole.ADMIN,
        hashed_password='x',
    )
    session.add(user)
    session.flush()

    project = Project(owner_id=user.id, name='ocean_test')
    session.add(project)
    session.flush()

    r1_path = tmp_path / 'sample_R1.fq.gz'
    r2_path = tmp_path / 'sample_R2.fq.gz'
    r1_path.write_text('@read1\nACGT\n+\nIIII\n', encoding='utf-8')
    r2_path.write_text('@read2\nTGCA\n+\nIIII\n', encoding='utf-8')

    sample: Sample | None = None
    if create_sample:
        sample = Sample(owner_id=user.id, project_id=project.id, sample_name='SRR3960581', meta_data={})
        session.add(sample)
        session.flush()
        session.add_all(
            [
                SampleFiles(sample_uid=sample.uid, file_path=str(r1_path), tag='r1'),
                SampleFiles(sample_uid=sample.uid, file_path=str(r2_path), tag='r2'),
            ]
        )
        session.flush()

    current_user = CurrentUser(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=True,
    )
    return current_user, project, sample, r1_path, r2_path


def _create_tool(
    session: Session,
    *,
    name: str,
    tool_type: ToolType = ToolType.DOCKER,
    inputs: list[tuple[str, bool]] | None = None,
    outputs: list[tuple[str, str, bool]] | None = None,
    command_template: str = 'run_cmd',
    image_tag: str = '1.0',
) -> ToolDefine:
    # Avoid ARRAY bind issues on SQLite by inserting ToolImage via raw SQL.
    image_uid = random_uid()
    image_cfg = ImageConfig(repository='quay.io/test', name=name.lower().replace(' ', '-'), tag=image_tag)
    session.execute(
        text(
            'INSERT INTO images (uid, ref, name, aliases, version, description, image) '
            'VALUES (:uid, :ref, :name, :aliases, :version, :description, :image)'
        ),
        {
            'uid': image_uid.hex,
            'ref': f'test/{name}:{image_tag}',
            'name': name,
            'aliases': '[]',
            'version': image_tag,
            'description': name,
            'image': image_cfg.model_dump_json(),
        },
    )
    file_mounts: list[FileMount] = []
    for input_name, is_passthrough in inputs or []:
        file_mounts.append(
            FileMount(
                name=input_name,
                description=input_name,
                file_type='INPUT',
                file_path=f'passthrough:{{{input_name}}}' if is_passthrough else f'{{{input_name}}}',
                mount_path='/data/raw',
                is_passthrough=is_passthrough,
            )
        )
    for out_name, out_path, is_required in outputs or []:
        file_mounts.append(
            FileMount(
                name=out_name,
                description=out_name,
                file_type='OUTPUT',
                file_path=out_path,
                mount_path='/data/output',
                is_required=is_required,
            )
        )
    tool = ToolDefine(
        uid=random_uid(),
        name=name,
        image_uid=image_uid,
        description=name,
        tool_type=tool_type,
        command_template=command_template,
        dynamic_params=[],
        immutable_static_params='',
        modifiable_static_params='',
        file_mounts=file_mounts,
    )
    session.add(tool)
    session.flush()
    return tool


def _attach_workflow(
    session: Session,
    project: Project,
    owner_id: int,
    definition: WorkflowDefinition,
    *,
    scope: ExecutionScope = ExecutionScope.SAMPLE_LEVEL,
) -> Workflow:
    wf = Workflow(
        uid=random_uid(),
        name='test_wf',
        owner_id=owner_id,
        description='desc',
        wf_type=WorkflowType.TEMPLATE,
        execution_scope=scope,
        workflow=definition,
    )
    session.add(wf)
    session.flush()
    session.add(ProjectWorkflow(project_id=project.id, workflow_uid=wf.uid))
    session.flush()
    return wf


def _mark_node_succeeded_with_files(
    session: Session,
    node: NodeRun,
    outputs: dict[str, Path | str],
    *,
    workspace: ToolWorkspace | None = None,
    create_files: bool = True,
) -> None:
    if node.base_dir and create_files:
        Path(node.base_dir).mkdir(parents=True, exist_ok=True)
    if create_files:
        for path_value in outputs.values():
            raw = str(path_value)
            if raw.startswith('mountlist:'):
                continue
            p = Path(raw)
            if p.suffix:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('ok\n', encoding='utf-8')
            else:
                p.mkdir(parents=True, exist_ok=True)
    node.status = NodeRunStatus.SUCCEEDED
    node.tool_output = ToolOutput(result=outputs, workspace=workspace, sample_name='SRR3960581')
    now = utc_now()
    node.start_time = node.start_time or now
    node.end_time = now
    session.flush()


def _node_by_id(session: Session, run_uid: uuid.UUID) -> dict[str, NodeRun]:
    nodes = node_run_crud.list_for_runs(session, [run_uid])
    return {node.definition_node_id: node for node in nodes}


def test_first_run_numbers_only_dir_nodes_and_persists_fingerprints(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, _, _ = _seed_user_project(db_session, tmp_path)
    fastp = _create_tool(
        db_session,
        name='fastp',
        inputs=[('read1', False), ('read2', False)],
        outputs=[('filtered_read1', '{output_dir}/r1.fq.gz', True)],
    )
    megahit = _create_tool(
        db_session,
        name='MEGAHIT',
        inputs=[('r1', False)],
        outputs=[('contigs', '{output_dir}/final.contigs.fa', True)],
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('A', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('B', 'tool', {'tool_uid': str(fastp.uid)}),
                _node('C', 'tool', {'tool_uid': str(megahit.uid)}),
            ],
            'edges': [
                _edge('e1', 'A', 'r1', 'B', 'read1'),
                _edge('e2', 'A', 'r2', 'B', 'read2'),
                _edge('e3', 'B', 'filtered_read1', 'C', 'r1'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()

    assert sample is not None
    result = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes = _node_by_id(db_session, result.run_uids[0])
    assert nodes['A'].base_dir is None
    assert nodes['A'].status == NodeRunStatus.READY
    assert Path(nodes['B'].base_dir or '').name == '01_fastp'
    assert Path(nodes['C'].base_dir or '').name == '02_megahit'
    for node in nodes.values():
        assert isinstance(node.prepared_data.get('fingerprint'), str)
        assert node.prepared_data.get('reused') is False


def test_project_158_insert_samtools_view_reuses_upstream_and_allocates_clean_dir(
    db_session: Session, tmp_path: Path
) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    fastp = _create_tool(
        db_session,
        name='fastp',
        inputs=[('read1', False), ('read2', False)],
        outputs=[('filtered_read1', '{output_dir}/r1.fq', True), ('filtered_read2', '{output_dir}/r2.fq', True)],
    )
    megahit = _create_tool(
        db_session,
        name='MEGAHIT',
        inputs=[('r1', False), ('r2', False)],
        outputs=[('contigs', '{output_dir}/contigs.fa', True)],
    )
    bowtie2 = _create_tool(
        db_session,
        name='Bowtie2',
        inputs=[('reads_1', False), ('reads_2', False), ('contigs', False)],
        outputs=[('sam', '{output_dir}/aligned.sam', True)],
    )
    samtools_view = _create_tool(
        db_session,
        name='samtools view',
        inputs=[('sam', False)],
        outputs=[('bam', '{output_dir}/aligned.bam', True)],
    )
    samtools_sort = _create_tool(
        db_session,
        name='samtools sort BAM',
        inputs=[('bam_in', False)],
        outputs=[('bam_sorted', '{output_dir}/sorted.bam', True)],
    )
    antismash = _create_tool(
        db_session,
        name='antiSMASH',
        inputs=[('genome', False)],
        outputs=[('results', '{output_dir}/results', True)],
    )

    initial_defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('A', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('B', 'tool', {'tool_uid': str(fastp.uid)}),
                _node('C', 'tool', {'tool_uid': str(megahit.uid)}),
                _node('F', 'tool', {'tool_uid': str(bowtie2.uid)}),
                _node('G', 'tool', {'tool_uid': str(samtools_sort.uid)}),
                _node('S', 'tool', {'tool_uid': str(antismash.uid)}),
            ],
            'edges': [
                _edge('e1', 'A', 'r1', 'B', 'read1'),
                _edge('e2', 'A', 'r2', 'B', 'read2'),
                _edge('e3', 'B', 'filtered_read1', 'C', 'r1'),
                _edge('e4', 'B', 'filtered_read2', 'C', 'r2'),
                _edge('e5', 'B', 'filtered_read1', 'F', 'reads_1'),
                _edge('e6', 'B', 'filtered_read2', 'F', 'reads_2'),
                _edge('e7', 'C', 'contigs', 'F', 'contigs'),
                _edge('e8', 'F', 'sam', 'G', 'bam_in'),
                _edge('e9', 'C', 'contigs', 'S', 'genome'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, initial_defn)
    db_session.commit()

    assert sample is not None
    res1 = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    run_uid = res1.run_uids[0]
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    nodes1 = _node_by_id(db_session, run_uid)

    # Simulate A, B, C, F, S succeeded and G failed.
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    _mark_node_succeeded_with_files(
        db_session,
        nodes1['B'],
        {
            'filtered_read1': Path(nodes1['B'].base_dir or '') / 'r1.fq',
            'filtered_read2': Path(nodes1['B'].base_dir or '') / 'r2.fq',
        },
    )
    _mark_node_succeeded_with_files(
        db_session, nodes1['C'], {'contigs': Path(nodes1['C'].base_dir or '') / 'contigs.fa'}
    )
    _mark_node_succeeded_with_files(db_session, nodes1['F'], {'sam': Path(nodes1['F'].base_dir or '') / 'aligned.sam'})
    _mark_node_succeeded_with_files(db_session, nodes1['S'], {'results': Path(nodes1['S'].base_dir or '') / 'results'})
    failed_g_dir = Path(nodes1['G'].base_dir or '')
    failed_g_dir.mkdir(parents=True, exist_ok=True)
    (failed_g_dir / 'partial.tmp').write_text('bad', encoding='utf-8')
    nodes1['G'].status = NodeRunStatus.FAILED
    nodes1['G'].start_time = utc_now()
    nodes1['G'].end_time = utc_now()
    run.status = RunStatus.FAILED
    run.settled = True
    db_session.commit()

    old_uids = {k: nodes1[k].uid for k in ('A', 'B', 'C', 'F', 'S')}
    old_dirs = {k: nodes1[k].base_dir for k in ('B', 'C', 'F', 'S')}
    old_base_dir = run.base_dir

    # Edit workflow: insert samtools view between F and G.
    wf.workflow = WorkflowDefinition.model_validate(
        {
            'nodes': [
                *initial_defn.model_dump()['nodes'],
                _node('V', 'tool', {'tool_uid': str(samtools_view.uid)}),
            ],
            'edges': [
                _edge('e1', 'A', 'r1', 'B', 'read1'),
                _edge('e2', 'A', 'r2', 'B', 'read2'),
                _edge('e3', 'B', 'filtered_read1', 'C', 'r1'),
                _edge('e4', 'B', 'filtered_read2', 'C', 'r2'),
                _edge('e5', 'B', 'filtered_read1', 'F', 'reads_1'),
                _edge('e6', 'B', 'filtered_read2', 'F', 'reads_2'),
                _edge('e7', 'C', 'contigs', 'F', 'contigs'),
                _edge('e8a', 'F', 'sam', 'V', 'sam'),
                _edge('e8b', 'V', 'bam', 'G', 'bam_in'),
                _edge('e9', 'C', 'contigs', 'S', 'genome'),
            ],
        }
    )
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    db_session.expire_all()
    run2 = db_session.get(WorkflowRun, run_uid)
    assert run2 is not None
    assert run2.generation == 2
    assert run2.base_dir == old_base_dir
    assert not failed_g_dir.exists()

    nodes2 = _node_by_id(db_session, run_uid)
    for k in ('A', 'B', 'C', 'F', 'S'):
        assert nodes2[k].uid == old_uids[k]
        assert nodes2[k].status == NodeRunStatus.SUCCEEDED
        assert nodes2[k].prepared_data.get('reused') is True
        if k in old_dirs:
            assert nodes2[k].base_dir == old_dirs[k]

    assert nodes2['V'].status == NodeRunStatus.READY
    assert nodes2['V'].prepared_data.get('reused') is False
    # F was 04_bowtie2, S was 03_antismash -> max parent of V is 04 -> V gets 05_samtools-view.
    assert Path(nodes2['V'].base_dir or '').name == '05_samtools-view'
    assert nodes2['G'].status == NodeRunStatus.PENDING
    assert Path(nodes2['G'].base_dir or '').name == '06_samtools-sort-bam'


def test_node_id_renumbering_and_swap_reuses_without_collision(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    fastp = _create_tool(
        db_session,
        name='fastp',
        inputs=[('read1', False), ('read2', False)],
        outputs=[('filtered_read1', '{output_dir}/r1.fq', True)],
    )
    defn1 = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('A', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('B', 'tool', {'tool_uid': str(fastp.uid)}),
            ],
            'edges': [
                _edge('e1', 'A', 'r1', 'B', 'read1'),
                _edge('e2', 'A', 'r2', 'B', 'read2'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn1)
    db_session.commit()
    assert sample is not None
    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    _mark_node_succeeded_with_files(
        db_session, nodes1['B'], {'filtered_read1': Path(nodes1['B'].base_dir or '') / 'r1.fq'}
    )
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    old_a_uid, old_b_uid = nodes1['A'].uid, nodes1['B'].uid

    # Swap A <-> B IDs completely.
    wf.workflow = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('B', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(fastp.uid)}),
            ],
            'edges': [
                _edge('e1', 'B', 'r1', 'A', 'read1'),
                _edge('e2', 'B', 'r2', 'A', 'read2'),
            ],
        }
    )
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['B'].uid == old_a_uid
    assert nodes2['A'].uid == old_b_uid
    assert nodes2['B'].prepared_data.get('reused') is True
    assert nodes2['A'].prepared_data.get('reused') is True
    run_after = db_session.get(WorkflowRun, run_uid)
    assert run_after is not None
    assert run_after.status == RunStatus.SUCCEEDED
    assert run_after.settled is True
    public = workflow_runs_service.to_public(db_session, run_after)
    assert all(node['data']['run_data']['reused'] is True for node in public.nodes)


def test_downstream_summary_reuse_and_stale_script_cleanup(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    t1 = _create_tool(db_session, name='QC', inputs=[('r1', False)], outputs=[('out', '{output_dir}/qc.txt', True)])
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(t1.uid)}),
                _node('T', 'downstream_summary', {'prompt': 'summarize v1'}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'A', 'r1'),
                _edge('e2', 'A', 'out', 'T', 'input_files'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['R'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    qc_out = Path(nodes1['A'].base_dir or '') / 'qc.txt'
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'out': qc_out})

    summary_dir = Path(nodes1['T'].base_dir or '')
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / 'qc.txt').symlink_to(qc_out)
    run_script = summary_dir.parent / 'run_downstream.sh'
    run_script.write_text('claude -p "summarize v1"', encoding='utf-8')
    nodes1['T'].status = NodeRunStatus.SUCCEEDED
    nodes1['T'].tool_output = ToolOutput(result={})
    nodes1['T'].start_time = utc_now()
    nodes1['T'].end_time = utc_now()
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    # Unchanged rerun keeps downstream_summary & run_downstream.sh intact.
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['T'].prepared_data.get('reused') is True
    assert run_script.is_file()

    # Prompt change invalidates downstream_summary and removes stale dir + run_downstream.sh.
    updated_defn = wf.workflow.model_copy(deep=True)
    updated_defn.nodes[2].data['prompt'] = 'summarize v2'
    wf.workflow = updated_defn
    db_session.commit()
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes3 = _node_by_id(db_session, run_uid)
    assert nodes3['A'].prepared_data.get('reused') is True
    assert nodes3['T'].prepared_data.get('reused') is False
    assert nodes3['T'].status == NodeRunStatus.READY
    assert not summary_dir.exists()
    assert not run_script.exists()


def test_legacy_run_without_fingerprint_backfills_and_respects_mtime_guard(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    fastp = _create_tool(
        db_session,
        name='fastp',
        inputs=[('read1', False), ('read2', False)],
        outputs=[('filtered_read1', '{output_dir}/r1.fq', True)],
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('A', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('B', 'tool', {'tool_uid': str(fastp.uid)}),
            ],
            'edges': [
                _edge('e1', 'A', 'r1', 'B', 'read1'),
                _edge('e2', 'A', 'r2', 'B', 'read2'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    _mark_node_succeeded_with_files(
        db_session, nodes1['B'], {'filtered_read1': Path(nodes1['B'].base_dir or '') / 'r1.fq'}
    )
    # Strip stored fingerprints to simulate a pre-upgrade legacy run (like project 158).
    for node in nodes1.values():
        pd = dict(node.prepared_data)
        pd.pop('fingerprint', None)
        pd.pop('reused', None)
        node.prepared_data = pd
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    old_b_uid = nodes1['B'].uid
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['B'].uid == old_b_uid
    assert nodes2['B'].prepared_data.get('reused') is True

    # Now simulate legacy run whose input file mtime is newer than A.end_time -> must NOT reuse!
    for node in nodes2.values():
        pd = dict(node.prepared_data)
        pd.pop('fingerprint', None)
        node.prepared_data = pd
    nodes2['A'].end_time = utc_now() - timedelta(hours=1)
    future_ts = (utc_now() + timedelta(minutes=5)).timestamp()
    os.utime(r1_path, (future_ts, future_ts))
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes3 = _node_by_id(db_session, run_uid)
    assert nodes3['A'].prepared_data.get('reused') is False
    assert nodes3['B'].prepared_data.get('reused') is False
    assert nodes3['A'].status == NodeRunStatus.READY


def test_invalidation_causes_and_isolated_parallel_branch_preservation(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    tool_a = _create_tool(
        db_session,
        name='ToolA',
        inputs=[('read1', False)],
        outputs=[('out', '{output_dir}/a.txt', True)],
    )
    tool_b = _create_tool(
        db_session,
        name='ToolB',
        inputs=[('read2', False)],
        outputs=[('out', '{output_dir}/b.txt', True)],
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('S', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(tool_a.uid), 'modifiable_params': '-t 4'}),
                _node('B', 'tool', {'tool_uid': str(tool_b.uid), 'modifiable_params': '-t 4'}),
            ],
            'edges': [
                _edge('e1', 'S', 'r1', 'A', 'read1'),
                _edge('e2', 'S', 'r2', 'B', 'read2'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['S'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'out': Path(nodes1['A'].base_dir or '') / 'a.txt'})
    _mark_node_succeeded_with_files(db_session, nodes1['B'], {'out': Path(nodes1['B'].base_dir or '') / 'b.txt'})
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    old_b_uid = nodes1['B'].uid

    # 1) Change ToolA parameter -> A invalidates, B stays reused.
    updated_defn = wf.workflow.model_copy(deep=True)
    updated_defn.nodes[1].data['modifiable_params'] = '-t 8'
    wf.workflow = updated_defn
    db_session.commit()
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['A'].prepared_data.get('reused') is False
    assert nodes2['A'].status == NodeRunStatus.READY
    assert nodes2['B'].uid == old_b_uid
    assert nodes2['B'].prepared_data.get('reused') is True

    # Mark A succeeded again, then test: 2) Tool definition change & 3) deleted disk file.
    _mark_node_succeeded_with_files(db_session, nodes2['A'], {'out': Path(nodes2['A'].base_dir or '') / 'a.txt'})
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    tool_a.command_template = 'run_cmd --v2'
    (Path(nodes2['B'].base_dir or '') / 'b.txt').unlink()
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes3 = _node_by_id(db_session, run_uid)
    assert nodes3['S'].prepared_data.get('reused') is True
    assert nodes3['A'].prepared_data.get('reused') is False
    assert nodes3['B'].prepared_data.get('reused') is False

    # Mark A/B succeeded, then modify input resource file content -> S, A, B all invalidate.
    _mark_node_succeeded_with_files(db_session, nodes3['A'], {'out': Path(nodes3['A'].base_dir or '') / 'a.txt'})
    _mark_node_succeeded_with_files(db_session, nodes3['B'], {'out': Path(nodes3['B'].base_dir or '') / 'b.txt'})
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    r1_path.write_text('@read1_updated\nACGTACGT\n+\nIIIIIIII\n', encoding='utf-8')
    os.utime(r1_path, (r1_path.stat().st_atime + 2, r1_path.stat().st_mtime + 2))
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes4 = _node_by_id(db_session, run_uid)
    assert nodes4['S'].prepared_data.get('reused') is False
    assert nodes4['A'].prepared_data.get('reused') is False
    assert nodes4['B'].prepared_data.get('reused') is False


def test_parallel_branch_insertion_skips_pinned_indices(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    t1 = _create_tool(db_session, name='StepOne', inputs=[('r1', False)], outputs=[('out', '{output_dir}/1.txt', True)])
    t2 = _create_tool(
        db_session, name='StepTwo', inputs=[('inp', False)], outputs=[('out', '{output_dir}/2.txt', True)]
    )
    t3 = _create_tool(
        db_session, name='ParallelQC', inputs=[('inp', False)], outputs=[('out', '{output_dir}/qc.txt', True)]
    )

    defn1 = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(t1.uid)}),
                _node('B', 'tool', {'tool_uid': str(t2.uid)}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'A', 'r1'),
                _edge('e2', 'A', 'out', 'B', 'inp'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn1)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['R'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'out': Path(nodes1['A'].base_dir or '') / '1.txt'})
    _mark_node_succeeded_with_files(db_session, nodes1['B'], {'out': Path(nodes1['B'].base_dir or '') / '2.txt'})
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    # Insert parallel branch P after A while A (01_stepone) and B (02_steptwo) are both pinned.
    wf.workflow = WorkflowDefinition.model_validate(
        {
            'nodes': [
                *defn1.model_dump()['nodes'],
                _node('P', 'tool', {'tool_uid': str(t3.uid)}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'A', 'r1'),
                _edge('e2', 'A', 'out', 'B', 'inp'),
                _edge('e3', 'A', 'out', 'P', 'inp'),
            ],
        }
    )
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert Path(nodes2['A'].base_dir or '').name == '01_stepone'
    assert Path(nodes2['B'].base_dir or '').name == '02_steptwo'
    assert Path(nodes2['P'].base_dir or '').name == '03_parallelqc'
    assert nodes2['P'].status == NodeRunStatus.READY


def test_passthrough_taint_rules(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    sort_tool = _create_tool(
        db_session,
        name='samtools sort',
        tool_type=ToolType.DOCKER_PASSTHROUGH,
        inputs=[('bam_in', False)],
        outputs=[('bam_sorted', '{output_dir}/sorted.bam', True)],
    )
    index_tool = _create_tool(
        db_session,
        name='samtools index',
        tool_type=ToolType.DOCKER_PASSTHROUGH,
        inputs=[('bam', True)],
        outputs=[('bai', '{output_dir}/sorted.bam.bai', True)],
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('S', 'tool', {'tool_uid': str(sort_tool.uid)}),
                _node('I', 'tool', {'tool_uid': str(index_tool.uid)}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'S', 'bam_in'),
                _edge('e2', 'S', 'bam_sorted', 'I', 'bam'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['R'], {'r1': r1_path, 'r2': r2_path}, create_files=False)

    ws_dir = Path(nodes1['S'].base_dir or '')
    ws = ToolWorkspace(path=str(ws_dir), run_uid=str(run_uid), mounts={'/data/output': str(ws_dir)})
    _mark_node_succeeded_with_files(
        db_session,
        nodes1['S'],
        {'bam_sorted': ws_dir / 'sorted.bam'},
        workspace=ws,
    )
    _mark_node_succeeded_with_files(
        db_session,
        nodes1['I'],
        {'bam': ws_dir / 'sorted.bam', 'bai': ws_dir / 'sorted.bam.bai'},
        workspace=ws,
    )
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    # Case A: both succeeded & unchanged -> both reused.
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes_a = _node_by_id(db_session, run_uid)
    assert nodes_a['S'].prepared_data.get('reused') is True
    assert nodes_a['I'].prepared_data.get('reused') is True

    # Case B: I fails after starting (start_time set) -> I taints S's workspace, so BOTH S and I rerun.
    nodes_a['I'].status = NodeRunStatus.FAILED
    nodes_a['I'].start_time = utc_now()
    nodes_a['I'].end_time = utc_now()
    run.status = RunStatus.FAILED
    run.settled = True
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes_b = _node_by_id(db_session, run_uid)
    assert nodes_b['R'].prepared_data.get('reused') is True
    assert nodes_b['S'].prepared_data.get('reused') is False
    assert nodes_b['I'].prepared_data.get('reused') is False
    assert nodes_b['S'].status == NodeRunStatus.READY

    # Case C: S succeeded, I was BLOCKED (never started: start_time is None) -> S IS reused, I starts READY.
    ws_dir_c = Path(nodes_b['S'].base_dir or '')
    ws_c = ToolWorkspace(path=str(ws_dir_c), run_uid=str(run_uid), mounts={'/data/output': str(ws_dir_c)})
    _mark_node_succeeded_with_files(
        db_session,
        nodes_b['S'],
        {'bam_sorted': ws_dir_c / 'sorted.bam'},
        workspace=ws_c,
    )
    nodes_b['I'].status = NodeRunStatus.BLOCKED
    nodes_b['I'].start_time = None
    nodes_b['I'].end_time = utc_now()
    run.status = RunStatus.FAILED
    run.settled = True
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes_c = _node_by_id(db_session, run_uid)
    assert nodes_c['S'].prepared_data.get('reused') is True
    assert nodes_c['I'].prepared_data.get('reused') is False
    assert nodes_c['I'].status == NodeRunStatus.READY


def test_dynamic_marks_preserved_when_reused_and_deleted_when_removed(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    t1 = _create_tool(
        db_session, name='Aligner', inputs=[('r1', False)], outputs=[('out', '{output_dir}/out.bam', True)]
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(t1.uid)}),
                _node('M1', 'sample_mark', {'mark_name': 'aligned_bam', 'description': 'BAM'}),
                _node('M2', 'sample_mark', {'mark_name': 'old_tag', 'description': 'Old'}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'A', 'r1'),
                _edge('e2', 'A', 'out', 'M1', 'file'),
                _edge('e3', 'A', 'out', 'M2', 'file'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['R'], {'r1': r1_path, 'r2': r2_path}, create_files=False)
    bam_path = Path(nodes1['A'].base_dir or '') / 'out.bam'
    _mark_node_succeeded_with_files(db_session, nodes1['A'], {'out': bam_path})
    _mark_node_succeeded_with_files(db_session, nodes1['M1'], {'file_path': bam_path}, create_files=False)
    _mark_node_succeeded_with_files(db_session, nodes1['M2'], {'file_path': bam_path}, create_files=False)
    db_session.add_all(
        [
            SampleFiles(
                sample_uid=sample.uid,
                file_path=str(bam_path.absolute()),
                tag='aligned_bam',
                description='BAM',
                is_dynamic=True,
                producer_workflow_uid=wf.uid,
                producer_run_uid=run_uid,
            ),
            SampleFiles(
                sample_uid=sample.uid,
                file_path=str(bam_path.absolute()),
                tag='old_tag',
                description='Old',
                is_dynamic=True,
                producer_workflow_uid=wf.uid,
                producer_run_uid=run_uid,
            ),
        ]
    )
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    # Remove M2 from workflow, keep M1 unchanged.
    wf.workflow = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('A', 'tool', {'tool_uid': str(t1.uid)}),
                _node('M1', 'sample_mark', {'mark_name': 'aligned_bam', 'description': 'BAM'}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'A', 'r1'),
                _edge('e2', 'A', 'out', 'M1', 'file'),
            ],
        }
    )
    db_session.commit()

    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['M1'].prepared_data.get('reused') is True
    tags = set(db_session.scalars(select(SampleFiles.tag).where(SampleFiles.sample_uid == sample.uid)))
    assert 'aligned_bam' in tags
    assert 'old_tag' not in tags


def test_foreach_reuse_and_inner_subgraph_edit_invalidation(db_session: Session, tmp_path: Path) -> None:
    user, project, sample, r1_path, r2_path = _seed_user_project(db_session, tmp_path)
    inner_tool = _create_tool(
        db_session,
        name='InnerTool',
        inputs=[('in_file', False)],
        outputs=[('out_file', '{output_dir}/out.txt', True)],
    )
    child_graph = WorkflowDefinition(
        nodes=[
            WorkflowDefinition.model_validate(
                {'nodes': [_node('IN', 'tool', {'tool_uid': str(inner_tool.uid), 'modifiable_params': '-a'})]}
            ).nodes[0]
        ],
        edges=[],
        interface=WorkflowInterface(
            inputs=[InterfaceInput(id='p_in', name='in', targets=[PortTarget(node_id='IN', handle='in_file')])],
            outputs=[InterfaceOutput(id='p_out', name='out', source=PortTarget(node_id='IN', handle='out_file'))],
        ),
    )
    defn = WorkflowDefinition.model_validate(
        {
            'nodes': [
                _node('R', 'resource_sequence', {'r1': 'sample:r1', 'r2': 'sample:r2'}),
                _node('C', 'collect_file_collection', {'pattern': '*.txt', 'subfolder_prefix': 'item'}),
                _node('F', 'subgraph', {'name': 'Batch', 'workflow': child_graph.model_dump(mode='json')}),
            ],
            'edges': [
                _edge('e1', 'R', 'r1', 'C', 'folder_path'),
                _edge('e2', 'C', 'collection', 'F', 'p_in'),
            ],
        }
    )
    wf = _attach_workflow(db_session, project, user.id, defn)
    db_session.commit()
    assert sample is not None

    run_uid = workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    ).run_uids[0]
    nodes1 = _node_by_id(db_session, run_uid)
    _mark_node_succeeded_with_files(db_session, nodes1['R'], {'r1': r1_path, 'r2': r2_path}, create_files=False)

    coll_dir = Path(nodes1['C'].base_dir or '')
    coll_dir.mkdir(parents=True, exist_ok=True)
    item_file = tmp_path / 'item1.txt'
    item_file.write_text('item', encoding='utf-8')
    manifest_file = coll_dir / 'collection.json'
    manifest_file.write_text(
        json.dumps(
            {
                'version': 1,
                'items': [{'key': 'k1', 'name': 'item1.txt', 'relative_path': 'item1.txt', 'path': str(item_file)}],
            }
        ),
        encoding='utf-8',
    )
    _mark_node_succeeded_with_files(db_session, nodes1['C'], {'collection': manifest_file}, create_files=False)

    fe_dir = Path(nodes1['F'].base_dir or '')
    results_dir = fe_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'manifest.json').write_text(json.dumps({'version': 1, 'items': []}), encoding='utf-8')
    _mark_node_succeeded_with_files(db_session, nodes1['F'], {'results_folder': results_dir}, create_files=False)

    child_run = WorkflowRun(
        uid=random_uid(),
        generation=1,
        name='child',
        owner_id=user.id,
        project_id=project.id,
        parent_node_uid=nodes1['F'].uid,
        item_key='k1',
        item_index=0,
        item_name='item1.txt',
        base_dir=str(fe_dir / 'children' / 'item_1'),
        workflow=child_graph,
        status=RunStatus.SUCCEEDED,
        settled=True,
    )
    db_session.add(child_run)
    run = db_session.get(WorkflowRun, run_uid)
    assert run is not None
    run.status = RunStatus.SUCCEEDED
    run.settled = True
    db_session.commit()

    old_f_uid = nodes1['F'].uid
    child_run_uid = child_run.uid

    # Unchanged rerun: ForEach and child run preserved intact.
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes2 = _node_by_id(db_session, run_uid)
    assert nodes2['F'].uid == old_f_uid
    assert nodes2['F'].prepared_data.get('reused') is True
    assert db_session.get(WorkflowRun, child_run_uid) is not None

    # Modify inner node parameter -> ForEach invalidated, old dir & child run deleted.
    updated_fe_defn = wf.workflow.model_copy(deep=True)
    updated_fe_defn.nodes[2].data['workflow']['nodes'][0]['data']['modifiable_params'] = '-b'
    wf.workflow = updated_fe_defn
    db_session.commit()
    workflow_runs_service.create_project_runs(
        db_session,
        project_id=project.id,
        project_name=project.name,
        workflow=wf,
        samples=[sample],
        current_user=user,
        run_name_prefix=None,
    )
    nodes3 = _node_by_id(db_session, run_uid)
    assert nodes3['C'].prepared_data.get('reused') is True
    assert nodes3['F'].prepared_data.get('reused') is False
    assert nodes3['F'].status == NodeRunStatus.READY
    assert not results_dir.exists()
    db_session.expire_all()
    assert db_session.get(WorkflowRun, child_run_uid) is None
