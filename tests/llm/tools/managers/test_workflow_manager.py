import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.schemas.tools import Handle
from llm.states import BuildNode
from llm.workflow_builder.drafts import build_workflow_draft

MODULE_PATH = Path(__file__).parents[4] / 'llm/tools/managers/workflow_manager.py'
SPEC = importlib.util.spec_from_file_location('workflow_manager_under_test', MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
WORKFLOW_MANAGER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKFLOW_MANAGER
SPEC.loader.exec_module(WORKFLOW_MANAGER)

_create_workflow_draft = WORKFLOW_MANAGER.__dict__['_create_workflow_draft']
_delete_workflow_draft = WORKFLOW_MANAGER.__dict__['_delete_workflow_draft']
_save_all_workflows = WORKFLOW_MANAGER.__dict__['_save_all_workflows']
_update_workflow_metadata = WORKFLOW_MANAGER.__dict__['_update_workflow_metadata']


def _runtime(state: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(state=state, tool_call_id='test-call', context=SimpleNamespace(user_id=2))


def _node() -> BuildNode:
    return BuildNode(
        id='A',
        name='Input',
        type='resource_file',
        input_handles=[],
        output_handles=[Handle(name='file_path', description='Resolved path')],
        x=0,
        y=0,
    )


def _message_content(command: Any) -> str:
    return command.update['messages'][0].content


def test_create_workflow_draft_rejects_normalized_duplicate_name() -> None:
    existing = build_workflow_draft(
        draft_id='wf_A',
        name='Sample Alignment',
        description='Existing workflow',
        execution_scope='sample_level',
    )
    runtime = _runtime(
        {
            'active_workflow_id': 'wf_A',
            'workflow_drafts': {'wf_A': existing},
            'exist_nodes': [],
            'exist_edges': [],
        }
    )

    command = _create_workflow_draft('  sample   alignment ', 'Duplicate', 'sample_level', runtime)

    assert set(command.update['workflow_drafts']) == {'wf_A'}
    assert 'already exists as wf_A' in _message_content(command)
    assert 'active_workflow_id' not in command.update


def test_delete_active_workflow_draft_clears_panel() -> None:
    active = build_workflow_draft(
        draft_id='wf_A',
        name='Mistaken Draft',
        description='Created accidentally',
        execution_scope='sample_level',
        nodes=[_node()],
    )
    runtime = _runtime(
        {
            'active_workflow_id': 'wf_A',
            'workflow_drafts': {'wf_A': active},
            'exist_nodes': [_node()],
            'exist_edges': [],
        }
    )

    command = _delete_workflow_draft('wf_A', runtime)

    assert command.update['workflow_drafts'] == {}
    assert command.update['active_workflow_id'] is None
    assert command.update['exist_nodes'].value == []
    assert command.update['exist_edges'].value == []


def test_delete_inactive_workflow_draft_preserves_active_panel() -> None:
    active = build_workflow_draft(
        draft_id='wf_A',
        name='Intended Draft',
        description='Keep this draft',
        execution_scope='sample_level',
        nodes=[_node()],
    )
    extra = build_workflow_draft(
        draft_id='wf_B',
        name='Extra Draft',
        description='Delete this draft',
        execution_scope='sample_level',
    )
    runtime = _runtime(
        {
            'active_workflow_id': 'wf_A',
            'workflow_drafts': {'wf_A': active, 'wf_B': extra},
            'exist_nodes': [_node()],
            'exist_edges': [],
        }
    )

    command = _delete_workflow_draft('wf_B', runtime)

    assert set(command.update['workflow_drafts']) == {'wf_A'}
    assert 'active_workflow_id' not in command.update
    assert 'exist_nodes' not in command.update


def test_update_workflow_metadata_rejects_duplicate_name() -> None:
    first = build_workflow_draft(
        draft_id='wf_A',
        name='Alignment',
        description='First workflow',
        execution_scope='sample_level',
    )
    second = build_workflow_draft(
        draft_id='wf_B',
        name='Calling',
        description='Second workflow',
        execution_scope='sample_level',
    )
    runtime = _runtime(
        {
            'active_workflow_id': 'wf_B',
            'workflow_drafts': {'wf_A': first, 'wf_B': second},
            'exist_nodes': [],
            'exist_edges': [],
        }
    )

    command = _update_workflow_metadata('ALIGNMENT', None, None, runtime)

    assert command.update['workflow_drafts']['wf_B'].name == 'Calling'
    assert 'already used by wf_A' in _message_content(command)


def test_save_all_workflows_rejects_duplicate_names_before_persistence() -> None:
    first = build_workflow_draft(
        draft_id='wf_A',
        name='Alignment',
        description='First workflow',
        execution_scope='sample_level',
        nodes=[_node()],
    )
    second = build_workflow_draft(
        draft_id='wf_B',
        name=' alignment ',
        description='Duplicate workflow',
        execution_scope='sample_level',
        nodes=[_node()],
    )
    runtime = _runtime(
        {
            'active_workflow_id': None,
            'workflow_drafts': {'wf_A': first, 'wf_B': second},
            'exist_nodes': [],
            'exist_edges': [],
        }
    )

    command = asyncio.run(_save_all_workflows(runtime))

    assert 'duplicate workflow drafts exist: wf_A, wf_B' in _message_content(command)
    assert 'workflow_uids' not in command.update
