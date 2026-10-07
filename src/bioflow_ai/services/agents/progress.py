from typing import Any

from bioflow_ai.models.agent import AgentRun
from bioflow_ai.schemas.agent import AgentName
from bioflow_ai.utils.jsonable import to_json_object

SAMPLE_MANAGER_PROGRESS_KEYS = (
    'sample_list_project_samples',
    'sample_set_base_dir',
    'sample_run_command',
    'sample_add_samples',
    'sample_add_sample_file',
    'sample_add_project_file',
    'sample_write_conclusion',
    'sample_ask_human',
)

WORKFLOW_PHASE_KEY = 'workflow_phase'
WORKFLOW_MANAGER_ACTION_KEY = 'workflow_manager_action'


def event_payload(payload: object) -> dict[str, Any]:
    """Normalize LangGraph custom stream values into stable UI activities."""
    converted = to_json_object(payload)
    if isinstance(payload, str):
        return {'kind': 'info', 'message': payload}

    sample_progress_key = next((key for key in SAMPLE_MANAGER_PROGRESS_KEYS if key in converted), None)
    if sample_progress_key is not None:
        sample_progress = converted[sample_progress_key]
        if isinstance(sample_progress, dict):
            return {'kind': sample_progress_key, **sample_progress}

    workflow_phase = converted.get(WORKFLOW_PHASE_KEY)
    if isinstance(workflow_phase, dict):
        phase = workflow_phase.get('phase')
        return {
            'kind': WORKFLOW_PHASE_KEY,
            **workflow_phase,
            **({'tool_call_id': f'workflow-phase:{phase}'} if isinstance(phase, str) else {}),
        }

    workflow_action = converted.get(WORKFLOW_MANAGER_ACTION_KEY)
    if isinstance(workflow_action, dict):
        return {'kind': WORKFLOW_MANAGER_ACTION_KEY, **workflow_action}

    info = converted.get('info')
    if isinstance(info, str):
        return {'kind': 'info', 'message': info}

    todo = converted.get('todo')
    if isinstance(todo, dict):
        return {'kind': 'todo', **todo}

    if 'validate_node' in converted:
        return {
            'kind': 'validation',
            'message': 'Validating the generated tool',
            'detail': converted['validate_node'],
        }

    message = converted.get('message')
    if isinstance(message, str):
        return {'kind': 'info', 'message': message}

    key = next(iter(converted), 'activity')
    return {
        'kind': 'activity',
        'message': key.replace('_', ' ').strip().capitalize(),
        'detail': converted,
    }


def result_payload(run: AgentRun, values: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract UI artifacts and resource invalidations from a completed run."""
    result: dict[str, Any] = {}
    changes: list[dict[str, Any]] = []
    if run.agent_name == AgentName.WORKFLOW_BUILDER.value:
        workflow_uids = [str(value) for value in values.get('workflow_uids') or []]
        subgraph_uids = [str(value) for value in values.get('new_subgraphs') or []]
        tool_uids = [str(value) for value in values.get('new_tools') or []]
        database_ids = [int(value) for value in values.get('new_dbs') or []]
        result.update(
            workflow_uids=workflow_uids,
            subgraph_uids=subgraph_uids,
            tool_uids=tool_uids,
            database_ids=database_ids,
        )
        changes.extend({'resource': 'workflow', 'uid': uid, 'project_id': run.project_id} for uid in workflow_uids)
        changes.extend({'resource': 'workflow', 'uid': uid} for uid in subgraph_uids)
        changes.extend({'resource': 'tool', 'uid': uid} for uid in tool_uids)
        changes.extend({'resource': 'database', 'id': value} for value in database_ids)
        if values.get('samples_changed'):
            changes.extend(
                [
                    {'resource': 'sample', 'project_id': run.project_id},
                    {'resource': 'project_file_mapping', 'project_id': run.project_id},
                ]
            )
    elif run.agent_name == AgentName.TOOL_GENERATOR.value:
        if tool_uid := values.get('tool_uid'):
            uid = str(tool_uid)
            artifact = {
                'type': 'tool',
                'uid': uid,
                'name': values.get('name'),
                'description': values.get('description'),
            }
            result.update(tool_uid=uid, name=values.get('name'), artifacts=[artifact])
            changes.append({'resource': 'tool', 'uid': uid})
    elif run.agent_name == AgentName.SAMPLE_MANAGER.value:
        result['summary'] = values.get('summary')
        changes.extend(
            [
                {'resource': 'sample', 'project_id': run.project_id},
                {'resource': 'project_file_mapping', 'project_id': run.project_id},
            ]
        )
    elif run.agent_name == AgentName.WORKFLOW_ASSISTANT.value:
        result.update(
            diagnosis_path=values.get('diagnosis_path'),
            source_run_uid=values.get('source_run_uid'),
            workflow_uid=values.get('workflow_uid'),
        )
    elif run.agent_name == AgentName.WORKFLOW_FIXER.value:
        workflow_uid = values.get('target_workflow_uid')
        changed = bool(values.get('workflow_changed'))
        result.update(
            workflow_uid=workflow_uid,
            source_run_uid=values.get('source_run_uid'),
            workflow_changed=changed,
            diagnosis_run_uid=values.get('diagnosis_run_uid'),
            update_path=values.get('update_path'),
            changed_workflow_uids=values.get('changed_workflow_uids') or [],
        )
        if changed and workflow_uid:
            changes.extend(
                {'resource': 'workflow', 'uid': str(uid), 'project_id': run.project_id}
                for uid in dict.fromkeys(values.get('changed_workflow_uids') or [workflow_uid])
            )
    return result, changes
