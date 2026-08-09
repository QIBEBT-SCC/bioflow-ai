from typing import Any

from app.models.agent import AgentRun
from app.schemas.agent import AgentName
from app.utils.jsonable import to_json_object


def event_payload(payload: object) -> dict[str, Any]:
    """Normalize LangGraph custom stream values into stable UI activities."""
    converted = to_json_object(payload)
    if isinstance(payload, str):
        return {'kind': 'info', 'message': payload}

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
        tool_uids = [str(value) for value in values.get('new_tools') or []]
        database_ids = [int(value) for value in values.get('new_dbs') or []]
        result.update(workflow_uids=workflow_uids, tool_uids=tool_uids, database_ids=database_ids)
        changes.extend({'resource': 'workflow', 'uid': uid, 'project_id': run.project_id} for uid in workflow_uids)
        changes.extend({'resource': 'tool', 'uid': uid} for uid in tool_uids)
        changes.extend({'resource': 'database', 'id': value} for value in database_ids)
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
    return result, changes
