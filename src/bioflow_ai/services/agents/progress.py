from collections.abc import Sequence
from typing import Any

from bioflow_ai.models.agent import AgentRun
from bioflow_ai.schemas.agent import AgentName
from bioflow_ai.utils.jsonable import to_json_object

PROJECT_PLANNER_PROGRESS_KEYS = (
    'planner_list_project_samples',
    'planner_set_base_dir',
    'planner_run_command',
    'planner_add_samples',
    'planner_add_sample_file',
    'planner_add_project_file',
    'planner_ask_human',
    'planner_write_file',
    'planner_web_run',
    'planner_view_pdf',
    'planner_confirm_plan',
)

WORKFLOW_MANAGER_ACTION_KEY = 'workflow_manager_action'
TOOL_ACTIVITY_KEYS = (WORKFLOW_MANAGER_ACTION_KEY, 'agent_tool', 'tool_generator_step')
MODEL_USAGE_KEY = 'model_usage'
TASK_NAMESPACE_KEY = 'task_ns'
NAMESPACE_SEPARATOR = '|'


class ProgressScopes:
    """Attribute stream values to the displayed step whose task invoked the emitting sub-agent.

    Step activities carry the checkpoint namespace of their task. A graph invoked inside that task streams with the
    same namespace split on ``|``, so its values nest under the step. Values outside every step belong to the main
    agent, whatever the depth of the graph that runs it.
    """

    def __init__(self) -> None:
        self._steps: dict[str, tuple[str, int]] = {}

    def scope(self, namespace: Sequence[str]) -> dict[str, Any]:
        """Return the nesting depth and parent step id of a value streamed from ``namespace``."""
        for end in range(len(namespace), 0, -1):
            step = self._steps.get(NAMESPACE_SEPARATOR.join(namespace[:end]))
            if step is not None:
                parent_id, depth = step
                return {'depth': depth + 1, 'parent_id': parent_id}
        return {'depth': 0, 'parent_id': None}

    def progress(self, payload: object, namespace: Sequence[str]) -> dict[str, Any]:
        """Normalize one progress value with its scope, registering the step it opens for sub-agents."""
        task_namespace = payload.get(TASK_NAMESPACE_KEY) if isinstance(payload, dict) else None
        if isinstance(payload, dict) and TASK_NAMESPACE_KEY in payload:
            payload = {key: value for key, value in payload.items() if key != TASK_NAMESPACE_KEY}
        event = event_payload(payload)
        scope = self.scope(namespace)
        step_id = event.get('tool_call_id')
        if isinstance(task_namespace, str) and isinstance(step_id, str):
            self._steps[task_namespace] = (step_id, scope['depth'])
        return {**event, 'scope': scope}

    def usage(self, payload: object, namespace: Sequence[str]) -> dict[str, Any] | None:
        """Return the model-call usage of a stream value with its scope, if it carries any."""
        usage = usage_payload(payload)
        return None if usage is None else {**usage, 'scope': self.scope(namespace)}


def usage_payload(payload: object) -> dict[str, Any] | None:
    """Return the model-call usage carried by a custom stream value, if any."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get(MODEL_USAGE_KEY)
    return to_json_object(usage) if isinstance(usage, dict) else None


def event_payload(payload: object) -> dict[str, Any]:
    """Normalize LangGraph custom stream values into stable UI activities."""
    converted = to_json_object(payload)
    if isinstance(payload, str):
        return {'kind': 'info', 'message': payload}

    planner_progress_key = next((key for key in PROJECT_PLANNER_PROGRESS_KEYS if key in converted), None)
    if planner_progress_key is not None:
        planner_progress = converted[planner_progress_key]
        if isinstance(planner_progress, dict):
            return {'kind': planner_progress_key, **planner_progress}

    tool_activity_key = next((key for key in TOOL_ACTIVITY_KEYS if key in converted), None)
    if tool_activity_key is not None:
        tool_activity = converted[tool_activity_key]
        if isinstance(tool_activity, dict):
            return {'kind': tool_activity_key, **tool_activity}

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
    elif run.agent_name == AgentName.PROJECT_PLANNER.value:
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
