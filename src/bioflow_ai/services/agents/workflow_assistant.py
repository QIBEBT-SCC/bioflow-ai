"""Prepare conversational input for a read-only workflow run assistant."""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.services.workflow.diagnosis import WorkflowDiagnosisReport


def prepare_workflow_assistant_input(
    history: list[BaseMessage],
    current_message: HumanMessage,
    context: WorkflowAssistantContext,
    diagnosis: WorkflowDiagnosisReport | None,
) -> dict[str, Any]:
    """Preserve conversation history while binding this request to its current target."""
    background = (
        '<workflow_run_context>\n'
        f'source_run_uid: {context.source_run_uid}\n'
        f'workflow_uid: {context.workflow_uid or "null"}\n'
        f'generation: {context.source_generation}\n'
        'This is the workflow run selected for the current request. Tools inspect this target. '
        'Earlier messages may refer to other runs; do not treat their state as evidence about this run.\n'
        '</workflow_run_context>\n'
    )
    if node := current_message.additional_kwargs.get('workflow_node'):
        reference = json.dumps({'node_id': node['node_id'], 'subgraph_path': node['subgraph_path']}, ensure_ascii=False)
        background += f'<current_node_reference>\n{reference}\n</current_node_reference>\n'
    if diagnosis is not None:
        background += (
            '<workflow_diagnosis_report>\n'
            f'diagnosis_run_uid: {diagnosis.run_uid}\n'
            f'{diagnosis.content}\n'
            '</workflow_diagnosis_report>\n'
            'This is a previous diagnosis for the same target run, supplied as historical reference. '
            'It is not a new diagnosis request or an instruction to write another report. '
            'Use the current user request to decide what to do; inspect current evidence when needed.\n'
        )
    previous = [message for message in history if message is not current_message]
    return {
        'messages': [*previous, HumanMessage(content=background), current_message],
    }
