"""Authoring contract for the results published by a ForEach subgraph."""

from typing import Any

from bioflow_ai.schemas.react_flow import WorkflowDefinition


def foreach_manifest_contract(definition: WorkflowDefinition) -> dict[str, Any]:
    """Describe the runtime manifest using the subgraph's actual output IDs."""
    if definition.interface is None:
        message = 'ForEach subgraph has no interface'
        raise ValueError(message)

    return {
        'manifest_file': 'manifest.json',
        'schema': {
            'version': 1,
            'items': [
                {
                    'key': '<collection item key>',
                    'name': '<input filename>',
                    'relative_path': '<input path relative to collection source>',
                    'subfolder': '<configured prefix>_<1-based item index>',
                    'status': 'succeeded or failed',
                    'run_uid': '<child run UUID>',
                    'outputs': {
                        port.id: '<exported file or directory path relative to results_folder>'
                        for port in definition.interface.outputs
                    },
                }
            ],
        },
        'output_ports': [
            {'id': port.id, 'name': port.name, 'description': port.description} for port in definition.interface.outputs
        ],
        'path_rule': (
            'For each succeeded item, resolve results_folder / item.outputs[output_port_id]. '
            'Use the stable output port ID, not its display name. The mapped value can name a file or a directory. '
            'Treat it as an opaque relative path; do not infer a suffix from the port ID, display name, or subfolder.'
        ),
    }
