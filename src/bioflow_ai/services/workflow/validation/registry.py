"""Extensible, runtime-independent node semantics."""

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bioflow_ai.schemas.tools import Handle
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.schemas.workflow_nodes import PROCESSORS, RESOURCES, FrozenPathSchema
from bioflow_ai.services.workflow.validation.types import Diagnostic, ValidationContext

PRESENTATION_NODE_TYPES = frozenset({'note', 'image_preview'})
CODE_NODE_TYPES = frozenset({'code_bash', 'code_python', 'code_R'})


class NodeSpec(BaseModel):
    """Parameters, ports and incoming-connection policy for one node type."""

    model_config = ConfigDict(frozen=True)

    input_handles: tuple[Handle, ...] = ()
    output_handles: tuple[Handle, ...] = ()
    args: type[BaseModel] | None = None
    input_policy: Literal['none', 'one', 'many', 'any'] = 'none'
    checks: tuple[
        Callable[[BuildNode, Mapping[str, BuildNode], list[BuildEdge], ValidationContext], list[Diagnostic]], ...
    ] = ()


NODE_SPECS: dict[str, NodeSpec] = {
    name: NodeSpec(
        input_handles=tuple(value['input_handles']),
        output_handles=tuple(value['output_handles']),
        args=value['args'],
        input_policy='one' if name == 'collect_file_collection' else 'many',
    )
    for name, value in PROCESSORS.items()
}
NODE_SPECS.update(
    {
        name if name == 'value_string' else f'resource_{name}': NodeSpec(
            input_handles=tuple(value['input_handles']),
            output_handles=tuple(value['output_handles']),
            args=value['args'],
            input_policy='any' if name == 'ncbi_genome' else 'none',
        )
        for name, value in RESOURCES.items()
    }
)
for name in CODE_NODE_TYPES | {'downstream_summary'}:
    NODE_SPECS[name] = NodeSpec(
        input_handles=(Handle(name='input_files', description='Bound CODE inputs'),),
        output_handles=()
        if name == 'downstream_summary'
        else (Handle(name='output_folder', description='CODE results'),),
        input_policy='many' if name == 'downstream_summary' else 'none',
    )
NODE_SPECS.update(
    {
        'tool': NodeSpec(input_policy='one'),
        'subgraph': NodeSpec(input_policy='one'),
        'foreach': NodeSpec(input_policy='one'),
        'resource_frozen_path': NodeSpec(
            output_handles=(Handle(name='file_path', description='Internal path'),), args=FrozenPathSchema
        ),
        'note': NodeSpec(),
        'image_preview': NodeSpec(input_handles=(Handle(name='image_path', description='Preview image'),)),
    }
)


def supported_node_types() -> frozenset[str]:
    """Return types supported by compilation without importing runtime executors."""
    return frozenset(NODE_SPECS) - PRESENTATION_NODE_TYPES - {'subgraph'}
