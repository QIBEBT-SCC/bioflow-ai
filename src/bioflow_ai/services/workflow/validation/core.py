"""Pure workflow analysis over dictionaries and explicit tool/context snapshots."""

import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from bioflow_ai.schemas.react_flow import Edge, InterfaceInput, NodeBase, WorkflowDefinition, WorkflowInterface
from bioflow_ai.schemas.tool_config import ToolType
from bioflow_ai.schemas.tools import Handle, WorkflowToolDefinition
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.schemas.workflow_config import WorkflowType
from bioflow_ai.services.workflow.batch_graph import lower_batch_subgraphs
from bioflow_ai.services.workflow.subgraphs import MAX_SUBGRAPH_DEPTH, definition_id
from bioflow_ai.services.workflow.validation._expansion import _flatten_workflow
from bioflow_ai.services.workflow.validation._rules import (
    _check_code_binding_sources,
    _check_code_node_edges,
    _check_edge_handles,
    _check_ncbi_genome_edges,
    _check_processor_edges,
    _check_tool_node_edges,
    _validate_node_context,
    _validate_note_anchor,
)
from bioflow_ai.services.workflow.validation._topology import order_nodes
from bioflow_ai.services.workflow.validation.adapters import copy_graph
from bioflow_ai.services.workflow.validation.passthrough import passthrough_diagnostics, tool_mount_diagnostics
from bioflow_ai.services.workflow.validation.registry import CODE_NODE_TYPES, NODE_SPECS, PRESENTATION_NODE_TYPES
from bioflow_ai.services.workflow.validation.types import (
    Diagnostic,
    SourceLocation,
    ValidationContext,
    ValidationResult,
)

# Outputs whose runtime value is a `mountlist:` manifest that only Tool execution can expand.
MOUNTLIST_OUTPUTS = {
    'resource_sample_mark_collection': 'mount_collection',
    'collect_mount_dirs': 'mount_dir_list',
}


def pointer(value: Any) -> str:
    """Escape one JSON pointer component."""
    return str(value).replace('~', '~0').replace('/', '~1')


def _input_error(policy: str, edges: list[BuildEdge], handles: list[Handle]) -> str | None:
    if policy == 'one':
        return _check_tool_node_edges(edges, handles)
    if policy == 'any':
        return _check_ncbi_genome_edges(edges, handles)
    return _check_processor_edges(edges, handles)


def run_rules(
    graph: Any,
    *,
    context: ValidationContext | None = None,
    catalog: Mapping[str, WorkflowToolDefinition] | None = None,
) -> ValidationResult:
    """Report independent issues without querying services or changing the graph."""
    context = context or ValidationContext()
    result = ValidationResult()
    raw = copy_graph(graph)
    origins: dict[str, tuple[str, str, list[str]]] = {}
    edge_origins: dict[str, tuple[str, str, list[str]]] = {}
    tool_nodes = {}
    parse_failed = False

    def report(
        code: str,
        message: str,
        path: str = '',
        node_id: str | None = None,
        edge_id: str | None = None,
        handle: str | None = None,
        parents: tuple[str, ...] = (),
    ) -> None:
        result.diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                path=path,
                node_id=node_id,
                edge_id=edge_id,
                handle=handle,
                subgraph_path=list(parents),
            )
        )

    def invalid(
        error: ValidationError,
        path: str,
        parents: tuple[str, ...] = (),
        node_id: str | None = None,
    ) -> None:
        nonlocal parse_failed
        parse_failed = True
        for item in error.errors():
            suffix = ''.join('/' + pointer(part) for part in item['loc'])
            report('graph.parse', item['msg'], path + suffix, node_id=node_id, parents=parents)

    def visit(
        value: Any,
        path: str = '',
        parents: tuple[str, ...] = (),
        *,
        foreach: bool = False,
    ) -> dict[str, Any] | None:
        nonlocal parse_failed
        if len(parents) > MAX_SUBGRAPH_DEPTH:
            report('subgraph.depth', 'Subgraph nesting exceeds 32 levels', path, parents=parents)
            parse_failed = True
            return None
        if not isinstance(value, dict):
            report('graph.parse', 'Workflow must be an object', path, parents=parents)
            parse_failed = True
            return None
        for key in ('nodes', 'edges'):
            if not isinstance(value.get(key, []), list):
                report('graph.parse', f'{key} must be a list', f'{path}/{key}', parents=parents)
                parse_failed = True
                return None
        if value.get('wf_type') in (WorkflowType.SUBMODULE, 'submodule') and value.get('interface') is None:
            report(
                'subgraph.interface', 'Subgraphs require an interface definition', path + '/interface', parents=parents
            )
        nodes: list[BuildNode] = []
        paths: dict[str, str] = {}
        normalized_nodes = []
        raw_types = {
            n.get('id'): n.get('type')
            for n in value.get('nodes', [])
            if isinstance(n, dict) and isinstance(n.get('id'), str)
        }
        batch_targets = {
            edge.get('target')
            for edge in value.get('edges', [])
            if isinstance(edge, dict)
            and isinstance(edge.get('source'), str)
            and raw_types.get(edge['source']) == 'collect_file_collection'
            and isinstance(edge.get('target'), str)
        }
        if not value.get('nodes', []):
            report('graph.empty', 'Workflow needs at least one node', path + '/nodes', parents=parents)
        batch_input_counts = Counter(
            edge.get('target')
            for edge in value.get('edges', [])
            if isinstance(edge, dict)
            and isinstance(edge.get('source'), str)
            and raw_types.get(edge['source']) == 'collect_file_collection'
            and isinstance(edge.get('target'), str)
        )
        for index, item in enumerate(value.get('nodes', [])):
            location = f'{path}/nodes/{index}'
            try:
                node = NodeBase.model_validate(item)
            except ValidationError as error:
                invalid(error, location, parents)
                continue
            if not node.id:
                report('node.id', 'Node IDs must not be empty', location + '/id', node_id=node.id, parents=parents)
            if node.id in paths:
                report(
                    'node.duplicate',
                    f'Duplicate node ID: {node.id}',
                    location + '/id',
                    node_id=node.id,
                    parents=parents,
                )
            paths[node.id] = location
            origins[definition_id((*parents, node.id))] = (location, node.id, list(parents))
            spec = NODE_SPECS.get(node.type)
            inputs, outputs = [], []
            data = node.data
            if node.type == 'subgraph_inputs' and 'output_handles' in item:
                try:
                    outputs = [Handle.model_validate(handle) for handle in item['output_handles']]
                except (ValidationError, TypeError):
                    report('node.ports', 'Invalid editing input handles', location, node_id=node.id, parents=parents)
                    parse_failed = True
                normalized_nodes.append({**node.model_dump(), 'dragHandle': '', 'zIndex': 0})
                nodes.append(
                    BuildNode(
                        id=node.id, name=node.type, type=node.type, data=data, input_handles=[], output_handles=outputs
                    )
                )
                continue
            if spec is None:
                report(
                    'node.type',
                    f'Unsupported node type {node.type!r}',
                    location + '/type',
                    node_id=node.id,
                    parents=parents,
                )
            else:
                inputs, outputs = list(spec.input_handles), list(spec.output_handles)
                if spec.args is not None:
                    try:
                        spec.args.model_validate(data)
                    except ValidationError as error:
                        invalid(error, location + '/data', parents, node.id)
            if node.type == 'tool':
                try:
                    uid = str(uuid.UUID(str(data.get('tool_uid', ''))))
                except ValueError:
                    report(
                        'tool.uid',
                        'tool_uid must be a UUID',
                        location + '/data/tool_uid',
                        node_id=node.id,
                        parents=parents,
                    )
                else:
                    if catalog is None:
                        report(
                            'tool.definition',
                            'Tool definition is required',
                            location + '/data/tool_uid',
                            node_id=node.id,
                            parents=parents,
                        )
                    elif uid not in catalog:
                        report(
                            'tool.missing',
                            f'Tool {uid} not found',
                            location + '/data/tool_uid',
                            node_id=node.id,
                            parents=parents,
                        )
                    else:
                        for diagnostic in tool_mount_diagnostics(catalog[uid].file_mounts, catalog[uid].tool_type):
                            report(
                                diagnostic.code,
                                diagnostic.message,
                                location + '/data/tool_uid',
                                node_id=node.id,
                                handle=diagnostic.handle,
                                parents=parents,
                            )
                        if uid not in tool_nodes:
                            tool_nodes[uid] = catalog[uid].to_node()
                        tool = tool_nodes[uid]
                        inputs, outputs = tool.input_handles, tool.output_handles
                        if foreach and catalog[uid].tool_type == ToolType.DOCKER_PASSTHROUGH:
                            report(
                                'passthrough.foreach',
                                'Pass-through is unavailable inside ForEach',
                                location,
                                node_id=node.id,
                                parents=parents,
                            )
            if node.type in {'subgraph', 'foreach'}:
                if data.get('execution_scope', context.execution_scope) != context.execution_scope:
                    report(
                        'subgraph.scope',
                        'Subgraph execution scope must match its parent workflow',
                        location + '/data/execution_scope',
                        node_id=node.id,
                        parents=parents,
                    )
                is_batch = node.type == 'foreach' or node.id in batch_targets
                child = visit(
                    data.get('workflow'), location + '/data/workflow', (*parents, node.id), foreach=foreach or is_batch
                )
                iface = child.get('interface') if child else None
                if is_batch and batch_input_counts[node.id] > 1:
                    report(
                        'foreach.inputs',
                        'Only one collection input is supported',
                        location,
                        node_id=node.id,
                        parents=parents,
                    )
                if iface is None:
                    report(
                        'subgraph.interface',
                        'Subgraph has no interface definition',
                        location + '/data/workflow/interface',
                        node_id=node.id,
                        parents=parents,
                    )
                else:
                    inputs = [
                        Handle(name=port['id'], description=port.get('description', '')) for port in iface['inputs']
                    ]
                    outputs = [
                        Handle(name=port['id'], description=port.get('description', '')) for port in iface['outputs']
                    ]
                    if is_batch:
                        consumed = {
                            e.get('targetHandle', '').removeprefix(f'{node.id}-in-')
                            for e in value.get('edges', [])
                            if isinstance(e, dict)
                            and e.get('target') == node.id
                            and isinstance(e.get('source'), str)
                            and raw_types.get(e['source']) == 'collect_file_collection'
                            and isinstance(e.get('targetHandle'), str)
                        }
                        if any(port['id'] == 'collection' and port['id'] not in consumed for port in iface['inputs']):
                            report(
                                'foreach.reserved_port',
                                'collection is reserved for the batch input',
                                location,
                                node_id=node.id,
                                parents=parents,
                            )
                        if not iface['outputs']:
                            report(
                                'foreach.outputs',
                                'ForEach child needs at least one output',
                                location,
                                node_id=node.id,
                                parents=parents,
                            )
                        outputs = [Handle(name='results_folder', description='ForEach results')]
                    if node.type == 'foreach':
                        inputs = [
                            Handle(name='collection', description='Collection'),
                            *[h for h in inputs if h.name != data.get('item_port_id')],
                        ]
                if child:
                    data = {**data, 'workflow': child}
            if foreach and node.type in {'foreach', 'collect_file_collection', 'sample_mark', 'project_mark'}:
                report(
                    'foreach.node',
                    f'{node.type} is unavailable inside ForEach',
                    location + '/type',
                    node_id=node.id,
                    parents=parents,
                )
            if node.type == 'foreach':
                report(
                    'foreach.internal',
                    'Use a collection connected to a subgraph instead of an embedded ForEach node',
                    location,
                    node_id=node.id,
                    parents=parents,
                )
            if node.type == 'resource_frozen_path' and not node.id.startswith('@foreach/input/'):
                report(
                    'node.internal',
                    'Fixed-path sources are reserved for ForEach child runs',
                    location + '/type',
                    node_id=node.id,
                    parents=parents,
                )
            nodes.append(
                BuildNode(
                    id=node.id, name=node.type, type=node.type, data=data, input_handles=inputs, output_handles=outputs
                )
            )
            normalized_nodes.append(
                {
                    **node.model_dump(),
                    'data': data,
                    'dragHandle': item.get('dragHandle', ''),
                    'zIndex': item.get('zIndex', 0),
                }
            )
        by_id = {node.id: node for node in nodes}
        mark_nodes = defaultdict(list)
        for node in nodes:
            if node.type in {'project_mark', 'sample_mark'}:
                mark_nodes[str(node.data.get('mark_name', ''))].append(node)
        incoming = defaultdict(list)
        outgoing = Counter()
        bindings = Counter()
        normalized_edges = []
        build_edges = []
        dependencies = []
        for index, item in enumerate(value.get('edges', [])):
            location = f'{path}/edges/{index}'
            try:
                edge = Edge.model_validate(item)
            except ValidationError as error:
                invalid(error, location, parents)
                continue
            edge_origins[edge.id] = (location, edge.id, list(parents))
            normalized_edges.append(edge.model_dump())
            source, target = by_id.get(edge.source), by_id.get(edge.target)
            if source is None or target is None:
                report('edge.node', 'Source or target node does not exist', location, edge_id=edge.id, parents=parents)
                continue
            if source.type not in PRESENTATION_NODE_TYPES and target.type not in PRESENTATION_NODE_TYPES:
                dependencies.append((source.id, target.id))
            source_prefix, target_prefix = f'{source.id}-out-', f'{target.id}-in-'
            if not edge.sourceHandle.startswith(source_prefix) or not edge.targetHandle.startswith(target_prefix):
                report('edge.handle_format', 'Invalid handle prefix', location, edge_id=edge.id, parents=parents)
                continue
            binding = BuildEdge(
                source_node=source.id,
                source_handle=edge.sourceHandle.removeprefix(source_prefix),
                target_node=target.id,
                target_handle=edge.targetHandle.removeprefix(target_prefix),
            )
            if (
                not (source.type == 'tool' and catalog is None)
                and not (target.type == 'tool' and catalog is None)
                and (error := _check_edge_handles([binding], target.input_handles, [source]))
            ):
                source_invalid = binding.source_handle not in {handle.name for handle in source.output_handles}
                report(
                    'edge.port',
                    error,
                    location,
                    node_id=source.id if source_invalid else target.id,
                    edge_id=edge.id,
                    handle=binding.source_handle if source_invalid else binding.target_handle,
                    parents=parents,
                )
            incoming[target.id].append(binding)
            build_edges.append(binding)
            outgoing[source.id] += 1
            key = (source.id, binding.source_handle, target.id, binding.target_handle)
            bindings[key] += 1
            if bindings[key] > 1:
                report('edge.duplicate', 'Duplicate parameter binding', location, edge_id=edge.id, parents=parents)
            if source.type in PRESENTATION_NODE_TYPES or target.type == 'note':
                report(
                    'edge.presentation',
                    'Invalid presentation node connection',
                    location,
                    edge_id=edge.id,
                    parents=parents,
                )
            if source.type == 'collect_file_collection' and (
                binding.source_handle != 'collection' or target.type != 'subgraph'
            ):
                report(
                    'collection.target',
                    'Collection must connect to a subgraph input',
                    location,
                    edge_id=edge.id,
                    parents=parents,
                )
            if MOUNTLIST_OUTPUTS.get(source.type) == binding.source_handle and target.type not in {'tool', 'subgraph'}:
                report(
                    'mountlist.target',
                    f'`{binding.source_handle}` is a virtual folder that only Tool nodes can mount. Connect it '
                    'directly to a Tool folder input instead of a CODE, processor, or resource node.',
                    location,
                    edge_id=edge.id,
                    parents=parents,
                )
        interface = None
        boundary_inputs = defaultdict(list)
        if value.get('interface') is not None:
            try:
                interface = WorkflowInterface.model_validate(value['interface'])
            except ValidationError as error:
                invalid(error, path + '/interface', parents)
            else:
                for direction, ports in [('input', interface.inputs), ('output', interface.outputs)]:
                    for index, port in enumerate(ports):
                        targets = port.targets if isinstance(port, InterfaceInput) else [port.source]
                        for ti, target in enumerate(targets):
                            loc = f'{path}/interface/{direction}s/{index}/' + (
                                f'targets/{ti}' if direction == 'input' else 'source'
                            )
                            node = by_id.get(target.node_id)
                            handles = getattr(node, f'{direction}_handles', [])
                            if node is not None and node.type == 'tool' and catalog is None:
                                pass
                            elif node is None or target.handle not in {h.name for h in handles}:
                                report(
                                    'interface.port',
                                    f'Unknown exposed {direction} handle {target.node_id}.{target.handle}',
                                    loc,
                                    node_id=target.node_id,
                                    handle=target.handle,
                                    parents=parents,
                                )
                            elif direction == 'output' and node.type == 'collect_file_collection':
                                report(
                                    'collection.interface',
                                    'Expose the connected batch results_folder instead of a collection output',
                                    loc,
                                    node_id=node.id,
                                    parents=parents,
                                )
                            elif node.type == 'note' or (
                                direction == 'output' and node.type in PRESENTATION_NODE_TYPES
                            ):
                                report(
                                    'interface.presentation',
                                    'Presentation nodes cannot carry this interface binding',
                                    loc,
                                    node_id=node.id,
                                    parents=parents,
                                )
                            if direction == 'input':
                                boundary_inputs[target.node_id].append(
                                    BuildEdge(
                                        source_node='@subgraph_inputs',
                                        source_handle=port.id,
                                        target_node=target.node_id,
                                        target_handle=target.handle,
                                    )
                                )
        for node in nodes:
            location = paths[node.id]
            spec = NODE_SPECS.get(node.type)
            if spec:
                for check in spec.checks:
                    for item in check(node, by_id, build_edges, context):
                        result.diagnostics.append(
                            item.model_copy(
                                update={
                                    'path': item.path or location,
                                    'node_id': item.node_id if item.node_id is not None else node.id,
                                    'subgraph_path': list(parents),
                                }
                            )
                        )
            edges = [*incoming[node.id], *boundary_inputs[node.id]]
            if (
                spec
                and spec.input_policy != 'none'
                and not (node.type == 'tool' and catalog is None)
                and (error := _input_error(spec.input_policy, edges, node.input_handles))
            ):
                multiple = (
                    any(count > 1 for count in Counter(e.target_handle for e in edges).values())
                    and spec.input_policy == 'one'
                )
                report(
                    'input.multiple' if multiple else 'input.missing',
                    error,
                    location,
                    node_id=node.id,
                    parents=parents,
                )
            if node.type in CODE_NODE_TYPES:
                if error := _check_code_node_edges(
                    incoming[node.id], [by_id[e.source_node] for e in incoming[node.id] if e.source_node in by_id]
                ):
                    report('code.source', error, location, node_id=node.id, parents=parents)
                # Boundary inputs feeding CODE directly are equally invalid.
                if boundary_inputs[node.id]:
                    report(
                        'code.source',
                        'CODE inputs must come from bind_param nodes',
                        location,
                        node_id=node.id,
                        parents=parents,
                    )
                node_types = {nid: item.type for nid, item in by_id.items()}
                bind_params = {
                    nid: str(item.data.get('parameter', '')).strip()
                    for nid, item in by_id.items()
                    if item.type == 'bind_param'
                }
                for error in _check_code_binding_sources(
                    incoming[node.id], node_types, incoming, bind_parameters=bind_params
                ):
                    report('code.source', error, location, node_id=node.id, parents=parents)
            if node.type == 'image_preview' and len(edges) > 1:
                report(
                    'input.multiple',
                    'Image preview accepts one image_path input',
                    location,
                    node_id=node.id,
                    parents=parents,
                )
            if node.type == 'collect_file_collection' and outgoing[node.id] > 1:
                report(
                    'collection.output',
                    'Connect collection to exactly one subgraph input',
                    location,
                    node_id=node.id,
                    parents=parents,
                )
            if node.type == 'note' and (error := _validate_note_anchor(node.data.get('anchor_node_id'), nodes)):
                report('note.anchor', error, location + '/data/anchor_node_id', node_id=node.id, parents=parents)
            peers = (
                [n for n in mark_nodes[str(node.data.get('mark_name', ''))] if n is not node]
                if node.type in {'project_mark', 'sample_mark'}
                else []
            )
            if error := _validate_node_context(
                node.type,
                node.data,
                peers,
                scope=context.execution_scope,
                purpose='foreach' if foreach else context.subgraph_purpose,
            ):
                report('node.context', error, location, node_id=node.id, parents=parents)
            if node.type in {'project_mark', 'sample_mark'} and (
                error := context.mark_conflicts.get(f'{node.type}:{node.data.get("mark_name", "")}')
            ):
                report('mark.storage', error, location + '/data/mark_name', node_id=node.id, parents=parents)
        try:
            order_nodes([node.id for node in by_id.values() if node.type not in PRESENTATION_NODE_TYPES], dependencies)
        except ValueError as error:
            report('graph.cycle', str(error), path, parents=parents)
        return {
            'nodes': normalized_nodes,
            'edges': normalized_edges,
            'interface': interface.model_dump(mode='json') if interface else None,
        }

    normalized = visit(raw, foreach=context.subgraph_purpose == 'foreach')
    if (
        normalized is not None
        and not parse_failed
        and not any(n.get('type') == 'subgraph_inputs' for n in normalized['nodes'])
    ):
        try:
            source_map: dict[str, SourceLocation] = {}
            transformed = lower_batch_subgraphs(WorkflowDefinition.model_validate(normalized))
            flattened = _flatten_workflow(transformed, source_map=source_map)
            for diagnostic in passthrough_diagnostics(
                flattened, dict(catalog or {}), dynamic_marks=context.dynamic_marks
            ):
                origin = source_map.get('node:' + str(diagnostic.node_id))
                if origin:
                    diagnostic.path = origin.path
                    diagnostic.node_id = origin.node_id
                    diagnostic.subgraph_path = origin.subgraph_path
                result.diagnostics.append(diagnostic)
            flat_types = {item.id: item.type for item in flattened.nodes}
            flat_bind_params = {
                item.id: str(item.data.get('parameter', '')).strip()
                for item in flattened.nodes
                if item.type == 'bind_param'
            }
            flat_incoming: defaultdict[str, list[BuildEdge]] = defaultdict(list)
            for edge in flattened.edges:
                flat_incoming[edge.target].append(
                    BuildEdge(
                        source_node=edge.source,
                        source_handle=edge.sourceHandle.removeprefix(f'{edge.source}-out-'),
                        target_node=edge.target,
                        target_handle=edge.targetHandle.removeprefix(f'{edge.target}-in-'),
                    )
                )
            display_ids = {
                key.removeprefix('node:'): loc.node_id
                for key, loc in source_map.items()
                if key.startswith('node:') and loc.node_id is not None
            }
            reported_code_sources = {
                (d.node_id, tuple(d.subgraph_path), d.message) for d in result.diagnostics if d.code == 'code.source'
            }
            for flat_node in flattened.nodes:
                if flat_node.type not in CODE_NODE_TYPES:
                    continue
                origin = source_map.get('node:' + flat_node.id)
                mapped_path = origin.path if origin else ''
                mapped_node_id = origin.node_id if origin else flat_node.id
                mapped_parents = tuple(origin.subgraph_path) if origin else ()
                for error in _check_code_binding_sources(
                    flat_incoming[flat_node.id],
                    flat_types,
                    flat_incoming,
                    bind_parameters=flat_bind_params,
                    display_ids=display_ids,
                ):
                    key = (mapped_node_id, mapped_parents, error)
                    if key not in reported_code_sources:
                        reported_code_sources.add(key)
                        report('code.source', error, mapped_path, node_id=mapped_node_id, parents=mapped_parents)
            bindings = set()
            for edge in flattened.edges:
                key = (edge.source, edge.sourceHandle, edge.target, edge.targetHandle)
                if key in bindings:
                    origin = source_map.get('edge:' + edge.id)
                    report(
                        'edge.duplicate',
                        'Duplicate parameter binding after subgraph expansion',
                        origin.path if origin else '',
                        edge_id=origin.edge_id if origin else edge.id,
                        parents=tuple(origin.subgraph_path) if origin else (),
                    )
                bindings.add(key)
            executable = {n.id for n in flattened.nodes if n.type not in PRESENTATION_NODE_TYPES}
            order_nodes(
                list(executable),
                [(e.source, e.target) for e in flattened.edges if e.source in executable and e.target in executable],
            )
        except ValidationError as error:
            invalid(error, '')
        except (ValueError, KeyError) as error:
            message = str(error)
            code = 'graph.cycle' if 'contains a cycle' in message else 'graph.structure'
            location, node_id, parents = '', None, ()
            for expanded, origin in sorted(origins.items(), key=lambda pair: len(pair[0]), reverse=True):
                if message.startswith(
                    (
                        f'Node {expanded}:',
                        f'ForEach {expanded}:',
                        f'Subgraph {expanded}:',
                        f'Batch subgraph {expanded}:',
                    )
                ):
                    location, node_id, origin_parents = origin
                    parents = tuple(origin_parents)
                    break
            if not (code == 'graph.cycle' and any(item.code == code for item in result.diagnostics)):
                report(code, message, location, node_id=node_id, parents=parents)
    return result
