import json
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from llm.context import BaseContext
from llm.models import get_coding_model
from llm.prebuild.json_output import json_output
from llm.states import BuildEdge, BuildNode

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

BASH_CODE_SYSTEM = """You are a senior Bash engineer generating code for one workflow code node.
The Human message contains a natural-language requirement and an authoritative input contract.
First assess whether the request is eligible for this Bash-only node. Generate exactly one complete Bash script only
when the entire requested deterministic workflow transformation is eligible.

## Runtime

- The script runs with `/usr/bin/bash`.
- Its current working directory is a newly created writable output directory dedicated to this node invocation.
- Upstream inputs are passed as named command-line options from `bind_param` nodes.
- Standard Linux commands plus `fd`, `rg`, `jq`, `yq`, and `mlr` are available.
- Do not assume Python, R, third-party packages, bioinformatics programs, package installation, or network access.

## Request Eligibility

- Before generating code, determine whether the entire request can be implemented in the runtime described above.
- Set `reject` to `true` when the request requires Python, R, a bioinformatics program, any other unavailable
  executable or third-party package, package installation, or network access.
- Do not replace a requested bioinformatics tool or Python/R program with an improvised Bash implementation.

## Script Requirements

- Start with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Parse every declared input using `getopts`; never rely on argument order.
- Use exactly the short options provided in the authoritative input contract.
- Quote every path and validate all required options and expected file or directory types.
- Write results to the current working directory using deterministic plain file names.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input.
- Avoid writing large data to stdout; redirect results directly to named files.
- Do not invoke Python or R.
- Do not use bioinformatics software; such operations belong in tool nodes.

## Structured Result

- Always return the eligibility decision in `reject`.
- When `reject` is `false`, return raw script content in `code`, without Markdown fences.
- When `reject` is `true`, return an empty string in `code`.
"""

PYTHON_CODE_SYSTEM = """You are a senior Python engineer generating code for one workflow code node.
The Human message contains a natural-language requirement and an authoritative input contract.
Generate exactly one complete Python 3.13 script for the requested bounded, deterministic workflow transformation.

## Runtime

- The script runs as one file through uv in a container.
- Its current working directory is the writable node output directory.
- Upstream files and directories are mounted read-only and passed as named command-line options.
- Click is automatically provided by the runtime.
- The backend adds PEP 723 metadata; do not add it yourself.

## CLI and Files

- Define one required Click option for every input in the authoritative input contract.
- Use option names exactly as provided, including the leading `--`.
- Validate paths with `click.Path(exists=True, path_type=Path)`.
- Use `pathlib.Path` for filesystem operations.
- Write every result to the current working directory with a deterministic plain file name.
- Never modify or write beside upstream inputs.
- Include the standard `if __name__ == '__main__': main()` entry point.

## Dependencies

- Prefer the standard library when it keeps the implementation clear.
- List only additional Python distribution requirements in `dependencies`.
- Do not list Click.
- Use distribution names rather than import names, such as `PyYAML` instead of `yaml`.
- Do not use URL, Git, or local-path requirements.
- Do not create a virtual environment or invoke pip, uv, Bash, or R from the script.

## Structured Result

- Return raw Python source in `code`, without Markdown fences.
- Return additional Python distribution requirements in `dependencies`; use an empty list when none are needed.
"""


class CodeInput(BaseModel):
    """One authoritative CLI input passed to a generated code node."""

    parameter: str = Field(description='Exact CLI option supplied by the bind_param node')
    source_node: str = Field(description='Upstream workflow node name')
    source_handle: str = Field(description='Upstream output handle name')
    description: str = Field(description='Description of the upstream output')


class BashCodeResult(BaseModel):
    """Structured result produced for a Bash code node."""

    code: str = Field(description='Complete Bash script')
    reject: bool = Field(description='Whether to reject this code generation request', default=False)


class PythonCodeResult(BaseModel):
    """Structured result produced for a Python code node."""

    code: str = Field(description='Complete Python 3.13 script')
    dependencies: list[str] = Field(
        default_factory=list,
        description='Additional Python distribution requirements; Click must not be included',
    )


def build_code_inputs(
    input_edges: list[BuildEdge],
    exist_nodes: list[BuildNode],
    exist_edges: list[BuildEdge],
) -> list[CodeInput]:
    """Build the authoritative generated-code input contract from workflow state."""
    nodes_by_id = {node.id: node for node in exist_nodes}
    inputs: list[CodeInput] = []

    for edge in input_edges:
        bind_node = nodes_by_id[edge.source_node]
        parameter = bind_node.data.get('parameter')
        if not isinstance(parameter, str) or not parameter:
            msg = f'bind_param node "{bind_node.id}" has no valid parameter'
            raise ValueError(msg)

        upstream_edge = next((item for item in exist_edges if item.target_node == bind_node.id), None)
        if upstream_edge is None:
            msg = f'bind_param node "{bind_node.id}" is not connected to an upstream file'
            raise ValueError(msg)
        upstream_node = nodes_by_id.get(upstream_edge.source_node)
        if upstream_node is None:
            msg = f'upstream node "{upstream_edge.source_node}" does not exist'
            raise ValueError(msg)
        source_handle = next(
            (handle for handle in upstream_node.output_handles if handle.name == upstream_edge.source_handle),
            None,
        )
        if source_handle is None:
            msg = (
                f'upstream node "{upstream_node.id}" has no output handle named "{upstream_edge.source_handle}"'
            )
            raise ValueError(msg)

        inputs.append(
            CodeInput(
                parameter=parameter,
                source_node=upstream_node.name,
                source_handle=source_handle.name,
                description=source_handle.description,
            )
        )

    parameters = [item.parameter for item in inputs]
    if len(parameters) != len(set(parameters)):
        msg = f'Code node input parameters must be unique: {parameters}'
        raise ValueError(msg)
    return inputs


def _format_request(request: str, inputs: list[CodeInput]) -> str:
    return (
        f'<requirement>\n{request.strip()}\n</requirement>\n\n'
        f'<input_contract>\n{json.dumps([item.model_dump() for item in inputs], ensure_ascii=False, indent=2)}\n'
        f'</input_contract>'
    )


async def generate_bash_code(request: str, inputs: list[CodeInput], user_id: int) -> BashCodeResult:
    """Generate one structured Bash code-node artifact."""
    model = get_coding_model()
    metadata = cast('LLMMetadata', model.metadata)
    chain = json_output(
        model,
        BASH_CODE_SYSTEM,
        BashCodeResult,
        metadata=metadata,
        name='bash_code_generator',
    )

    return await chain.ainvoke(
        _format_request(request, inputs),
        context=BaseContext(user_id=user_id),
    )


async def generate_python_code(request: str, inputs: list[CodeInput], user_id: int) -> PythonCodeResult:
    """Generate one structured Python code-node artifact."""
    model = get_coding_model()
    metadata = cast('LLMMetadata', model.metadata)
    chain = json_output(
        model,
        PYTHON_CODE_SYSTEM,
        PythonCodeResult,
        metadata=metadata,
        name='python_code_generator',
    )

    return await chain.ainvoke(
        _format_request(request, inputs),
        context=BaseContext(user_id=user_id),
    )
