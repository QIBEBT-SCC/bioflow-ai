"""Shared LLM authoring contracts for reusable workflow CODE nodes."""

from app.schemas.code import CodeNodeType

BASH_CODE_CONTRACT = """## Target Runtime

- The script runs with `/usr/bin/bash`.
- Its current working directory is a newly created writable output directory dedicated to this node invocation.
- Upstream inputs are passed as named command-line options from `bind_param` nodes.
- Standard Linux commands plus `fd`, `rg`, `jq`, `yq`, and `mlr` are available.
- Do not assume Python, R, third-party packages, bioinformatics programs, package installation, or network access.

## Request Eligibility

- Before generating code, determine whether the entire request can be implemented in the runtime described above.
- Reject the request when it requires Python, R, a bioinformatics program, any other unavailable executable or
  third-party package, package installation, or network access.
- Do not replace a requested bioinformatics tool or Python/R program with an improvised Bash implementation.

## Script Requirements

- Start with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Parse every declared input using `getopts`; never rely on argument order.
- Use exactly the short options provided by the authoritative input contract.
- Quote every path and validate all required options and expected file or directory types.
- Write results to the current working directory using deterministic plain file names.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input.
- Avoid writing large data to stdout; redirect results directly to named files.
- Do not invoke Python or R.
- Do not use bioinformatics software; such operations belong in tool nodes.
"""

PYTHON_CODE_CONTRACT = """## Target Runtime

- The script runs as one file through uv in a Python 3.13 container.
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
- Declare only additional Python distribution requirements.
- Do not declare Click.
- Use distribution names rather than import names, such as `PyYAML` instead of `yaml`.
- Do not use URL, Git, or local-path requirements.
- Do not create a virtual environment or invoke pip, uv, Bash, or R from the script.
"""

R_CODE_CONTRACT = """## Target Runtime

- The script runs with `Rscript --vanilla` in the official `r-base:4.4.3` container.
- Its current working directory is the writable node output directory.
- Upstream files and directories are mounted read-only and passed as named command-line options.
- Additional CRAN packages declared for the CODE definition are installed before the script runs.

## CLI and Files

- Parse `commandArgs(trailingOnly = TRUE)` and require every option in the authoritative input contract.
- Use option names exactly as provided, including their leading hyphens; do not rely on argument order.
- Validate that every input path exists before reading it.
- Write every result to the current working directory with a deterministic plain file name.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input.

## Compatibility and Dependencies

- Generate syntax and APIs compatible with R 4.4.3.
- Prefer base and recommended packages when practical.
- Declare only additional CRAN package names; do not include version constraints, base packages, Bioconductor
  packages, GitHub repositories, URLs, or local paths.
- Do not invoke Bash, Python, package installers, or network clients from the script.
"""

CODE_CONTRACTS = {
    CodeNodeType.BASH: BASH_CODE_CONTRACT,
    CodeNodeType.PYTHON: PYTHON_CODE_CONTRACT,
    CodeNodeType.R: R_CODE_CONTRACT,
}

SCRIPT_NAMES = {
    CodeNodeType.BASH: 'script.sh',
    CodeNodeType.PYTHON: 'script.py',
    CodeNodeType.R: 'script.R',
}


def code_node_contract(node_type: CodeNodeType) -> str:
    """Return the authoritative target-runtime contract for one CODE language."""
    return CODE_CONTRACTS[node_type]


def code_agent_workspace_instructions(node_type: CodeNodeType) -> str:
    """Adapt a CODE runtime contract to an ACP file-editing workspace."""
    script_name = SCRIPT_NAMES[node_type]
    editable_files = f'`{script_name}`'
    if node_type is not CodeNodeType.BASH:
        editable_files += ' and `dependencies.txt`'
    dependency_files = (
        '- Do not create `dependencies.txt`; Bash CODE definitions cannot declare runtime dependencies.'
        if node_type is CodeNodeType.BASH
        else (
            '- Put one direct dependency per line in `dependencies.txt`; do not add installer flags or commands.\n'
            '- Leave `dependencies.txt` empty when the script needs no additional dependency.'
        )
    )
    return f"""# BioFlowAI CODE Authoring Workspace

You are editing one reusable `{node_type.value}` workflow CODE definition. BioFlowAI, not the agent, handles catalog
metadata and persistence.

## Authoring Workspace

- `/workspace` is an authoring workspace, not the working directory used when the saved CODE node runs.
- Modify only {editable_files}.
- Leave one complete, directly runnable script in `{script_name}`; do not wrap source code in Markdown fences.
{dependency_files}
- Use `/workspace/output/` only for temporary checks during this editing session. Its contents are never proposed or
  available to the saved CODE node.
- Do not make the saved script read from or write to `/workspace` or `/workspace/output`.
- Do not edit `AGENTS.md` or treat tools installed in the coding-agent container as target-runtime capabilities.

## Input Contract

- Existing command-line options in the script and exact option names stated by the user are authoritative.
- The reusable CODE catalog editor does not automatically receive a workflow graph's `bind_param` input contract.
- Preserve existing options unless the user explicitly asks to change them. Do not silently invent, rename, or remove
  inputs; ask for the exact option names when they are required but unspecified.
- CODE inputs are paths supplied by `bind_param` nodes. Treat them as read-only even during tests.

## Target CODE Node Contract

{code_node_contract(node_type)}
## Delivery

- Apply requested changes through the controlled workspace files; BioFlowAI will build and display the proposal.
- When editing an existing script, preserve behavior outside the requested change.
- If the request cannot be satisfied by this node type and target runtime, explain why instead of producing a fake or
  partial implementation.
"""
