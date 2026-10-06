"""Temporary CODE workspaces, candidate validation, and review diffs."""

import difflib
import hashlib
import json
import tempfile
from pathlib import Path

from bioflow_ai.llm.utils.code_node_contract import SCRIPT_NAMES, code_workspace_instructions
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.tasks.code import normalize_python_dependencies, normalize_r_dependencies


def create_workspace(node_type: CodeNodeType) -> Path:
    """Create a session-only workspace with CODE authoring instructions."""
    workspace = Path(tempfile.mkdtemp(prefix='bioflow-code-workspace-'))
    workspace.chmod(0o777)
    (workspace / 'output').mkdir(mode=0o777)
    agents = workspace / 'AGENTS.md'
    agents.write_text(code_workspace_instructions(node_type), encoding='utf-8')
    agents.chmod(0o666)
    return workspace


def write_baseline(workspace: Path, node_type: CodeNodeType, source: str, dependencies: list[str]) -> None:
    """Replace only the files controlled by the CODE editor."""
    script = workspace / SCRIPT_NAMES[node_type]
    script.write_text(source, encoding='utf-8')
    script.chmod(0o666)
    dependency_path = workspace / 'dependencies.txt'
    if node_type is CodeNodeType.BASH:
        dependency_path.unlink(missing_ok=True)
    else:
        dependency_path.write_text('\n'.join(dependencies) + ('\n' if dependencies else ''), encoding='utf-8')
        dependency_path.chmod(0o666)


def _read_allowed_file(workspace: Path, filename: str) -> str:
    path = workspace / filename
    if path.is_symlink() or workspace.resolve() not in path.resolve(strict=True).parents:
        message = f'Agent replaced {filename} with an unsafe path'
        raise ValueError(message)
    return path.read_text(encoding='utf-8')


def read_candidate(workspace: Path, node_type: CodeNodeType) -> tuple[str, list[str], list[str]]:
    """Read and validate the allowed candidate files, ignoring everything else."""
    source = _read_allowed_file(workspace, SCRIPT_NAMES[node_type])
    if not source.strip():
        message = 'Agent produced an empty script'
        raise ValueError(message)
    dependencies: list[str] = []
    if node_type is not CodeNodeType.BASH:
        dependency_path = workspace / 'dependencies.txt'
        if dependency_path.exists():
            dependencies = _read_allowed_file(workspace, 'dependencies.txt').splitlines()
        dependencies = (
            normalize_python_dependencies(dependencies)
            if node_type is CodeNodeType.PYTHON
            else normalize_r_dependencies(dependencies)
        )
    allowed = {'AGENTS.md', SCRIPT_NAMES[node_type], 'output'}
    if node_type is not CodeNodeType.BASH:
        allowed.add('dependencies.txt')
    warnings = [f'Ignored extra file: {path.name}' for path in workspace.iterdir() if path.name not in allowed]
    return source, dependencies, warnings


def baseline_hash(node_type: CodeNodeType, source: str, dependencies: list[str]) -> str:
    """Hash the exact editor baseline associated with a proposal."""
    payload = json.dumps(
        {'node_type': node_type.value, 'source': source, 'dependencies': dependencies},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def proposal_diff(
    node_type: CodeNodeType,
    before_source: str,
    after_source: str,
    before_dependencies: list[str],
    after_dependencies: list[str],
) -> str:
    """Render one unified diff for the whole turn."""
    filename = SCRIPT_NAMES[node_type]
    chunks = list(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            after_source.splitlines(keepends=True),
            fromfile=f'a/{filename}',
            tofile=f'b/{filename}',
        )
    )
    if node_type is not CodeNodeType.BASH:
        chunks.extend(
            difflib.unified_diff(
                [f'{item}\n' for item in before_dependencies],
                [f'{item}\n' for item in after_dependencies],
                fromfile='a/dependencies.txt',
                tofile='b/dependencies.txt',
            )
        )
    return ''.join(chunks)
