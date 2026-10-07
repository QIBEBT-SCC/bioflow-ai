"""Temporary CODE workspaces, candidate validation, and review diffs."""

import difflib
import tempfile
from pathlib import Path

from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.tasks.code import normalize_python_dependencies, normalize_r_dependencies


def create_workspace(node_type: CodeNodeType) -> Path:
    """Create a session-only workspace recording the trusted target CODE node type."""
    workspace = Path(tempfile.mkdtemp(prefix='bioflow-code-workspace-'))
    workspace.chmod(0o777)
    (workspace / 'output').mkdir(mode=0o777)
    agents = workspace / 'AGENTS.md'
    agents.write_text(f'# BioFlowAI CODE authoring workspace\n\nCurrent target: {node_type.value}\n', encoding='utf-8')
    agents.chmod(0o666)
    return workspace


def write_baseline(workspace: Path, node_type: CodeNodeType, source: str, dependencies: list[str]) -> None:
    """Replace only the files controlled by the CODE editor."""
    filename = {CodeNodeType.BASH: 'script.sh', CodeNodeType.PYTHON: 'script.py', CodeNodeType.R: 'script.R'}[node_type]
    script = workspace / filename
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


def read_candidate(
    workspace: Path, node_type: CodeNodeType, *, allow_empty: bool = False
) -> tuple[str, list[str], list[str]]:
    """Read and validate the allowed candidate files, ignoring everything else."""
    filename = {CodeNodeType.BASH: 'script.sh', CodeNodeType.PYTHON: 'script.py', CodeNodeType.R: 'script.R'}[node_type]
    source = _read_allowed_file(workspace, filename)
    if not source.strip() and not allow_empty:
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
    allowed = {'AGENTS.md', filename, 'output'}
    if node_type is not CodeNodeType.BASH:
        allowed.add('dependencies.txt')
    warnings = [f'Ignored extra file: {path.name}' for path in workspace.iterdir() if path.name not in allowed]
    return source, dependencies, warnings


def proposal_diff(
    node_type: CodeNodeType,
    before_source: str,
    after_source: str,
    before_dependencies: list[str],
    after_dependencies: list[str],
) -> str:
    """Render one unified diff for the whole turn."""
    filename = {CodeNodeType.BASH: 'script.sh', CodeNodeType.PYTHON: 'script.py', CodeNodeType.R: 'script.R'}[node_type]
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
    return ''.join(chunk if chunk.endswith('\n') else f'{chunk}\n\\ No newline at end of file\n' for chunk in chunks)
