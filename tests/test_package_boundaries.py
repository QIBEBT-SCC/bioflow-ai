import ast
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / 'src' / 'bioflow_ai'
SOURCE = PACKAGE.parent
LEGACY_PACKAGES = {'app', 'llm', 'script'}


def _module_exists(name: str) -> bool:
    path = SOURCE.joinpath(*name.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def test_original_modules_have_destinations() -> None:
    mapping = json.loads((Path(__file__).parent / 'fixtures' / 'module_map.json').read_text())
    baseline = json.loads((Path(__file__).parent / 'fixtures' / 'layout_baseline.json').read_text())
    assert len(mapping) == baseline['python_files']
    for original, destinations in mapping.items():
        assert all(_module_exists(name) for name in destinations), original
    assert all(not (SOURCE.parent / name).exists() for name in LEGACY_PACKAGES)


def test_import_and_presentation_boundaries() -> None:
    for source_file in PACKAGE.rglob('*.py'):
        is_service = source_file.is_relative_to(PACKAGE / 'services')
        is_task = source_file.is_relative_to(PACKAGE / 'tasks')
        for node in ast.walk(ast.parse(source_file.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, source_file
                modules = [node.module or '']
            for name in modules:
                assert name.split('.')[0] not in LEGACY_PACKAGES, (source_file, name)
                if name.startswith('bioflow_ai'):
                    assert _module_exists(name), (source_file, name)
                if is_service or is_task:
                    assert not name.startswith('bioflow_ai.cli'), (source_file, name)
                if is_service:
                    assert name.split('.')[0] not in {'click', 'rich', 'questionary', 'tqdm'}, (source_file, name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Runtime module strings must move too; CODE filenames retain
                # their separate script.py/script.R execution contract.
                assert not node.value.startswith(('app.', 'llm.')), (source_file, node.value)
                if node.value.startswith('script.'):
                    assert node.value in {'script.py', 'script.R', 'script.sh'}, (source_file, node.value)
